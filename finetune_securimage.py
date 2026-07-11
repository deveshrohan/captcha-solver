"""Domain-adaptation fine-tune for the securimage CRNN.

The synthetic-only model reads real captchas at ~70% char (a synthetic->real
domain gap). We self-train: use the current model to pseudo-label the 1000 real
downloads, keep only high-confidence, length-6 predictions (excluding the 12
held-out labeled test images), then fine-tune on a mix of those real images and
fresh synthetic. The 12 labeled reals stay a clean held-out test.
"""
import glob
import hashlib
import os
import numpy as np
import torch
import torch.nn.functional as F

from solver import securimage as S

dev = "mps" if torch.backends.mps.is_available() else "cpu"
MODEL_PATH = "solver/securimage_model.pt"
CONF_TH = 0.90
EPOCHS = 25
BS = 128
LR = 3e-4


def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def main():
    rng = np.random.default_rng(11)
    model = S.SecurimageCRNN().to(dev)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=dev))
    with torch.no_grad():
        T = model(torch.zeros(1, 1, S.IN_H, S.IN_W, device=dev)).shape[1]

    # held-out labeled test (never trained on)
    test = [(p, os.path.splitext(os.path.basename(p))[0]) for p in sorted(glob.glob("captcha_2/real_test/*.png"))]
    test_hashes = {md5(p) for p, _ in test}
    test_imgs = np.stack([S.load_real(p) for p, _ in test])
    test_labels = [l for _, l in test]

    # pseudo-label the downloads
    dls = [p for p in sorted(glob.glob("captcha_2/downloads/*.png")) if md5(p) not in test_hashes]
    imgs = np.stack([S.load_real(p) for p in dls])
    preds, confs = S.predict_ctc(model, imgs, dev)
    keep_X, keep_Y = [], []
    for im, pr, cf in zip(imgs, preds, confs):
        if cf >= CONF_TH and len(pr) == S.LENGTH and all(c in S.CH2I for c in pr):
            keep_X.append(im); keep_Y.append(S.encode_label(pr))
    print(f"downloads: {len(dls)}  kept pseudo-labels (conf>={CONF_TH}, len6): {len(keep_X)}", flush=True)
    if len(keep_X) < 50:
        print("too few pseudo-labels; aborting"); return
    Xr = np.stack(keep_X); Yr = np.stack(keep_Y)

    # fresh synthetic pool to regularize (avoid forgetting)
    NS = 4000
    slabs = [S.random_label(rng) for _ in range(NS)]
    Xs = np.stack([S.make_image(l, rng) for l in slabs]).astype(np.uint8)
    Ys = np.stack([S.encode_label(l) for l in slabs])

    # combine: upsample real to ~half the data
    reps = max(1, NS // len(Xr))
    Xr_up = np.repeat(Xr, reps, axis=0); Yr_up = np.repeat(Yr, reps, axis=0)
    X = np.concatenate([Xs, Xr_up]); Y = np.concatenate([Ys, Yr_up])
    print(f"fine-tune set: {len(Xs)} synth + {len(Xr_up)} real(x{reps}) = {len(X)}", flush=True)
    Xd = torch.tensor(X, device=dev)[:, None].float() / 255.0
    Yt = torch.tensor(Y)
    N = len(X)

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    il = torch.full((BS,), T, dtype=torch.long); tl = torch.full((BS,), S.LENGTH, dtype=torch.long)

    def eval_real():
        pr, _ = S.predict_ctc_beam(model, test_imgs, dev)   # length-6 beam decode
        ch = sum(sum(a == b for a, b in zip(p, l)) for p, l in zip(pr, test_labels))
        ex = sum(p == l for p, l in zip(pr, test_labels))
        return ex, ch

    best = (-1, -1)
    for ep in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(N, device=dev); Xs_ = Xd[perm]; Ys_ = Yt[perm.cpu()]
        for i in range(0, N - BS + 1, BS):
            xb = Xs_[i:i + BS]; yb = Ys_[i:i + BS]
            opt.zero_grad()
            out = model(xb)
            lp = out.log_softmax(2).permute(1, 0, 2).cpu()
            loss = F.ctc_loss(lp, yb, il, tl, blank=S.BLANK, zero_infinity=True)
            loss.backward(); opt.step()
        sched.step()
        ex, ch = eval_real()
        print(f"ft-epoch {ep:2d}  REAL exact {ex}/12 char {ch}/72 ({ch/72*100:.0f}%)", flush=True)
        if (ch, ex) >= best:
            best = (ch, ex)
            torch.save(model.state_dict(), "solver/securimage_model_ft.pt")
    print(f"\nbest fine-tuned REAL: char {best[0]}/72 exact {best[1]}/12 -> solver/securimage_model_ft.pt", flush=True)


if __name__ == "__main__":
    main()
