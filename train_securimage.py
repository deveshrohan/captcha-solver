"""Train the securimage CRNN+CTC model on synthetic data.

Why CRNN+CTC (not the flatten+fixed-head CNN): with wave/jitter the characters
are not at fixed positions, so a flatten+multi-head model only *memorizes*
(train acc rises, held-out val stays at chance). CTC slides one classifier
across the image width and is translation-invariant, so it generalizes to
unseen (and real) captchas. CTC loss is unimplemented on MPS, so we run the
network on MPS and compute the CTC loss on CPU (gradients flow back).

Validates each epoch on held-out synthetic AND the hand-labeled real samples.
"""
import glob
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from solver import securimage as S

SEED = 7
N_TRAIN = 14000
N_VAL = 1000
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 70
BS = 128
LR = 2.5e-3
MODEL_PATH = "solver/securimage_model.pt"


def gen(n, rng):
    X = np.empty((n, S.IN_H, S.IN_W), np.uint8)
    Y = np.empty((n, S.LENGTH), np.int64)
    for i in range(n):
        lab = S.random_label(rng)
        X[i] = S.make_image(lab, rng)
        Y[i] = S.encode_label(lab)
    return X, Y


def real_set():
    items = []
    for p in sorted(glob.glob("captcha_2/real_test/*.png")):
        lab = os.path.splitext(os.path.basename(p))[0]
        if len(lab) == S.LENGTH and all(c in S.CH2I for c in lab):
            items.append((p, lab))
    return items


def eval_strings(preds, labels):
    char_ok = sum(sum(a == b for a, b in zip(p, l)) for p, l in zip(preds, labels))
    exact = sum(p == l for p, l in zip(preds, labels))
    return exact, char_ok, len(labels) * S.LENGTH


def main():
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    print(f"device={dev} input={S.IN_W}x{S.IN_H} generating {N_TRAIN}+{N_VAL} ...", flush=True)
    t = time.time()
    Xtr, Ytr = gen(N_TRAIN, rng)
    Xva, Yva = gen(N_VAL, rng)
    vlabels = ["".join(S.I2CH[i] for i in row) for row in Yva]
    print(f"generated in {time.time()-t:.0f}s", flush=True)

    Xtr_d = torch.tensor(Xtr, device=dev)[:, None].float() / 255.0
    Ytr_t = torch.tensor(Ytr)                         # cpu targets for CTC
    model = S.SecurimageCRNN().to(dev)
    with torch.no_grad():                # derive CRNN time steps from a dummy forward
        T = model(torch.zeros(1, 1, S.IN_H, S.IN_W, device=dev)).shape[1]
    print(f"CRNN time steps T={T}", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    items = real_set()
    print(f"real eval set: {len(items)} images", flush=True)
    real_imgs = np.stack([S.load_real(p) for p, _ in items]) if items else None
    real_labels = [lab for _, lab in items]

    il = torch.full((BS,), T, dtype=torch.long)
    tl = torch.full((BS,), S.LENGTH, dtype=torch.long)
    best = (-1, -1.0)
    for ep in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(N_TRAIN, device=dev)
        Xs = Xtr_d[perm]; Ys = Ytr_t[perm.cpu()]
        run = 0.0; nb = 0
        for i in range(0, N_TRAIN - BS + 1, BS):
            xb = Xs[i:i + BS]; yb = Ys[i:i + BS]
            opt.zero_grad()
            out = model(xb)
            lp = out.log_softmax(2).permute(1, 0, 2).cpu()   # [T,B,C] on cpu for CTC
            loss = F.ctc_loss(lp, yb, il, tl, blank=S.BLANK, zero_infinity=True)
            loss.backward(); opt.step()
            run += float(loss.detach()); nb += 1
        sched.step()
        # synthetic held-out
        vpred, _ = S.predict_ctc(model, Xva, dev)
        vex, vch, vtot = eval_strings(vpred, vlabels)
        # real
        if real_imgs is not None:
            rpred, _ = S.predict_ctc(model, real_imgs, dev)
            rex, rch, rtot = eval_strings(rpred, real_labels)
        else:
            rex = rch = rtot = 0
        print(f"epoch {ep:3d}  loss {run/nb:.3f}  synthVAL exact {vex/N_VAL:.3f} char {vch/vtot:.3f}  |  "
              f"REAL exact {rex}/{len(items)} char {rch}/{rtot} ({rch/max(rtot,1)*100:.0f}%)", flush=True)
        score = (rch, vch)
        if score >= best:
            best = score
            torch.save(model.state_dict(), MODEL_PATH)
    print(f"\nsaved best -> {MODEL_PATH}  (real char {best[0]}, synthVAL char {best[1]:.3f})", flush=True)


if __name__ == "__main__":
    main()
