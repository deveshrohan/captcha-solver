"""Close the synthetic->real gap for the GST reader by self-training.

The synthetic generator reproduces the SimpleCaptcha pipeline exactly except for
one thing that could not be pinned down: the exact font the server's fontconfig
resolves to. That residual shows up as errors concentrated on the fisheye-
magnified middle digits.

Self-training fixes what a better guess at the font would: run the synthetic-
trained model over the *unlabeled* real captchas, keep only high-confidence
reads as pseudo-labels, and fine-tune on a mix of fresh synthetic and those real
images. The hand-labeled evaluation set is excluded from the pseudo-label pool
and is only ever scored, never trained on.

    python3 finetune_gst.py [epochs]
"""
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

from solver import gst as G
from train_gst import gen, score

MODEL_IN = "solver/gst_model.pt"
MODEL_OUT = "solver/gst_model.pt"
LABELS_JSON = "gst_labels.json"
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 12
BS = 128
LR = 3e-4
# The beam confidence is only weakly calibrated here (~88% exact even above
# 0.99), so pseudo-labels carry real noise. 0.98 keeps ~58% of the pool; the
# run only overwrites the checkpoint when the dev half actually improves.
MIN_CONF = 0.98
N_SYNTH = 8000


def main():
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    labels = json.load(open(LABELS_JSON))
    items = [(p, l) for p, l in sorted(labels.items()) if os.path.exists(p)]
    held_out = set(labels)

    # Split the hand-labeled set: choose the checkpoint on DEV, report on TEST.
    # Selecting the epoch on the same images you quote would inflate the number.
    rs = np.random.RandomState(0)
    order = rs.permutation(len(items))
    dev_idx = set(order[: len(items) // 2].tolist())
    dev_items = [it for i, it in enumerate(items) if i in dev_idx]
    test_items = [it for i, it in enumerate(items) if i not in dev_idx]

    dev_imgs = np.stack([G.load_real(p) for p, _ in dev_items])
    dev_labels = [l for _, l in dev_items]
    test_imgs = np.stack([G.load_real(p) for p, _ in test_items])
    test_labels = [l for _, l in test_items]
    print(f"selection split: dev={len(dev_items)}  test={len(test_items)}", flush=True)

    model = G.GstCRNN().to(dev)
    model.load_state_dict(torch.load(MODEL_IN, map_location=dev))

    pool_paths = [p for p in sorted(glob.glob("gst_raw/g*.png")) if p not in held_out]
    print(f"unlabeled pool: {len(pool_paths)} images (labeled set of {len(items)} excluded)", flush=True)

    pool_imgs = np.stack([G.load_real(p) for p in pool_paths])
    preds, confs = G.predict(model, pool_imgs, dev)
    keep = [i for i, (t, c) in enumerate(zip(preds, confs))
            if c >= MIN_CONF and len(t) == G.LENGTH]
    print(f"pseudo-labels kept at conf>={MIN_CONF}: {len(keep)}/{len(pool_paths)}", flush=True)
    if len(keep) < 50:
        print("too few confident reads to adapt; aborting", flush=True)
        return

    Xr = pool_imgs[keep]
    Yr = np.array([G.encode_label(preds[i]) for i in keep], dtype=np.int64)

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    with torch.no_grad():
        T = model(torch.zeros(1, 3, G.IN_H, G.IN_W, device=dev)).shape[1]
    il = torch.full((BS,), T, dtype=torch.long)
    tl = torch.full((BS,), G.LENGTH, dtype=torch.long)

    d0, _ = G.predict(model, dev_imgs, dev)
    de0, dc0, dtot = score(d0, dev_labels)
    t0, _ = G.predict(model, test_imgs, dev)
    te0, tc0, ttot = score(t0, test_labels)
    print(f"before:  dev exact {de0}/{len(dev_items)} digits {dc0}/{dtot}"
          f"  |  TEST exact {te0}/{len(test_items)} digits {tc0}/{ttot} ({tc0/ttot*100:.2f}%)",
          flush=True)

    best = (de0, dc0)
    torch.save(model.state_dict(), MODEL_OUT)
    for ep in range(1, EPOCHS + 1):
        Xs, Ys = gen(N_SYNTH, 4200 + ep)
        X = np.concatenate([Xs, Xr])
        Y = np.concatenate([Ys, Yr])
        Xd = (torch.tensor(X, device=dev).permute(0, 3, 1, 2).contiguous().float() / 255.0)
        Yt = torch.tensor(Y)

        model.train()
        perm = torch.randperm(len(Xd), device=dev)
        Xp, Yp = Xd[perm], Yt[perm.cpu()]
        run, nb = 0.0, 0
        for i in range(0, len(Xp) - BS + 1, BS):
            opt.zero_grad()
            lp = model(Xp[i:i + BS]).log_softmax(2).permute(1, 0, 2).cpu()
            loss = F.ctc_loss(lp, Yp[i:i + BS], il, tl, blank=G.BLANK, zero_infinity=True)
            loss.backward()
            opt.step()
            run += float(loss.detach())
            nb += 1
        sched.step()
        del Xd, Xp

        dp, _ = G.predict(model, dev_imgs, dev)
        dex, dch, dtot = score(dp, dev_labels)
        tp, _ = G.predict(model, test_imgs, dev)
        tex, tch, ttot = score(tp, test_labels)
        star = ""
        if (dex, dch) >= best:                     # selection uses DEV only
            best = (dex, dch)
            torch.save(model.state_dict(), MODEL_OUT)
            star = "  *saved"
        print(f"epoch {ep:2d}  loss {run/nb:.3f}  dev exact {dex}/{len(dev_items)} "
              f"digits {dch}/{dtot}  |  TEST exact {tex}/{len(test_items)} "
              f"digits {tch}/{ttot} ({tch/ttot*100:.2f}%){star}", flush=True)

    model.load_state_dict(torch.load(MODEL_OUT, map_location=dev))
    fp, _ = G.predict(model, test_imgs, dev)
    fex, fch, ftot = score(fp, test_labels)
    ap, _ = G.predict(model, np.concatenate([dev_imgs, test_imgs]), dev)
    aex, ach, atot = score(ap, dev_labels + test_labels)
    print(f"\nselected checkpoint -> TEST exact {fex}/{len(test_items)} "
          f"digits {fch}/{ftot} ({fch/ftot*100:.2f}%)", flush=True)
    print(f"                       all 98 exact {aex}/98 digits {ach}/{atot} "
          f"({ach/atot*100:.2f}%)", flush=True)


if __name__ == "__main__":
    main()
