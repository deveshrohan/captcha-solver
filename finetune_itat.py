"""Close the synthetic->real gap for the ITAT reader using hand-labeled reals.

The synthetic-only model reads a DejaVu-Condensed rendering; the real face is a
thinner, unidentified sans, so there is a weight/letterform domain gap the same
shape as MCA's. Because ITAT's ink is colour-separable (the darkness projection),
the reals are cheap to read, so this trains on real labels directly, mixing fresh
synthetic in each epoch so the model does not forget the classes the small real
set under-samples.

Split (pinned in itat_split.json so later label additions cannot reshuffle it):
  * train — real labels (oversampled) mixed with fresh synthetic each epoch
  * dev   — chooses the epoch
  * test  — never trained on, never selected on; the number to quote

    python3 finetune_itat.py [epochs]
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

from solver import itat as I
from train_itat import gen, score


def augment(X, rng):
    """Random small affine jitter on a batch of (N, IN_H, IN_W) uint8 inputs.

    With only ~60 real images, oversampling them verbatim (REAL_REPEAT) just
    teaches the model to memorise those exact pixels — train hits 100% while the
    held-out set stalls at ~70%. Showing each copy under a small translation /
    scale / rotation instead turns 60 images into many, which is the standard
    fix for that overfit and the cheapest lever short of labelling more."""
    n = len(X)
    t = torch.tensor(X, dtype=torch.float32)[:, None] / 255.0
    ang = torch.tensor(rng.uniform(-4, 4, n) * np.pi / 180, dtype=torch.float32)
    sc = torch.tensor(rng.uniform(0.92, 1.08, n), dtype=torch.float32)
    tx = torch.tensor(rng.uniform(-0.05, 0.05, n), dtype=torch.float32)
    ty = torch.tensor(rng.uniform(-0.06, 0.06, n), dtype=torch.float32)
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    theta = torch.zeros(n, 2, 3)
    theta[:, 0, 0] = cos; theta[:, 0, 1] = -sin; theta[:, 0, 2] = tx
    theta[:, 1, 0] = sin; theta[:, 1, 1] = cos; theta[:, 1, 2] = ty
    grid = F.affine_grid(theta, t.shape, align_corners=False)
    out = F.grid_sample(t, grid, align_corners=False, padding_mode="zeros")
    return (out[:, 0].clamp(0, 1) * 255.0).round().to(torch.uint8).numpy()

MODEL_IN = "solver/itat_model.pt"
MODEL_OUT = "solver/itat_model.pt"
LABELS = "itat_labels.json"
SPLIT_JSON = "itat_split.json"
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 30
BS = 128
LR = 5e-4
# The synthetic base is a poor match for the real thin face (22% char on reals),
# so real is weighted heavily — but synthetic is kept for class coverage (31 real
# images under-sample a 32-class alphabet). Dev selects the epoch, guarding the
# overfitting that oversampling 31 images invites.
N_SYNTH = 3000
REAL_REPEAT = 45
SEED = 3


def main():
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(SEED)

    labels = json.load(open(LABELS))
    items = [(p, l.upper()) for p, l in sorted(labels.items())
             if os.path.exists(p) and len(l) == I.LENGTH and all(c in I.CH2I for c in l.upper())]

    if os.path.exists(SPLIT_JSON):
        sp = json.load(open(SPLIT_JSON))
        test_p, dev_p = set(sp["test"]), set(sp["dev"])
    else:
        rs = np.random.RandomState(SEED)
        order = rs.permutation(len(items))
        ntest = max(1, len(items) // 3)
        test_p = {items[i][0] for i in order[:ntest]}
        dev_p = {items[i][0] for i in order[ntest:ntest + ntest // 2]}

    split = {"train": [], "dev": [], "test": []}
    for p, l in items:
        split["test" if p in test_p else "dev" if p in dev_p else "train"].append((p, l))
    print("labels: %d  ->  train %d / dev %d / test %d"
          % (len(items), len(split["train"]), len(split["dev"]), len(split["test"])), flush=True)

    def pack(rows):
        if not rows:
            return np.empty((0, I.IN_H, I.IN_W), np.uint8), np.empty((0, I.LENGTH), np.int64), []
        X = np.stack([I.load_real(p) for p, _ in rows])
        Y = np.array([I.encode_label(l) for _, l in rows], dtype=np.int64)
        return X, Y, [l for _, l in rows]

    Xtr_r, Ytr_r, _ = pack(split["train"])
    Xdev, _, Ldev = pack(split["dev"])
    Xte, _, Lte = pack(split["test"])

    model = I.ItatCRNN().to(dev)
    model.load_state_dict(torch.load(MODEL_IN, map_location=dev))
    with torch.no_grad():
        T = model(torch.zeros(1, 1, I.IN_H, I.IN_W, device=dev)).shape[1]

    def evaluate(X, L):
        # eval on CPU — the just-trained MPS model mis-reads in place (see the
        # note in train_itat.py); CPU eval is correct and matches api.py.
        if len(L) == 0:
            return 0, 0, 0
        was = next(model.parameters()).device
        model.to("cpu")
        p, _ = I.predict_greedy(model, X, "cpu")
        model.to(was)
        return score(p, L)

    de, dc, dt = evaluate(Xdev, Ldev)
    te, tc, tt = evaluate(Xte, Lte)
    print("before:  dev %d/%d exact %d/%d chars  |  TEST %d/%d exact %d/%d chars (%.2f%%)"
          % (de, len(Ldev), dc, dt, te, len(Lte), tc, tt, 100 * tc / max(tt, 1)), flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    il = torch.full((BS,), T, dtype=torch.long)
    tl = torch.full((BS,), I.LENGTH, dtype=torch.long)

    best = (de, dc)
    torch.save(model.state_dict(), MODEL_OUT)
    for ep in range(1, EPOCHS + 1):
        rng = np.random.default_rng(SEED * 1000 + ep)
        Xs, Ys = gen(N_SYNTH, SEED * 1000 + ep)
        if len(Xtr_r):
            reps = augment(np.tile(Xtr_r, (REAL_REPEAT, 1, 1)), rng)   # jittered copies
            repsY = np.tile(Ytr_r, (REAL_REPEAT, 1))
            X = np.concatenate([Xs, reps])
            Y = np.concatenate([Ys, repsY])
        else:
            X, Y = Xs, Ys
        Xd = torch.tensor(X)[:, None].float() / 255.0
        Yt = torch.tensor(Y)

        model.train()
        perm = torch.randperm(len(Xd))
        run, nb = 0.0, 0
        for i in range(0, len(perm) - BS + 1, BS):
            idx = perm[i:i + BS]
            opt.zero_grad()
            lp = model(Xd[idx].to(dev)).log_softmax(2).permute(1, 0, 2).cpu()
            loss = F.ctc_loss(lp, Yt[idx], il, tl, blank=I.BLANK, zero_infinity=True)
            loss.backward()
            opt.step()
            run += float(loss.detach())
            nb += 1
        sched.step()
        del Xd

        de, dc, dt = evaluate(Xdev, Ldev)
        te, tc, tt = evaluate(Xte, Lte)
        star = ""
        if (de, dc) >= best:
            best = (de, dc)
            torch.save(model.state_dict(), MODEL_OUT)
            star = "  *saved"
        print("epoch %2d  loss %.3f  dev %d/%d exact %d/%d chars  |  TEST %d/%d exact "
              "%d/%d chars (%.2f%%)%s"
              % (ep, run / nb, de, len(Ldev), dc, dt, te, len(Lte), tc, tt,
                 100 * tc / max(tt, 1), star), flush=True)

    model.load_state_dict(torch.load(MODEL_OUT, map_location=dev))
    te, tc, tt = evaluate(Xte, Lte)
    print("\nselected checkpoint -> TEST %d/%d exact (%.1f%%), %d/%d chars (%.2f%%)"
          % (te, len(Lte), 100 * te / max(len(Lte), 1), tc, tt, 100 * tc / max(tt, 1)), flush=True)


if __name__ == "__main__":
    main()
