"""Train the MCA v2 (200x80, Nov 2026) slot classifier.

Each epoch mixes slot windows from the synthetic renderer in solver/mca2.py
with windows from the hand-labelled real TRAIN split, each real window taken
at a random horizontal shift (+-6 px) so the 152 real captchas are not shown
as identical copies. The checkpoint is chosen on the VALIDATION split; the
TEST split (labelled cold, before any model existed) is scored once at the
end and never influences anything.

    python3 train_mca2.py [epochs]
"""
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from solver import mca2 as M

SEED = 23
N_SYNTH = 24000          # images -> 6 windows each
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 12
BS = 256
LR = 2e-3
REAL_REPEAT = 40         # real windows per epoch = 6 * train * REAL_REPEAT
MODEL_PATH = "solver/mca2_model.pt"


def _gen(args):
    n, seed = args
    rng = np.random.default_rng(seed)
    X = np.empty((n * M.LENGTH, M.IN_H, 2 * M.HALF), np.uint8)
    Y = np.empty(n * M.LENGTH, np.int64)
    for i in range(n):
        lab = M.random_label(rng)
        X[i * 6:(i + 1) * 6] = M.slot_windows(M.make_input(lab, rng))
        Y[i * 6:(i + 1) * 6] = M.encode_label(lab)
    return X, Y


def gen(n, seed, pool):
    k = pool._processes
    parts = pool.map(_gen, [(n // k + (j < n % k), seed * 100 + j) for j in range(k)])
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


def pack(paths, labels):
    return np.stack([M.load_real(p) for p in paths]), [labels[p] for p in paths]


def score(model, X, L):
    p, c = M.predict(model, X, "cpu")
    exact = sum(a == b for a, b in zip(p, L))
    chars = sum(sum(x == y for x, y in zip(a, b)) for a, b in zip(p, L))
    return exact, chars, p, c


def main():
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(SEED)
    labels = json.load(open("mca2_labels.json"))
    split = json.load(open("mca2_split.json"))
    Xtr, Ltr = pack(split["train"], labels)
    Xva, Lva = pack(split["validation"], labels)
    print("train %d / validation %d / test %d" % (len(Xtr), len(Xva), len(split["test"])),
          flush=True)
    # real windows at every shift in -6..6, sampled per epoch
    shifts = range(-6, 7)
    Rw = np.stack([np.concatenate([M.slot_windows(x, s) for x in Xtr]) for s in shifts])
    Ry = np.concatenate([M.encode_label(l) for l in Ltr])

    pool = mp.Pool(max(1, min(8, (os.cpu_count() or 4) - 1)))
    t = time.time()
    Sx, Sy = gen(N_SYNTH, SEED, pool)
    pool.close()
    print("synthetic windows %d in %.0fs" % (len(Sx), time.time() - t), flush=True)

    model = M.Mca2SlotCNN().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR, total_steps=EPOCHS * ((len(Sx) + len(Ry) * REAL_REPEAT) // BS + 1))
    best = (-1, -1)
    rng = np.random.default_rng(SEED)
    for ep in range(1, EPOCHS + 1):
        pick = rng.integers(0, len(shifts), size=len(Ry) * REAL_REPEAT)
        ridx = np.tile(np.arange(len(Ry)), REAL_REPEAT)
        X = np.concatenate([Sx, Rw[pick, ridx]])
        Y = np.concatenate([Sy, Ry[ridx]])
        perm = torch.randperm(len(X))
        Xt, Yt = torch.from_numpy(X), torch.from_numpy(Y)
        model.train()
        run = nb = 0
        for i in range(0, len(perm), BS):
            idx = perm[i:i + BS]
            xb = (Xt[idx].to(dev).float() / 255.0)[:, None]
            loss = F.cross_entropy(model(xb), Yt[idx].to(dev), label_smoothing=0.05)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            run += float(loss.detach())
            nb += 1
        model.to("cpu")                        # CPU eval: see train_itat.py
        ve, vc, _, _ = score(model, Xva, Lva)
        star = ""
        if (vc, ve) >= best:
            best = (vc, ve)
            torch.save(model.state_dict(), MODEL_PATH)
            star = "  *saved"
        model.to(dev)
        print("epoch %2d  loss %.3f  validation %d/%d exact, %d/%d chars%s"
              % (ep, run / nb, ve, len(Lva), vc, 6 * len(Lva), star), flush=True)

    model = M.Mca2SlotCNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    Xte, Lte = pack(split["test"], labels)
    te, tc, preds, confs = score(model, Xte, Lte)
    print("\nselected checkpoint -> TEST %d/%d exact (%.1f%%), %d/%d chars (%.1f%%)"
          % (te, len(Lte), 100 * te / len(Lte), tc, 6 * len(Lte), 100 * tc / (6 * len(Lte))))
    for p, l, c, path in zip(preds, Lte, confs, split["test"]):
        if p != l:
            print("  miss %s  label %s  read %s  conf %.2f" % (path, l, p, c))


if __name__ == "__main__":
    main()
