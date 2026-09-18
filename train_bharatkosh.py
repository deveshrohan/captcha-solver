"""Train the Bharatkosh CRNN+CTC reader on synthetic data only.

    python3 train_bharatkosh.py [epochs]

The generator (solver/bharatkosh.py) draws specks, then six independently
styled glyphs, then the two fixed bars, and the input zeroes the bar rows for
real and synthetic alike. A FRESH pool is generated every epoch: the renderer is
~3.5ms/image, so 30k images cost a few seconds across the worker pool, and with
62 classes x faces x sizes x rotations there is no reason to let the model see
any image twice.

Checkpoints are selected on the `dev` groups of bharatkosh_split.json. The
`test` groups are never loaded here. No hand-labelled image is ever TRAINED on,
by this script or by finetune_bharatkosh.py -- every hand label is held out.
"""
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from solver import bharatkosh as B

SEED = 23
N_EPOCH_POOL = 30000
N_VAL = 2000
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 40
BS = 128
LR = 2e-3
MODEL_PATH = "solver/bharatkosh_model.pt"
RAW = "bharatkosh_raw"
LABELS_JSON = "bharatkosh_labels.json"
SPLIT_JSON = "bharatkosh_split.json"
RENDERS = 5


def _gen_chunk(args):
    n, seed = args
    rng = np.random.default_rng(seed)
    X = np.empty((n, B.IN_H, B.IN_W), np.uint8)
    Y = np.empty((n, B.LENGTH), np.int64)
    for i in range(n):
        lab = B.random_label(rng)
        X[i] = B.make_input(lab, rng)
        Y[i] = B.encode_label(lab)
    return X, Y


def gen(n, seed, pool):
    k = pool._processes
    sizes = [n // k + (1 if i < n % k else 0) for i in range(k)]
    out = pool.map(_gen_chunk, [(s, seed * 1000 + j) for j, s in enumerate(sizes) if s])
    return np.concatenate([o[0] for o in out]), np.concatenate([o[1] for o in out])


def dev_set():
    """Every render of every `dev` group -> (images, labels)."""
    if not (os.path.exists(LABELS_JSON) and os.path.exists(SPLIT_JSON)):
        return None, []
    labels = json.load(open(LABELS_JSON))
    X, L = [], []
    for g in json.load(open(SPLIT_JSON))["dev"]:
        for r in range(RENDERS):
            p = os.path.join(RAW, "%s_r%d.png" % (g, r))
            if os.path.exists(p):
                X.append(B.load_real(p))
                L.append(labels[g])
    return (np.stack(X) if X else None), L


def score(preds, labels):
    exact = sum(p == l for p, l in zip(preds, labels))
    chars = sum(sum(a == b for a, b in zip(p, l)) for p, l in zip(preds, labels))
    return exact, chars, len(labels) * B.LENGTH


def main():
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(SEED)
    pool = mp.Pool(max(1, min(8, (os.cpu_count() or 4) - 1)))
    print(f"device={dev} input={B.IN_W}x{B.IN_H} classes={B.N_CLASSES}", flush=True)

    Xva, Yva = gen(N_VAL, 999, pool)
    vlabels = ["".join(B.I2CH[i] for i in r) for r in Yva]
    Xdev, dlabels = dev_set()
    print(f"dev set: {len(dlabels)} real renders (selection only)", flush=True)

    model = B.BharatkoshCRNN().to(dev)
    with torch.no_grad():
        T = model(torch.zeros(1, 1, B.IN_H, B.IN_W, device=dev)).shape[1]
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR, total_steps=EPOCHS * (N_EPOCH_POOL // BS), pct_start=0.1)
    il = torch.full((BS,), T, dtype=torch.long)
    tl = torch.full((BS,), B.LENGTH, dtype=torch.long)

    best = (-1, -1)
    for ep in range(1, EPOCHS + 1):
        t0 = time.time()
        Xp, Yp = gen(N_EPOCH_POOL, SEED + ep, pool)     # uint8, fresh each epoch
        Xall, Yall = torch.from_numpy(Xp), torch.tensor(Yp)
        model.train()
        perm = torch.randperm(len(Xall))
        run, nb = 0.0, 0
        for i in range(0, len(perm) - BS + 1, BS):
            idx = perm[i:i + BS]
            opt.zero_grad()
            xb = (Xall[idx].to(dev).float() / 255.0).unsqueeze(1)
            lp = model(xb).log_softmax(2).permute(1, 0, 2).cpu()
            loss = F.ctc_loss(lp, Yall[idx], il, tl, blank=B.BLANK, zero_infinity=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            run += float(loss.detach())
            nb += 1

        # Evaluate on CPU: a live just-trained model on MPS returns garbage (the
        # gremlin train_itat.py documents), which silently freezes "best".
        model.to("cpu")
        vex, vch, vtot = score(B.greedy(B.log_probs(model, Xva)), vlabels)
        line = (f"epoch {ep:3d} loss {run/nb:.3f} synthVAL exact {vex/N_VAL:.3f} "
                f"char {vch/vtot:.3f}")
        cur = (vch, 0)
        if Xdev is not None:
            dex, dch, dtot = score(B.greedy(B.log_probs(model, Xdev)), dlabels)
            line += f" | DEV exact {dex}/{len(dlabels)} char {dch}/{dtot} ({dch/dtot*100:.1f}%)"
            cur = (dch, vch)
        model.to(dev)
        print(line + f"  [{time.time()-t0:.0f}s]", flush=True)
        if cur >= best:
            best = cur
            torch.save(model.state_dict(), MODEL_PATH)
    pool.close()
    print(f"saved best -> {MODEL_PATH} best={best}", flush=True)


if __name__ == "__main__":
    main()
