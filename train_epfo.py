"""Train the EPFO CRNN+CTC reader on synthetic data only.

The EPFO background is a fixed vertical gradient that is byte-identical in every
captcha, so subtracting it recovers the ink exactly (see solver/epfo.py). There
is no noise to model at all; the generator only has to place 5 glyphs.

    python3 train_epfo.py [epochs]
"""
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from solver import epfo as M

SEED = 13
N_TRAIN = 16000
N_VAL = 1500
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 45
BS = 128
LR = 2.5e-3
MODEL_PATH = "solver/epfo_model.pt"
LABELS_JSON = "epfo_labels.json"


def _gen_chunk(args):
    n, seed = args
    rng = np.random.default_rng(seed)
    X = np.empty((n, M.IN_H, M.IN_W), np.uint8)
    Y = np.empty((n, M.LENGTH), np.int64)
    for i in range(n):
        lab = M.random_label(rng)
        X[i] = M.make_input(lab, rng)
        Y[i] = M.encode_label(lab)
    return X, Y


def gen(n, seed, pool=None):
    if pool is None:
        return _gen_chunk((n, seed))
    k = pool._processes
    sizes = [n // k] * k
    for i in range(n - sum(sizes)):
        sizes[i] += 1
    out = pool.map(_gen_chunk, [(s, seed * 1000 + j) for j, s in enumerate(sizes) if s])
    return np.concatenate([o[0] for o in out]), np.concatenate([o[1] for o in out])


def real_set():
    """Labelled reals for *reporting only*, with the finetune test split removed.

    finetune_epfo.py holds out a pinned test set. Those images must not
    influence this run either, or the final number stops being held out — so the
    same seeded split is recomputed here and the test slice dropped.
    """
    if not os.path.exists(LABELS_JSON):
        return [], None, []
    labels = json.load(open(LABELS_JSON))
    items = [(p, l) for p, l in sorted(labels.items())
             if os.path.exists(p) and len(l) == M.LENGTH and all(c in M.CH2I for c in l)]
    if os.path.exists("epfo_split.json"):
        test_p = set(json.load(open("epfo_split.json"))["test"])
    else:
        order = np.random.RandomState(3).permutation(len(items))
        test_p = {items[i][0] for i in order[:44]}
    items = [it for it in items if it[0] not in test_p]
    if not items:
        return [], None, []
    return items, np.stack([M.load_real(p) for p, _ in items]), [l for _, l in items]


def score(preds, labels):
    exact = sum(p == l for p, l in zip(preds, labels))
    chars = sum(sum(a == b for a, b in zip(p, l)) for p, l in zip(preds, labels))
    return exact, chars, len(labels) * M.LENGTH


def main():
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(SEED)
    pool = mp.Pool(max(1, min(8, (os.cpu_count() or 4) - 1)))

    print(f"device={dev}  input={M.IN_W}x{M.IN_H}  classes={M.N_CLASSES}  "
          f"faces={len(M.FONT_FILES)}", flush=True)
    t = time.time()
    Xva, Yva = gen(N_VAL, 999, pool)
    vlabels = ["".join(M.I2CH[i] for i in r) for r in Yva]
    print(f"val generated in {time.time()-t:.0f}s", flush=True)

    items, real_imgs, real_labels = real_set()
    print(f"real eval set: {len(items)} hand-labeled images", flush=True)

    model = M.EpfoCRNN().to(dev)
    with torch.no_grad():
        T = model(torch.zeros(1, 1, M.IN_H, M.IN_W, device=dev)).shape[1]
    print(f"CRNN time steps T={T}", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    il = torch.full((BS,), T, dtype=torch.long)
    tl = torch.full((BS,), M.LENGTH, dtype=torch.long)

    best = (-1, -1)
    for ep in range(1, EPOCHS + 1):
        Xtr, Ytr = gen(N_TRAIN, SEED * 100 + ep, pool)
        Xd = torch.tensor(Xtr, device=dev)[:, None].float() / 255.0
        Yt = torch.tensor(Ytr)

        model.train()
        perm = torch.randperm(len(Xd), device=dev)
        Xs, Ys = Xd[perm], Yt[perm.cpu()]
        run, nb = 0.0, 0
        for i in range(0, len(Xs) - BS + 1, BS):
            opt.zero_grad()
            lp = model(Xs[i:i + BS]).log_softmax(2).permute(1, 0, 2).cpu()
            loss = F.ctc_loss(lp, Ys[i:i + BS], il, tl, blank=M.BLANK, zero_infinity=True)
            loss.backward()
            opt.step()
            run += float(loss.detach())
            nb += 1
        sched.step()
        del Xd, Xs

        vpred, _ = M.predict_greedy(model, Xva, dev)
        vex, vch, vtot = score(vpred, vlabels)
        line = f"epoch {ep:3d}  loss {run/nb:.3f}  synthVAL exact {vex/N_VAL:.3f} char {vch/vtot:.3f}"
        if real_imgs is not None:
            rpred, _ = M.predict_greedy(model, real_imgs, dev)
            rex, rch, rtot = score(rpred, real_labels)
            line += f"  |  REAL exact {rex}/{len(items)} char {rch}/{rtot} ({rch/max(rtot,1)*100:.1f}%)"
            cur = (rch, vch)
        else:
            cur = (vch, 0)
        print(line, flush=True)

        if cur >= best:
            best = cur
            torch.save(model.state_dict(), MODEL_PATH)
    pool.close()
    print(f"\nsaved best -> {MODEL_PATH}   best={best}", flush=True)


if __name__ == "__main__":
    main()
