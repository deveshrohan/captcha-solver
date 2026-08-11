"""Train the ITAT CRNN+CTC reader on synthetic data only.

The real captcha draws light-blue noise lines OVER near-black antialiased text, so
a darkness projection (`1 - max(r,g,b)/255`, see solver/itat.py) separates ink from
noise by colour; the model reads that projection. The generator reproduces the same
projection over its own render — black text under blue lines — so train and test see
the same fragmentation. Geometry is pinned to the measured real bbox; the residual
gap is the letterform weight (DejaVu is denser than the real thin face), which
finetune_itat.py closes with real labels.

    python3 train_itat.py [epochs]

A large synthetic pool is generated ONCE and reused across epochs (the renderer is
~8ms/image, so regenerating every epoch would dominate); the pool is big enough that
the model does not meaningfully memorise it over the epoch count, and the val set is
generated separately.
"""
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from solver import itat as I

SEED = 17
N_POOL = 24000
N_VAL = 2000
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 40
BS = 128
LR = 2.5e-3
MODEL_PATH = "solver/itat_model.pt"
LABELS_JSON = "itat_labels.json"
SPLIT_JSON = "itat_split.json"


def _gen_chunk(args):
    n, seed = args
    rng = np.random.default_rng(seed)
    X = np.empty((n, I.IN_H, I.IN_W), np.uint8)
    Y = np.empty((n, I.LENGTH), np.int64)
    for i in range(n):
        lab = I.random_label(rng)
        X[i] = I.make_input(lab, rng)
        Y[i] = I.encode_label(lab)
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
    """Labelled reals for *reporting only*, with the finetune test split removed
    (so the number train_itat prints stays genuinely held out even after a
    later fine-tune trains on part of the corpus)."""
    if not os.path.exists(LABELS_JSON):
        return [], None, []
    labels = json.load(open(LABELS_JSON))
    items = [(p, l) for p, l in sorted(labels.items())
             if os.path.exists(p) and len(l) == I.LENGTH and all(c in I.CH2I for c in l)]
    if os.path.exists(SPLIT_JSON):
        test_p = set(json.load(open(SPLIT_JSON)).get("test", []))
        items = [it for it in items if it[0] not in test_p]
    if not items:
        return [], None, []
    return items, np.stack([I.load_real(p) for p, _ in items]), [l for _, l in items]


def score(preds, labels):
    exact = sum(p == l for p, l in zip(preds, labels))
    chars = sum(sum(a == b for a, b in zip(p, l)) for p, l in zip(preds, labels))
    return exact, chars, len(labels) * I.LENGTH


def main():
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(SEED)
    pool = mp.Pool(max(1, min(8, (os.cpu_count() or 4) - 1)))

    print(f"device={dev}  input={I.IN_W}x{I.IN_H}  classes={I.N_CLASSES}  "
          f"charset={I.CHARSET}", flush=True)
    t = time.time()
    Xpool, Ypool = gen(N_POOL, SEED, pool)
    Xva, Yva = gen(N_VAL, 999, pool)
    vlabels = ["".join(I.I2CH[i] for i in r) for r in Yva]
    print(f"generated pool={N_POOL} val={N_VAL} in {time.time()-t:.0f}s", flush=True)

    items, real_imgs, real_labels = real_set()
    print(f"real eval set: {len(items)} hand-labeled images", flush=True)

    model = I.ItatCRNN().to(dev)
    with torch.no_grad():
        T = model(torch.zeros(1, 1, I.IN_H, I.IN_W, device=dev)).shape[1]
    print(f"CRNN time steps T={T}", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    il = torch.full((BS,), T, dtype=torch.long)
    tl = torch.full((BS,), I.LENGTH, dtype=torch.long)

    # Keep the pool as uint8 on CPU and convert each batch to float just in time.
    # Materialising the whole pool as a float tensor is ~10x larger and was enough
    # to drive the machine into swap; a uint8 pool + per-batch float is frugal.
    Xall = torch.from_numpy(Xpool)                # uint8 (N, IN_H, IN_W)
    Yall = torch.tensor(Ypool)

    best = (-1, -1)
    for ep in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(len(Xall))
        run, nb = 0.0, 0
        for i in range(0, len(perm) - BS + 1, BS):
            idx = perm[i:i + BS]
            opt.zero_grad()
            xb = (Xall[idx].to(dev).float() / 255.0).unsqueeze(1)
            lp = model(xb).log_softmax(2).permute(1, 0, 2).cpu()
            loss = F.ctc_loss(lp, Yall[idx], il, tl, blank=I.BLANK, zero_infinity=True)
            loss.backward()
            opt.step()
            run += float(loss.detach())
            nb += 1
        sched.step()

        # Evaluate on CPU. Evaluating the just-trained model on MPS returns
        # garbage here (a live-graph gremlin: the same weights saved and reloaded
        # read 95%+, but the in-place MPS model right after backward does not),
        # which silently froze the "best" checkpoint at an early epoch. CPU eval
        # is correct and cheap for these sizes, and matches how api.py runs.
        model.to("cpu")
        vpred, _ = I.predict_greedy(model, Xva, "cpu")
        vex, vch, vtot = score(vpred, vlabels)
        line = f"epoch {ep:3d}  loss {run/nb:.3f}  synthVAL exact {vex/N_VAL:.3f} char {vch/vtot:.3f}"
        if real_imgs is not None:
            rpred, _ = I.predict_greedy(model, real_imgs, "cpu")
            rex, rch, rtot = score(rpred, real_labels)
            line += f"  |  REAL exact {rex}/{len(items)} char {rch}/{rtot} ({rch/max(rtot,1)*100:.1f}%)"
            cur = (rch, vch)
        else:
            cur = (vch, 0)
        model.to(dev)
        print(line, flush=True)

        if cur >= best:
            best = cur
            torch.save(model.state_dict(), MODEL_PATH)
    pool.close()
    print(f"\nsaved best -> {MODEL_PATH}   best={best}", flush=True)


if __name__ == "__main__":
    main()
