"""Train the GST CRNN+CTC reader on synthetic data only.

Same recipe that took the securimage reader to ~99%: a faithful port of the
generator (see solver/gst.py) plus CTC, which slides one classifier across the
image width and so stays translation-invariant — necessary here because the
fisheye moves every digit off any fixed position.

A *fresh* synthetic training set is generated every epoch (the generator runs at
~860 img/s, so this is nearly free and beats memorising a fixed corpus).

CTC loss is unimplemented on MPS, so the network runs on MPS while the loss is
computed on CPU; gradients flow back normally.

    python3 train_gst.py [epochs]
"""
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from solver import gst as G

SEED = 7
N_TRAIN = 16000
N_VAL = 1500
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 60
BS = 128
LR = 2.5e-3
MODEL_PATH = "solver/gst_model.pt"
LABELS_JSON = "gst_labels.json"


def _gen_chunk(args):
    n, seed = args
    rng = np.random.default_rng(seed)
    X = np.empty((n, G.IN_H, G.IN_W, 3), np.uint8)
    Y = np.empty((n, G.LENGTH), np.int64)
    for i in range(n):
        lab = G.random_label(rng)
        X[i] = G.to_input(G.make_image(lab, rng))
        Y[i] = G.encode_label(lab)
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
    """Hand-labeled real captchas — never trained on, only scored."""
    if not os.path.exists(LABELS_JSON):
        return [], None, []
    labels = json.load(open(LABELS_JSON))
    items = [(p, l) for p, l in sorted(labels.items())
             if os.path.exists(p) and len(l) == G.LENGTH and all(c in G.CH2I for c in l)]
    if not items:
        return [], None, []
    imgs = np.stack([G.load_real(p) for p, _ in items])
    return items, imgs, [l for _, l in items]


def score(preds, labels):
    exact = sum(p == l for p, l in zip(preds, labels))
    chars = sum(sum(a == b for a, b in zip(p, l)) for p, l in zip(preds, labels))
    return exact, chars, len(labels) * G.LENGTH


def main():
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(SEED)
    pool = mp.Pool(max(1, min(8, (os.cpu_count() or 4) - 1)))

    print(f"device={dev}  input={G.IN_W}x{G.IN_H}x3  fonts={[os.path.basename(f) for f in G.FONT_FILES]}", flush=True)
    t = time.time()
    Xva, Yva = gen(N_VAL, 999, pool)
    vlabels = ["".join(G.I2CH[i] for i in r) for r in Yva]
    print(f"val generated in {time.time()-t:.0f}s", flush=True)

    items, real_imgs, real_labels = real_set()
    print(f"real eval set: {len(items)} hand-labeled images", flush=True)

    model = G.GstCRNN().to(dev)
    with torch.no_grad():
        T = model(torch.zeros(1, 3, G.IN_H, G.IN_W, device=dev)).shape[1]
    print(f"CRNN time steps T={T}", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    il = torch.full((BS,), T, dtype=torch.long)
    tl = torch.full((BS,), G.LENGTH, dtype=torch.long)

    best = (-1, -1)
    for ep in range(1, EPOCHS + 1):
        Xtr, Ytr = gen(N_TRAIN, SEED * 100 + ep, pool)
        Xd = (torch.tensor(Xtr, device=dev).permute(0, 3, 1, 2).contiguous().float() / 255.0)
        Yt = torch.tensor(Ytr)

        model.train()
        perm = torch.randperm(len(Xd), device=dev)
        Xs, Ys = Xd[perm], Yt[perm.cpu()]
        run, nb = 0.0, 0
        for i in range(0, len(Xs) - BS + 1, BS):
            opt.zero_grad()
            out = model(Xs[i:i + BS])
            lp = out.log_softmax(2).permute(1, 0, 2).cpu()
            loss = F.ctc_loss(lp, Ys[i:i + BS], il, tl, blank=G.BLANK, zero_infinity=True)
            loss.backward()
            opt.step()
            run += float(loss.detach())
            nb += 1
        sched.step()
        del Xd, Xs

        vpred, _ = G.predict_greedy(model, Xva, dev)
        vex, vch, vtot = score(vpred, vlabels)
        line = (f"epoch {ep:3d}  loss {run/nb:.3f}  synthVAL exact {vex/N_VAL:.3f} char {vch/vtot:.3f}")
        rex = rch = rtot = 0
        if real_imgs is not None:
            rpred, _ = G.predict_greedy(model, real_imgs, dev)
            rex, rch, rtot = score(rpred, real_labels)
            line += f"  |  REAL exact {rex}/{len(items)} char {rch}/{rtot} ({rch/max(rtot,1)*100:.1f}%)"
        print(line, flush=True)

        cur = (rch, vch) if real_imgs is not None else (vch, 0)
        if cur >= best:
            best = cur
            torch.save(model.state_dict(), MODEL_PATH)
    pool.close()
    print(f"\nsaved best -> {MODEL_PATH}   best={best}", flush=True)


if __name__ == "__main__":
    main()
