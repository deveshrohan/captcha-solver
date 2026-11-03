"""Fine-tune the district-courts CRNN on eye-labelled real renders.

    python3 finetune_dcourts.py [epochs]

Starts from train_dcourts.py's synthetic model. Each batch mixes real train
renders with fresh synthetic ones, so the model keeps the renderer's coverage
of the charset. The checkpoint is chosen on the val split, scored per code by
voting over its renders. Codes labelled after the split was frozen
(dcourts_split.json) count as train. The test split is never read here;
eval_dcourts.py scores it.
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

from download_dcourts import load_manifest
from solver import dcourts as D
from train_dcourts import _gen

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 30
BS = 64
REAL_PER_BATCH = 32
LR = 3e-4
MODEL = "solver/dcourts_model.pt"


def codes(split_name):
    L = json.load(open("dcourts_labels.json"))
    S = json.load(open("dcourts_split.json"))
    if split_name == "train":
        frozen = set(S["test"]) | set(S["val"])
        return {k: v for k, v in L.items() if k not in frozen}
    return {k: L[k] for k in S[split_name]}


def renders(cs, man):
    X, Y, G = [], [], []
    for k, lab in sorted(cs.items()):
        for name in man[k]["renders"]:
            X.append(np.rint(D.to_input(D.load("dcourts_raw/" + name)) * 255).astype(np.uint8))
            Y.append([D.C2I[c] for c in lab])
            G.append(k)
    return np.stack(X), np.array(Y, np.int64), G


def augment(x, rng):
    """Small shifts only: the renders already vary warp, noise and lines."""
    dy, dx = rng.integers(-2, 3), rng.integers(-3, 4)
    return np.roll(np.roll(x, dy, 0), dx, 1)


def group_eval(model, X, G, labels):
    model.eval()
    with torch.no_grad():
        lp = model(torch.from_numpy(X).float().unsqueeze(1) / 255.0).log_softmax(2).numpy()
    per = {}
    single = 0
    for i, g in enumerate(G):
        words, probs = D._beam(lp[i])
        if words and words[0] == labels[g]:
            single += 1
        z = max(sum(probs), 1e-12)
        for w, p in zip(words, probs):
            per.setdefault(g, {})[w] = per.setdefault(g, {}).get(w, 0) + p / z
    voted = sum(max(s, key=s.get) == labels[g] for g, s in per.items())
    return single, len(G), voted, len(per)


def main():
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    man = load_manifest()
    tr, va = codes("train"), codes("val")
    Xt, Yt, _ = renders(tr, man)
    Xv, _, Gv = renders(va, man)
    print(f"train {len(tr)} codes / {len(Xt)} renders; val {len(va)} codes / {len(Xv)} renders", flush=True)
    model = D.DcourtsCRNN()
    model.load_state_dict(torch.load(MODEL, map_location="cpu"))
    s, n, v, nc = group_eval(model, Xv, Gv, va)
    print(f"start: val single {s}/{n}  voted {v}/{nc}", flush=True)
    best = (v, s)
    torch.save(model.state_dict(), MODEL.replace(".pt", "_synthetic.pt"))
    model.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    rng = np.random.default_rng(5)
    T = model(torch.zeros(1, 1, D.IN_H, D.IN_W, device=dev)).shape[1]
    il = torch.full((BS,), T, dtype=torch.long)
    tl = torch.full((BS,), D.LENGTH, dtype=torch.long)
    steps = max(1, len(Xt) // REAL_PER_BATCH)
    for ep in range(1, EPOCHS + 1):
        model.train()
        Xs, Ys = _gen((steps * (BS - REAL_PER_BATCH), 7000 + ep))
        run = 0.0
        for st in range(steps):
            idx = rng.integers(0, len(Xt), REAL_PER_BATCH)
            xr = np.stack([augment(Xt[i], rng) for i in idx])
            sl = slice(st * (BS - REAL_PER_BATCH), (st + 1) * (BS - REAL_PER_BATCH))
            xb = np.concatenate([xr, Xs[sl]])
            yb = np.concatenate([Yt[idx], Ys[sl]])
            x = torch.from_numpy(xb).float().div(255).unsqueeze(1).to(dev)
            opt.zero_grad()
            lp = model(x).log_softmax(2).permute(1, 0, 2).cpu()
            loss = F.ctc_loss(lp, torch.from_numpy(yb), il, tl, blank=D.BLANK, zero_infinity=True)
            loss.backward()
            opt.step()
            run += float(loss.detach())
        model.to("cpu")                      # CPU eval (MPS live-model gremlin)
        s, n, v, nc = group_eval(model, Xv, Gv, va)
        print(f"epoch {ep:3d} loss {run/steps:.3f} | val single {s}/{n} voted {v}/{nc}", flush=True)
        if (v, s) >= best:
            best = (v, s)
            torch.save(model.state_dict(), MODEL)
        model.to(dev)
    print("best", best, "->", MODEL, flush=True)


if __name__ == "__main__":
    main()
