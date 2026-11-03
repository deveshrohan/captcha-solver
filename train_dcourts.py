"""Pretrain the district-courts (securimage-wp) CRNN+CTC reader on synthetic
renders from solver/dcourts.render, a port of Securimage 4's doImage.

    python3 train_dcourts.py [epochs]

Labels are case-folded: the portal validates case-insensitively. Glyphs are
drawn in their real case. Real eye-labelled renders, minus the test split,
are only printed and used to pick the checkpoint. finetune_dcourts.py trains
on them.
"""
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from solver import dcourts as D

SEED = 23
STEPS = int(os.environ.get("DC_STEPS", "600"))     # fresh batches per epoch
N_VAL = 2000
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 25
BS = 128
LR = 2e-3
MODEL_PATH = "solver/dcourts_model.pt"


def _gen(args):
    n, seed = args
    rng = np.random.default_rng(seed)
    X = np.empty((n, D.IN_H, D.IN_W), np.uint8)
    Y = np.empty((n, D.LENGTH), np.int64)
    for i in range(n):
        code = D.random_code(rng)
        g = D.render(code, rng, font_scale=rng.uniform(0.86, 1.04))
        X[i] = np.rint(D.to_input(g) * 255)
        Y[i] = [D.C2I[c] for c in code.upper()]
    return X, Y


def gen(n, seed, pool):
    k = pool._processes
    sizes = [n // k + (j < n % k) for j in range(k)]
    out = pool.map(_gen, [(s, seed * 1000 + j) for j, s in enumerate(sizes) if s])
    return np.concatenate([o[0] for o in out]), np.concatenate([o[1] for o in out])


def real_set(split_names=("train", "val")):
    """(inputs, labels) of labelled real renders in the given splits."""
    if not os.path.exists("dcourts_split.json"):
        return None, []
    labels = json.load(open("dcourts_labels.json"))
    split = json.load(open("dcourts_split.json"))
    keep = {k for s in split_names for k in split.get(s, [])}
    from download_dcourts import load_manifest
    man = load_manifest()
    X, L = [], []
    for code_id in sorted(keep):
        for name in man[code_id]["renders"]:
            g = D.load("dcourts_raw/" + name)
            X.append(np.rint(D.to_input(g) * 255).astype(np.uint8))
            L.append(labels[code_id])
    return (np.stack(X) if X else None), L


def recalibrate(model, pool, n=12, seed=31337):
    """CPU copy of `model` with BatchNorm statistics recomputed from fresh
    renders. The running statistics accumulated while training on MPS are
    wrong here: a checkpoint scored 12.5% in eval mode and 29% after this
    pass, and the gap grew every epoch while the training loss kept falling."""
    import copy
    m = copy.deepcopy(model).to("cpu")
    for mod in m.modules():
        if isinstance(mod, torch.nn.BatchNorm2d):
            mod.reset_running_stats()
            mod.momentum = None                      # cumulative average
    m.train()
    with torch.no_grad():
        for Xb, _ in pool.imap_unordered(_gen, [(128, seed + i) for i in range(n)]):
            m(torch.from_numpy(Xb).float().unsqueeze(1) / 255.0)
    for mod in m.modules():
        if isinstance(mod, torch.nn.BatchNorm2d):
            mod.momentum = 0.1
    return m.eval()


def greedy(model, X):
    with torch.no_grad():
        idx = model(torch.from_numpy(X).float().unsqueeze(1) / 255.0).argmax(2).numpy()
    out = []
    for row in idx:
        s, prev = [], -1
        for c in row:
            if c != prev and c != D.BLANK:
                s.append(D.CLASSES[c])
            prev = c
        out.append("".join(s))
    return out


def main():
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(SEED)
    pool = mp.Pool(max(1, min(8, (os.cpu_count() or 4) - 1)))
    t = time.time()
    Xv, Yv = gen(N_VAL, 999, pool)
    vl = ["".join(D.CLASSES[i] for i in r) for r in Yv]
    print(f"device={dev} streaming {STEPS} fresh batches/epoch, val={N_VAL} ({time.time()-t:.0f}s)", flush=True)
    Xr, Lr = real_set()
    print(f"real (non-test) renders: {len(Lr)}", flush=True)

    model = D.DcourtsCRNN().to(dev)
    with torch.no_grad():
        T = model(torch.zeros(1, 1, D.IN_H, D.IN_W, device=dev)).shape[1]
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=EPOCHS * STEPS)
    il = torch.full((BS,), T, dtype=torch.long)
    tl = torch.full((BS,), D.LENGTH, dtype=torch.long)
    best = (-1, -1)
    for ep in range(1, EPOCHS + 1):
        model.train()
        run = nb = 0
        # Fresh renders every batch. A fixed 80k pool was memorised within 4
        # epochs (train CTC 0.10 while synthetic val fell to 1.6%).
        batches = pool.imap_unordered(_gen, [(BS, (SEED * 100 + ep) * 100000 + i) for i in range(STEPS)])
        for Xb, Yb in batches:
            xb = (torch.from_numpy(Xb).to(dev).float() / 255.0).unsqueeze(1)
            opt.zero_grad()
            lp = model(xb).log_softmax(2).permute(1, 0, 2).cpu()
            loss = F.ctc_loss(lp, torch.from_numpy(Yb), il, tl, blank=D.BLANK, zero_infinity=True)
            loss.backward()
            opt.step()
            sched.step()
            run += float(loss.detach()); nb += 1
        # evaluate a CPU copy with recomputed BatchNorm statistics (see
        # recalibrate); the training model itself stays on MPS
        ev = recalibrate(model, pool, seed=ep * 1000)
        vp = greedy(ev, Xv)
        vex = sum(a == b for a, b in zip(vp, vl))
        line = f"epoch {ep:3d} loss {run/nb:.3f} synthVAL exact {vex/N_VAL:.3f}"
        cur = (vex, 0)
        if Xr is not None:
            rp = greedy(ev, Xr)
            rex = sum(a == b for a, b in zip(rp, Lr))
            rch = sum(sum(x == y for x, y in zip(a, b)) for a, b in zip(rp, Lr))
            line += f" | REAL exact {rex}/{len(Lr)} char {rch/(5*len(Lr)):.3f}"
            cur = (vex, rex)       # synthetic val first: real labels are few
        print(line, flush=True)
        torch.save(ev.state_dict(), MODEL_PATH.replace(".pt", "_last.pt"))
        if cur >= best:
            best = cur
            torch.save(ev.state_dict(), MODEL_PATH)
    pool.close()
    print("saved best ->", MODEL_PATH, best, flush=True)


if __name__ == "__main__":
    main()
