"""Fine-tune the ITAT v2 reader on hand-labelled reals (after train_itat2.py).

Each epoch mixes fresh synthetic images with jittered copies of the real
TRAIN labels (identical copies just memorise them -- see finetune_itat.py).
The epoch is chosen on the pinned VALIDATION split only; the TEST split is
scored once, at the end, and never influences anything.

    python3 finetune_itat2.py [epochs]
"""
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F

from finetune_itat import augment
from solver import itat2 as I
from train_itat2 import gen, score

MODEL = "solver/itat2_model.pt"
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 30
BS = 128
LR = 5e-4
N_SYNTH = 3000
REAL_REPEAT = 30
SEED = 11


def pack(paths, labels):
    X = np.stack([I.load_real(p) for p in paths]) if paths else np.empty((0, I.IN_H, I.IN_W), np.uint8)
    Y = np.array([I.encode_label(labels[p]) for p in paths], np.int64)
    return X, Y, [labels[p] for p in paths]


def main():
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(SEED)
    labels = json.load(open("itat2_labels.json"))
    split = json.load(open("itat2_split.json"))
    Xtr, Ytr, _ = pack(split["train"], labels)
    Xva, _, Lva = pack(split["validation"], labels)
    print("train %d / validation %d / test %d (test scored once, at the end)"
          % (len(Xtr), len(Lva), len(split["test"])), flush=True)

    model = I.Itat2CRNN().to(dev)
    model.load_state_dict(torch.load(MODEL, map_location=dev))
    with torch.no_grad():
        T = model(torch.zeros(1, 1, I.IN_H, I.IN_W, device=dev)).shape[1]

    def evaluate(X, L):
        was = next(model.parameters()).device
        model.to("cpu")                       # CPU eval: see train_itat.py
        p, _ = I.predict(model, X, "cpu")
        model.to(was)
        return score(p, L)

    ve, vc, vt = evaluate(Xva, Lva)
    print("before: validation %d/%d exact, %d/%d chars" % (ve, len(Lva), vc, vt), flush=True)
    best = (vc, ve)
    torch.save(model.state_dict(), MODEL)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    il = torch.full((BS,), T, dtype=torch.long)
    tl = torch.full((BS,), I.LENGTH, dtype=torch.long)
    for ep in range(1, EPOCHS + 1):
        rng = np.random.default_rng(SEED * 1000 + ep)
        Xs, Ys = gen(N_SYNTH, SEED * 1000 + ep)
        X = np.concatenate([Xs, augment(np.tile(Xtr, (REAL_REPEAT, 1, 1)), rng)])
        Y = np.concatenate([Ys, np.tile(Ytr, (REAL_REPEAT, 1))])
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
        ve, vc, vt = evaluate(Xva, Lva)
        star = ""
        if (vc, ve) >= best:
            best = (vc, ve)
            torch.save(model.state_dict(), MODEL)
            star = "  *saved"
        print("epoch %2d  loss %.3f  validation %d/%d exact, %d/%d chars%s"
              % (ep, run / nb, ve, len(Lva), vc, vt, star), flush=True)
    model.load_state_dict(torch.load(MODEL, map_location=dev))
    Xte, _, Lte = pack(split["test"], labels)
    te, tc, tt = evaluate(Xte, Lte)
    print("\nselected checkpoint -> TEST %d/%d exact (%.1f%%), %d/%d chars (%.1f%%)"
          % (te, len(Lte), 100 * te / len(Lte), tc, tt, 100 * tc / tt), flush=True)


if __name__ == "__main__":
    main()
