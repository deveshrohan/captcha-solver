"""Close the synthetic->real gap for the MCA reader using hand-labeled reals.

The synthetic-only model reaches 97.7% per character on its own validation set
but only ~88% on real captchas: the generator's font pool does not match the
server's face, and glyph-IoU matching failed to identify it (best 0.67, no clear
winner). Rather than keep guessing the font, this trains on real labels
directly — cheap here because MCA ink is exactly separable, so the captchas are
trivial to read by eye.

Split (seeded, disjoint):
  * train — real labels mixed with fresh synthetic each epoch
  * dev   — chooses the epoch
  * test  — never trained on, never selected on; this is the number to quote

    python3 finetune_mca.py [epochs]
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

from solver import mca as M
from train_mca import gen, score

MODEL_IN = "solver/mca_model.pt"
MODEL_OUT = "solver/mca_model.pt"
LABELS = "mca_labels.json"
SPLIT_JSON = "mca_split.json"
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 25
BS = 128
LR = 4e-4
N_SYNTH = 6000
REAL_REPEAT = 20          # oversample real so it is not drowned by synthetic
SEED = 3


def main():
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(SEED)

    labels = json.load(open(LABELS))
    items = [(p, l) for p, l in sorted(labels.items()) if os.path.exists(p)]

    # The test/dev membership is pinned in mca_split.json rather than re-derived
    # from a seed, so that adding more training labels later cannot silently
    # reshuffle the test set and make runs incomparable.
    if os.path.exists(SPLIT_JSON):
        sp = json.load(open(SPLIT_JSON))
        test_p, dev_p = set(sp["test"]), set(sp["dev"])
    else:
        rs = np.random.RandomState(SEED)
        order = rs.permutation(len(items))
        test_p = {items[i][0] for i in order[:44]}
        dev_p = {items[i][0] for i in order[44:74]}

    split = {"train": [], "dev": [], "test": []}
    for p, l in items:
        split["test" if p in test_p else "dev" if p in dev_p else "train"].append((p, l))
    print("labels: %d  ->  train %d / dev %d / test %d"
          % (len(items), len(split["train"]), len(split["dev"]), len(split["test"])), flush=True)

    def pack(rows):
        X = np.stack([M.load_real(p) for p, _ in rows])
        Y = np.array([M.encode_label(l) for _, l in rows], dtype=np.int64)
        return X, Y, [l for _, l in rows]

    Xtr_r, Ytr_r, _ = pack(split["train"])
    Xdev, _, Ldev = pack(split["dev"])
    Xte, _, Lte = pack(split["test"])

    model = M.McaCRNN().to(dev)
    model.load_state_dict(torch.load(MODEL_IN, map_location=dev))
    with torch.no_grad():
        T = model(torch.zeros(1, 1, M.IN_H, M.IN_W, device=dev)).shape[1]

    def evaluate(X, L):
        p, _ = M.predict_greedy(model, X, dev)
        return score(p, L)

    de, dc, dt = evaluate(Xdev, Ldev)
    te, tc, tt = evaluate(Xte, Lte)
    print("before:  dev %d/%d exact, %d/%d chars  |  TEST %d/%d exact, %d/%d chars (%.2f%%)"
          % (de, len(Ldev), dc, dt, te, len(Lte), tc, tt, 100 * tc / tt), flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    il = torch.full((BS,), T, dtype=torch.long)
    tl = torch.full((BS,), M.LENGTH, dtype=torch.long)

    best = (de, dc)
    torch.save(model.state_dict(), MODEL_OUT)
    for ep in range(1, EPOCHS + 1):
        Xs, Ys = gen(N_SYNTH, SEED * 1000 + ep)
        X = np.concatenate([Xs] + [Xtr_r] * REAL_REPEAT)
        Y = np.concatenate([Ys] + [Ytr_r] * REAL_REPEAT)
        Xd = torch.tensor(X, device=dev)[:, None].float() / 255.0
        Yt = torch.tensor(Y)

        model.train()
        perm = torch.randperm(len(Xd), device=dev)
        Xp, Yp = Xd[perm], Yt[perm.cpu()]
        run, nb = 0.0, 0
        for i in range(0, len(Xp) - BS + 1, BS):
            opt.zero_grad()
            lp = model(Xp[i:i + BS]).log_softmax(2).permute(1, 0, 2).cpu()
            loss = F.ctc_loss(lp, Yp[i:i + BS], il, tl, blank=M.BLANK, zero_infinity=True)
            loss.backward()
            opt.step()
            run += float(loss.detach())
            nb += 1
        sched.step()
        del Xd, Xp

        de, dc, dt = evaluate(Xdev, Ldev)
        te, tc, tt = evaluate(Xte, Lte)
        star = ""
        if (de, dc) >= best:
            best = (de, dc)
            torch.save(model.state_dict(), MODEL_OUT)
            star = "  *saved"
        print("epoch %2d  loss %.3f  dev %d/%d exact %d/%d chars  |  TEST %d/%d exact "
              "%d/%d chars (%.2f%%)%s"
              % (ep, run / nb, de, len(Ldev), dc, dt, te, len(Lte), tc, tt, 100 * tc / tt, star),
              flush=True)

    model.load_state_dict(torch.load(MODEL_OUT, map_location=dev))
    te, tc, tt = evaluate(Xte, Lte)
    print("\nselected checkpoint -> TEST %d/%d exact (%.1f%%), %d/%d chars (%.2f%%)"
          % (te, len(Lte), 100 * te / len(Lte), tc, tt, 100 * tc / tt), flush=True)


if __name__ == "__main__":
    main()
