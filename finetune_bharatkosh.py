"""Fine-tune the Bharatkosh reader on real images WITHOUT hand labels.

    python3 finetune_bharatkosh.py [epochs]

Hand-reading this captcha is unreliable (about a quarter of groups could not be
read confidently even from five renders), so every hand label is spent on
evaluation and none on training. The real-image signal comes from the endpoint
instead: `?New=0` re-renders one answer, so a group's K renders are independent
looks at the same word, and the group VOTE is a far better label than any single
read. A group is accepted as a pseudo-label when

    vote confidence >= MIN_CONF   and   >= MIN_AGREE of its K single reads
                                        already equal the vote.

MIN_AGREE is deliberately K-1, not K. A unanimous group teaches nothing -- the
model already reads every render of it. The group with ONE dissenting render is
where the learning is: that render is a real image the model gets wrong, now
labelled by its four siblings.

Excluded from pseudo-labelling: every group in bharatkosh_labels.json (dev and
test alike -- they are held out). Epochs are selected on `dev`; `test` is never
loaded. Real renders are shown under per-epoch affine jitter, never as identical
repeats (the ITAT lesson), and mixed with fresh synthetic data for class
coverage.
"""
import json
import multiprocessing as mp
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

from solver import bharatkosh as B
from train_bharatkosh import dev_set, gen, score

MODEL = "solver/bharatkosh_model.pt"
RAW = "bharatkosh_raw"
LABELS = "bharatkosh_labels.json"
# The groups this fine-tune saw and trained on. eval_bharatkosh.py reads it so
# the label-free leg is ALSO reported on fresh groups: on trained-on groups,
# render/vote agreement rising is memorisation, not accuracy.
PSEUDO = "bharatkosh_pseudo.json"
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 12
BS = 128
LR = 3e-4
N_SYNTH = 12000
REAL_REPEAT = 8
MIN_CONF = 0.99
MIN_AGREE = 4
SEED = 5


def augment(X, rng):
    """Small random affine jitter on (N, IN_H, IN_W) uint8 inputs. Rotation is
    kept small: the bar band is axis-aligned and must stay where it is."""
    n = len(X)
    t = torch.tensor(X, dtype=torch.float32)[:, None] / 255.0
    ang = torch.tensor(rng.uniform(-2, 2, n) * np.pi / 180, dtype=torch.float32)
    sc = torch.tensor(rng.uniform(0.95, 1.05, n), dtype=torch.float32)
    tx = torch.tensor(rng.uniform(-0.04, 0.04, n), dtype=torch.float32)
    ty = torch.tensor(rng.uniform(-0.03, 0.03, n), dtype=torch.float32)
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    theta = torch.zeros(n, 2, 3)
    theta[:, 0, 0] = cos; theta[:, 0, 1] = -sin; theta[:, 0, 2] = tx
    theta[:, 1, 0] = sin; theta[:, 1, 1] = cos; theta[:, 1, 2] = ty
    grid = F.affine_grid(theta, t.shape, align_corners=False)
    out = F.grid_sample(t, grid, align_corners=False, padding_mode="zeros")
    return (out[:, 0].clamp(0, 1) * 255.0).round().to(torch.uint8).numpy()


def pseudo_labels(model):
    held_out = set(json.load(open(LABELS)))
    groups = {}
    for f in sorted(os.listdir(RAW)):
        if f.endswith(".png") and f[:5] not in held_out:
            groups.setdefault(f[:5], []).append(os.path.join(RAW, f))
    X, Y, ids, seen = [], [], [], 0
    for g, paths in groups.items():
        imgs = np.stack([B.load_real(p) for p in paths])
        lps = list(B.log_probs(model, imgs))
        word, conf = B.vote(lps)
        agree = sum(B.vote([lp])[0] == word for lp in lps)
        seen += 1
        if conf >= MIN_CONF and agree >= min(MIN_AGREE, len(lps)) and len(word) == B.LENGTH:
            X.append(imgs)
            Y += [B.encode_label(word)] * len(imgs)
            ids.append(g)
    print(f"pseudo-labelled {len(X)}/{seen} groups ({len(Y)} renders)", flush=True)
    # `seen` too, not just `trained`: a group the rule REJECTED was picked out by
    # the model's own disagreement, so it is no unbiased sample either. Only
    # groups harvested after this ran are fresh.
    json.dump({"trained": sorted(ids), "seen": sorted(groups)}, open(PSEUDO, "w"),
              indent=0)
    return np.concatenate(X), np.array(Y, np.int64)


def dev_score(model, Xdev, dlabels):
    """(voted-exact over dev groups, single-render chars) -- vote first, because
    the vote is what ships."""
    lps = B.log_probs(model, Xdev)
    _, ch, _ = score(B.greedy(lps), dlabels)
    k = 5
    ex = sum(B.vote(list(lps[i:i + k]))[0] == dlabels[i]
             for i in range(0, len(dlabels), k))
    return ex, ch


def main():
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    pool = mp.Pool(max(1, min(8, (os.cpu_count() or 4) - 1)))

    model = B.BharatkoshCRNN()
    model.load_state_dict(torch.load(MODEL, map_location="cpu"))
    Xr, Yr = pseudo_labels(model)
    Xdev, dlabels = dev_set()
    best = dev_score(model, Xdev, dlabels)
    print(f"before: DEV voted {best[0]}/{len(dlabels)//5} single-render chars "
          f"{best[1]}/{len(dlabels)*B.LENGTH}", flush=True)

    model.to(dev)
    with torch.no_grad():
        T = model(torch.zeros(1, 1, B.IN_H, B.IN_W, device=dev)).shape[1]
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    il = torch.full((BS,), T, dtype=torch.long)
    tl = torch.full((BS,), B.LENGTH, dtype=torch.long)

    for ep in range(1, EPOCHS + 1):
        Xs, Ys = gen(N_SYNTH, 7000 + ep, pool)
        Xa = np.concatenate([augment(Xr, rng) for _ in range(REAL_REPEAT)] + [Xs])
        Ya = np.concatenate([Yr] * REAL_REPEAT + [Ys])
        Xall, Yall = torch.from_numpy(Xa), torch.tensor(Ya)
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
            run += float(loss.detach())
            nb += 1
        model.to("cpu")                      # CPU eval: the MPS live-model gremlin
        cur = dev_score(model, Xdev, dlabels)
        keep = cur > best
        print(f"epoch {ep:2d} loss {run/nb:.3f} DEV voted {cur[0]}/{len(dlabels)//5} "
              f"chars {cur[1]}/{len(dlabels)*B.LENGTH}{'  *saved' if keep else ''}",
              flush=True)
        if keep:
            best = cur
            torch.save(model.state_dict(), MODEL)
        model.to(dev)
    pool.close()
    print(f"best DEV voted {best[0]} chars {best[1]}", flush=True)


if __name__ == "__main__":
    main()
