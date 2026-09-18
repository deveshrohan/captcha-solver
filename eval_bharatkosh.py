"""Score the Bharatkosh reader. Two legs, because neither is enough alone.

    python3 eval_bharatkosh.py [model.pt] [--split test|dev] [--census]

LEG 1 -- label-free, over every harvested group. `?New=0` re-renders one answer,
so the K renders of a group are independent looks at the same word. Reported:
how often a single render's read equals the group's voted read, and how many
groups are unanimous. It needs no labels, so it scales to the whole harvest and
is the drift signal. It CANNOT catch a confusion every render shares.

LEG 2 -- hand-labelled groups, held out. The split is pinned by GROUP id in
bharatkosh_split.json (never by image: renders of a test answer must not leak
into training) and every hand-labelled group is held out -- training uses
synthetic data and agreement pseudo-labels only. Reported as single-render
exact, and voted exact at K=3 (mean over all 10 subsets) and K=5, each
case-sensitive and case-folded. Case is a human guess for `c s v x ...` under
per-glyph random size, so the folded column is the one the labels can support.

Caveat, stated rather than hidden: groups a human could not read confidently
are in bharatkosh_ambiguous.json and excluded, so leg 2 is biased toward easier
answers. Leg 1 covers all groups and is the check on that bias.
"""
import itertools
import json
import os
import sys
from collections import Counter

import numpy as np
import torch

from solver import bharatkosh as B

ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
MODEL = ARGS[0] if ARGS else "solver/bharatkosh_model.pt"
SPLIT_NAME = sys.argv[sys.argv.index("--split") + 1] if "--split" in sys.argv else "test"
RAW = "bharatkosh_raw"
LABELS = "bharatkosh_labels.json"
SPLIT = "bharatkosh_split.json"


def groups():
    g = {}
    for f in sorted(os.listdir(RAW)):
        if f.endswith(".png"):
            g.setdefault(f[:5], []).append(os.path.join(RAW, f))
    return g


def census(labels):
    c = Counter("".join(labels.values()))
    n = sum(c.values())
    absent = [ch for ch in B.CHARSET if c[ch] == 0]
    print(f"\ncensus: {n} labelled chars over {len(labels)} groups")
    print("  absent:", "".join(absent) or "-")
    # P(a given class never appears | uniform over the full charset)
    print(f"  P(one given class absent by chance) = {(1 - 1/B.N_CLASSES) ** n:.3g}"
          f"  -> exclusions need this well below 0.01")


def main():
    model = B.BharatkoshCRNN()
    model.load_state_dict(torch.load(MODEL, map_location="cpu"))
    G = groups()
    lps = {g: B.log_probs(model, np.stack([B.load_real(p) for p in ps]))
           for g, ps in G.items()}
    single = {g: [B.vote([lp])[0] for lp in lp5] for g, lp5 in lps.items()}
    voted = {g: B.vote(list(lp5)) for g, lp5 in lps.items()}

    n_r = sum(len(v) for v in single.values())
    agree = sum(s == voted[g][0] for g, ss in single.items() for s in ss)
    agree_f = sum(s.lower() == voted[g][0].lower() for g, ss in single.items() for s in ss)
    unan = sum(len(set(ss)) == 1 for ss in single.values())
    unan_f = sum(len({s.lower() for s in ss}) == 1 for ss in single.values())
    print(f"LEG 1 (label-free)  {len(G)} groups, {n_r} renders")
    print(f"  render == group vote   {agree}/{n_r} ({agree/n_r:.1%})   folded {agree_f/n_r:.1%}")
    print(f"  unanimous groups       {unan}/{len(G)} ({unan/len(G):.1%})   folded {unan_f/len(G):.1%}")
    print(f"  median vote confidence {np.median([c for _, c in voted.values()]):.3f}")

    labels = json.load(open(LABELS))
    ids = [g for g in json.load(open(SPLIT))[SPLIT_NAME] if g in lps]
    print(f"\nLEG 2 (hand-labelled, split={SPLIT_NAME})  {len(ids)} groups")

    def rate(pairs):
        pairs = list(pairs)
        cs = sum(p == l for p, l in pairs)
        cf = sum(p.lower() == l.lower() for p, l in pairs)
        return f"{cs}/{len(pairs)} ({cs/len(pairs):.1%})   folded {cf}/{len(pairs)} ({cf/len(pairs):.1%})"

    print("  single render  ", rate((s, labels[g]) for g in ids for s in single[g]))
    print("  voted K=3      ", rate((B.vote([lps[g][i] for i in sub])[0], labels[g])
                                    for g in ids
                                    for sub in itertools.combinations(range(len(lps[g])), 3)))
    print("  voted K=5      ", rate((voted[g][0], labels[g]) for g in ids))
    ch = sum(a == b for g in ids for a, b in zip(voted[g][0], labels[g]))
    print(f"  voted K=5 chars {ch}/{len(ids) * B.LENGTH}")
    for g in ids:
        if voted[g][0] != labels[g]:
            print(f"    {g} want {labels[g]} got {voted[g][0]} (conf {voted[g][1]:.2f})"
                  f"  renders {' '.join(single[g])}")
    if "--census" in sys.argv:
        census(labels)


if __name__ == "__main__":
    main()
