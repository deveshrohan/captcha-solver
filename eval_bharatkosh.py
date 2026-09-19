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
synthetic data and agreement pseudo-labels only. Reported as voted exact at
every K from 1 (a single render) to 5, K<5 averaged over all K-subsets, each
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

SPLIT_NAME = sys.argv[sys.argv.index("--split") + 1] if "--split" in sys.argv else "test"
ARGS = [a for i, a in enumerate(sys.argv[1:], 1)
        if not a.startswith("--") and sys.argv[i - 1] != "--split"]
MODEL = ARGS[0] if ARGS else "solver/bharatkosh_model.pt"
RAW = "bharatkosh_raw"
LABELS = "bharatkosh_labels.json"
SPLIT = "bharatkosh_split.json"
PSEUDO = "bharatkosh_pseudo.json"


def groups():
    g = {}
    for f in sorted(os.listdir(RAW)):
        if f.endswith(".png"):
            g.setdefault(f[:5], []).append(os.path.join(RAW, f))
    return g


def census(labels):
    """Hand-label census over the FULL 62-class alnum set, so the exclusions
    baked into B.CHARSET stay visible and re-checkable as labels are added."""
    full = B.CHARSET + B.EXCLUDED
    c = Counter("".join(labels.values()))
    n = sum(c.values())
    absent = "".join(ch for ch in sorted(full) if c[ch] == 0)
    print(f"\ncensus: {n} hand-labelled chars over {len(labels)} groups")
    print(f"  absent: {absent or '-'}   (excluded by design: {B.EXCLUDED})")
    print(f"  P(all of the {len(B.EXCLUDED)} excluded absent | uniform 62) = "
          f"{((62 - len(B.EXCLUDED)) / 62) ** n:.2g}")
    print(f"  P(one given class absent by chance) = {(1 - 1/62) ** n:.3g}"
          f"  -> a single absent class alone proves nothing at this n")


def main():
    model = B.BharatkoshCRNN()
    model.load_state_dict(torch.load(MODEL, map_location="cpu"))
    G = groups()
    lps = {}
    for g, ps in G.items():
        try:
            lps[g] = B.log_probs(model, np.stack([B.load_real(p) for p in ps]))
        except OSError:                 # a render still being written by a harvest
            print(f"  skipping {g}: unreadable render")
    G = {g: G[g] for g in lps}
    single = {g: [B.vote([lp])[0] for lp in lp5] for g, lp5 in lps.items()}
    voted = {g: B.vote(list(lp5)) for g, lp5 in lps.items()}

    # Groups the fine-tune trained on (its own votes as labels) agree with their
    # vote by construction, and the ones it rejected were chosen BY disagreement,
    # so leg 1 is also reported over "fresh" groups: harvested after the
    # fine-tune ran and never labelled. That line is the honest one.
    seen = set(json.load(open(PSEUDO))["seen"]) if os.path.exists(PSEUDO) else set()
    labelled = set(json.load(open(LABELS)))
    print("LEG 1 (label-free)")
    for name, sel in (("all groups", set(single)),
                      ("fresh", set(single) - seen - labelled)):
        n_g = len(sel)
        if not n_g:
            continue
        n_r = sum(len(single[g]) for g in sel)
        agree = sum(s == voted[g][0] for g in sel for s in single[g])
        agree_f = sum(s.lower() == voted[g][0].lower() for g in sel for s in single[g])
        unan = sum(len(set(single[g])) == 1 for g in sel)
        print(f"  {name:17s} {n_g:3d} groups: render == vote {agree}/{n_r} ({agree/n_r:.1%},"
              f" folded {agree_f/n_r:.1%})   unanimous {unan}/{n_g} ({unan/n_g:.1%})")
    print(f"  median vote confidence {np.median([c for _, c in voted.values()]):.3f}")

    labels = json.load(open(LABELS))
    ids = [g for g in json.load(open(SPLIT))[SPLIT_NAME] if g in lps]
    print(f"\nLEG 2 (hand-labelled, split={SPLIT_NAME})  {len(ids)} groups")

    def rate(pairs):
        pairs = list(pairs)
        cs = sum(p == l for p, l in pairs)
        cf = sum(p.lower() == l.lower() for p, l in pairs)
        return f"{cs}/{len(pairs)} ({cs/len(pairs):.1%})   folded {cf}/{len(pairs)} ({cf/len(pairs):.1%})"

    # K=1 is the single-render read; K=2..4 average over every K-subset of a
    # group's renders, so each K uses all the data rather than one arbitrary pick
    for k in range(1, 6):
        pairs = ((B.vote([lps[g][i] for i in sub])[0], labels[g])
                 for g in ids
                 for sub in itertools.combinations(range(len(lps[g])), k))
        print(f"  voted K={k}      ", rate(pairs))
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
