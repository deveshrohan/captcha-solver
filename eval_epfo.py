"""Score the EPFO reader against the hand-labeled real captchas.

    python3 eval_epfo.py [model.pt] [labels.json]

EPFO excludes 0, I, N and O from its charset, so unlike MCA it has no homoglyph
pairs at all — the lenient and strict numbers are identical by construction.
"""
import json
import os
import sys
from collections import Counter

import numpy as np
import torch

from solver import epfo as M

MODEL = sys.argv[1] if len(sys.argv) > 1 else "solver/epfo_model.pt"
LABELS = sys.argv[2] if len(sys.argv) > 2 else "epfo_labels.json"

HOMOGLYPHS = [set("0O"), set("1lI"), set("5S"), set("2Z"), set("9g")]


def same_class(a, b):
    if a == b:
        return True
    return any(a in g and b in g for g in HOMOGLYPHS)


SPLIT = "epfo_split.json"


def main():
    labels = json.load(open(LABELS))
    items = [(p, l) for p, l in sorted(labels.items()) if os.path.exists(p)]
    # Report the held-out split by default. Scoring every label includes the
    # images the model was fine-tuned on and has memorised — that reads 96.5%
    # against a true 75.0%, which is the kind of number that hides a regression.
    if os.path.exists(SPLIT) and "--all" not in sys.argv:
        test = set(json.load(open(SPLIT))["test"])
        items = [(p, l) for p, l in items if p in test]
        print("scoring the held-out test split (pass --all to score every label)\n")
    model = M.EpfoCRNN()
    model.load_state_dict(torch.load(MODEL, map_location="cpu"))
    model.eval()

    imgs = np.stack([M.load_real(p) for p, _ in items])
    preds, confs = M.predict(model, imgs, "cpu")

    exact = sum(p == l for p, (_, l) in zip(preds, items))
    chars = sum(sum(a == b for a, b in zip(p, l)) for p, (_, l) in zip(preds, items))
    lenient_c = sum(sum(same_class(a, b) for a, b in zip(p, l)) for p, (_, l) in zip(preds, items))
    lenient_e = sum(all(same_class(a, b) for a, b in zip(p, l)) and len(p) == len(l)
                    for p, (_, l) in zip(preds, items))
    total = len(items) * M.LENGTH

    print(f"model  : {MODEL}")
    print(f"images : {len(items)} hand-labeled real captchas")
    print(f"exact  : {exact}/{len(items)} ({exact/len(items)*100:.1f}%)")
    print(f"chars  : {chars}/{total} ({chars/total*100:.2f}%)")
    print(f"exact, homoglyphs treated as equal : {lenient_e}/{len(items)} ({lenient_e/len(items)*100:.1f}%)")
    print(f"chars, homoglyphs treated as equal : {lenient_c}/{total} ({lenient_c/total*100:.2f}%)")
    print(f"mean confidence: {np.mean(confs):.3f}")

    conf = Counter()
    for p, (_, l) in zip(preds, items):
        if len(p) != len(l):
            continue
        for a, b in zip(l, p):
            if a != b:
                conf[(a, b)] += 1
    print("\ntop character confusions (true -> pred):")
    for (a, b), n in conf.most_common(15):
        tag = "  (homoglyph)" if same_class(a, b) else ""
        print(f"   {a} -> {b}   x{n}{tag}")


if __name__ == "__main__":
    main()
