"""Score the MCA reader against the hand-labeled real captchas.

    python3 eval_mca.py [model.pt] [labels.json]

Also breaks out how much of the residual error is the charset's inherent
homoglyph ambiguity (`0`/`O`, `1`/`l`/`I`), which no reader can resolve from the
glyph alone — a caller that can retry, or that treats those classes as
interchangeable, effectively sees the higher number.
"""
import json
import os
import sys
from collections import Counter

import numpy as np
import torch

from solver import mca as M

MODEL = sys.argv[1] if len(sys.argv) > 1 else "solver/mca_model.pt"
LABELS = sys.argv[2] if len(sys.argv) > 2 else "mca_labels.json"

HOMOGLYPHS = [set("0O"), set("1lI"), set("5S"), set("2Z"), set("9g")]


def same_class(a, b):
    if a == b:
        return True
    return any(a in g and b in g for g in HOMOGLYPHS)


def main():
    labels = json.load(open(LABELS))
    items = [(p, l) for p, l in sorted(labels.items()) if os.path.exists(p)]
    model = M.McaCRNN()
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
