"""Score the ITAT reader against the hand-labeled real captchas.

    python3 eval_itat.py [model.pt] [labels.json]

The reader is case-insensitive by design (see solver/itat.py): labels are stored
folded to uppercase and the model emits uppercase classes, so exact match here IS
the case-insensitive score. The charset is homoglyph-free (`0 1 I O` never occur
and case is folded), so unlike MCA there is no ambiguous-pair breakdown to report.
"""
import json
import os
import sys
from collections import Counter

import numpy as np
import torch

from solver import itat as I

MODEL = sys.argv[1] if len(sys.argv) > 1 else "solver/itat_model.pt"
LABELS = sys.argv[2] if len(sys.argv) > 2 else "itat_labels.json"
SPLIT = "itat_split.json"


def main():
    labels = json.load(open(LABELS))
    items = [(p, l.upper()) for p, l in sorted(labels.items())
             if os.path.exists(p) and len(l) == I.LENGTH and all(c in I.CH2I for c in l.upper())]
    # Report the held-out split by default. Scoring every label includes images
    # the model was fine-tuned on and has memorised, which reads far above the
    # true held-out number (the failure mode selfimprove.py exists to prevent).
    if os.path.exists(SPLIT) and "--all" not in sys.argv:
        test = set(json.load(open(SPLIT)).get("test", []))
        items = [(p, l) for p, l in items if p in test]
        print("scoring the held-out test split (pass --all to score every label)\n")

    model = I.ItatCRNN()
    model.load_state_dict(torch.load(MODEL, map_location="cpu"))
    model.eval()

    imgs = np.stack([I.load_real(p) for p, _ in items])
    preds, confs = I.predict(model, imgs, "cpu")

    exact = sum(p == l for p, (_, l) in zip(preds, items))
    chars = sum(sum(a == b for a, b in zip(p, l)) for p, (_, l) in zip(preds, items))
    total = len(items) * I.LENGTH

    print(f"model  : {MODEL}")
    print(f"images : {len(items)} hand-labeled real captchas (case-insensitive)")
    print(f"exact  : {exact}/{len(items)} ({exact/max(len(items),1)*100:.1f}%)")
    print(f"chars  : {chars}/{total} ({chars/max(total,1)*100:.2f}%)")
    print(f"mean confidence: {np.mean(confs):.3f}")

    conf = Counter()
    for p, (_, l) in zip(preds, items):
        if len(p) != len(l):
            continue
        for a, b in zip(l, p):
            if a != b:
                conf[(a, b)] += 1
    if conf:
        print("\ntop character confusions (true -> pred):")
        for (a, b), n in conf.most_common(15):
            print(f"   {a} -> {b}   x{n}")


if __name__ == "__main__":
    main()
