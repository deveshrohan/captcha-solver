"""Score the GST reader against the hand-labeled real captchas.

    python3 eval_gst.py [model.pt] [labels.json]

Prints exact-match and per-digit accuracy, plus every disagreement so the
labels themselves can be adjudicated (the middle digits are fisheye-magnified
and genuinely hard to read by eye).
"""
import json
import os
import sys

import numpy as np
import torch

from solver import gst as G

MODEL = sys.argv[1] if len(sys.argv) > 1 else "solver/gst_model.pt"
LABELS = sys.argv[2] if len(sys.argv) > 2 else "gst_labels.json"
UNCERTAIN = "gst_labels_uncertain.json"


def main():
    labels = json.load(open(LABELS))
    items = [(p, l) for p, l in sorted(labels.items()) if os.path.exists(p)]
    flagged = set(json.load(open(UNCERTAIN))) if os.path.exists(UNCERTAIN) else set()

    model = G.GstCRNN()
    model.load_state_dict(torch.load(MODEL, map_location="cpu"))
    model.eval()

    imgs = np.stack([G.load_real(p) for p, _ in items])
    preds, confs = G.predict(model, imgs, "cpu")

    exact = sum(p == l for p, (_, l) in zip(preds, items))
    chars = sum(sum(a == b for a, b in zip(p, l)) for p, (_, l) in zip(preds, items))
    total = len(items) * G.LENGTH

    print(f"model  : {MODEL}")
    print(f"images : {len(items)} hand-labeled real captchas")
    print(f"exact  : {exact}/{len(items)}  ({exact/len(items)*100:.1f}%)")
    print(f"digits : {chars}/{total}  ({chars/total*100:.2f}%)")
    print(f"mean confidence: {np.mean(confs):.3f}")

    bad = [(p, l, pr, c) for (p, l), pr, c in zip(items, preds, confs) if pr != l]
    if bad:
        print(f"\ndisagreements ({len(bad)}):")
        for p, l, pr, c in sorted(bad, key=lambda x: x[3]):
            mark = "  <- label flagged uncertain" if p in flagged else ""
            print(f"  {os.path.basename(p):12s} label={l}  pred={pr}  conf={c:.3f}{mark}")

    lo = sorted(zip(confs, [p for p, _ in items]))[:5]
    print("\nlowest-confidence reads:")
    for c, p in lo:
        print(f"  {os.path.basename(p):12s} conf={c:.3f}")


if __name__ == "__main__":
    main()
