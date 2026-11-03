#!/usr/bin/env python3
"""Score the CESTAT reader on its held-out split -> cestat_eval.json.

    python3 eval_cestat.py [train|test]

Labels are the four visible characters (see solver/cestat.py: the image
shows only four of the six-character code).
"""
import json
import sys
import time

from solver import cestat as C

GATE = 0.35          # below the train split's lowest confidence (0.403)


def main():
    split = sys.argv[1] if len(sys.argv) > 1 else "test"
    labels = json.load(open("cestat_labels.json"))
    paths = json.load(open("cestat_split.json"))[split]
    rows, chars, t0 = [], 0, time.time()
    for p in paths:
        text, conf = C.solve_image(p)
        lab = labels[p]
        chars += sum(a == b for a, b in zip(text, lab))
        rows.append(dict(path=p, label=lab, prediction=text, confidence=conf,
                         correct=text == lab, chars=sum(a == b for a, b in zip(text, lab))))
    ms = (time.time() - t0) * 1000 / len(paths)
    accepted = [r for r in rows if r["confidence"] >= GATE]
    out = dict(kind="cestat", split=split, n=len(rows),
               exact=sum(r["correct"] for r in rows),
               char_accuracy=chars / (C.VISIBLE * len(rows)), gate=GATE,
               accepted=len(accepted),
               accepted_correct=sum(r["correct"] for r in accepted),
               visible_chars=C.VISIBLE, answer_length=C.LENGTH,
               ms_per_solve=round(ms, 1), rows=rows)
    if split == "test":
        json.dump(out, open("cestat_eval.json", "w"), indent=1)
    print({k: v for k, v in out.items() if k != "rows"})


if __name__ == "__main__":
    main()
