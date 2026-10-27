"""Held-out scores for the High Court readers (gujhc, madhc, calhc, karhc).

    python3 eval_hc.py madhc [--split test|validation] [--output madhc_eval.json]

Two legs, as for NGT:

* labelled: exact / per-character accuracy on the pinned split, routed through
  `solve_bytes` exactly as a caller would (so a routing bug fails here too),
  plus precision above the reader's confidence gate and a replay of the
  <=3-fetch retry policy over disjoint groups of three. The replay compares
  the returned read with the label of the image fetched LAST, because that is
  the answer the server session would hold; it cannot pick a correct image.
* label-free (cover readers): the confidence of every harvested image. A
  generator change shows up here first, as a low-fidelity tail, without
  anyone labelling anything.

The test split is never used to build templates or pick a checkpoint.
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np

from solver.api import GATES, solve_bytes, solve_with_retry

RAW = {"gujhc": "gujhc_raw/*.png", "madhc": "madhc_raw/*.png", "calhc": "calhc_raw/*.jpg",
       "karhc": "karhc_raw/*.jpg", "itat2": "itat2_raw/*.png"}


def evaluate(kind, part="test", label_free=True):
    split = json.loads(Path(kind + "_split.json").read_text())
    labels = json.loads(Path(kind + "_labels.json").read_text())
    paths = split[part]
    if part == "test" and len(paths) < 30:
        raise ValueError("need at least 30 held-out images")
    gate = GATES[kind]
    rows = []
    for p in paths:
        text, conf, route = solve_bytes(p)
        if route != kind:
            raise ValueError("%s routed to %s" % (p, route))
        lab = labels[p]
        rows.append(dict(path=p, label=lab, prediction=text, confidence=conf,
                         correct=text == lab,
                         chars=sum(a == b for a, b in zip(text, lab))))
    groups = []
    for i in range(0, len(paths) - 2, 3):
        group, used = paths[i:i + 3], []

        def fetch():
            p = group[len(used)]
            used.append(p)
            return Path(p).read_bytes()

        t, c, _, _ = solve_with_retry(fetch, kind=kind, max_tries=3, min_conf=gate)
        ok = c >= gate
        groups.append(dict(paths=group, used=len(used), accepted=ok,
                           correct=ok and t == labels[used[-1]], confidence=c))
    accepted = [r for r in rows if r["confidence"] >= gate]
    n_chars = sum(len(r["label"]) for r in rows)
    out = dict(kind=kind, split=part, n=len(rows),
               exact=sum(r["correct"] for r in rows),
               char_accuracy=sum(r["chars"] for r in rows) / n_chars,
               gate=gate, accepted=len(accepted),
               accepted_correct=sum(r["correct"] for r in accepted),
               retry_groups=len(groups),
               within_three_correct=sum(g["correct"] for g in groups),
               mean_fetches=sum(g["used"] for g in groups) / max(len(groups), 1),
               rows=rows, groups=groups)
    if label_free:
        confs = [solve_bytes(p, kind=kind)[1] for p in sorted(glob.glob(RAW[kind]))]
        out["label_free"] = dict(n=len(confs),
                                 above_gate=int(sum(c >= gate for c in confs)),
                                 conf_percentiles={q: float(np.percentile(confs, q))
                                                   for q in (0, 1, 5, 50)})
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("kind", choices=sorted(RAW))
    ap.add_argument("--split", default="test", choices=["validation", "test"])
    ap.add_argument("--output", type=Path)
    ap.add_argument("--no-label-free", action="store_true")
    a = ap.parse_args()
    r = evaluate(a.kind, a.split, not a.no_label_free)
    if a.output:
        a.output.write_text(json.dumps(r, indent=1) + "\n")
    print(json.dumps({k: v for k, v in r.items() if k not in ("rows", "groups")}, indent=1))
    for row in r["rows"]:
        if not row["correct"]:
            print("  miss", row["path"], row["label"], "->", row["prediction"],
                  round(row["confidence"], 3))
