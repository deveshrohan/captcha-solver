#!/usr/bin/env python3
"""Score the Supreme Court reader on the test split -> sci_eval.json.

    python3 eval_sci.py

Rows are single renders. Groups are the renders of one captcha (one
session), scored both one render at a time (fetch until accepted) and
voted with solver.sci.solve_group.
"""
import json
import re
import time
from collections import defaultdict
from pathlib import Path

from solver import sci as S
from solver.api import GATES

ROOT = Path(__file__).resolve().parent


def answer(expr):
    a, op, b = re.fullmatch(r"(\d+)([+\-x])(\d+)", expr).groups()
    return str(S.OPS[op](int(a), int(b)))


def main():
    labels = json.loads((ROOT / "sci_labels.json").read_text())
    test = json.loads((ROOT / "sci_split.json").read_text())["test"]
    gate = GATES.get("sci", 0.6)          # until wired into solver.api
    rows, groups = [], defaultdict(list)
    S.templates()                                  # time warm solves only
    spent = 0.0
    for rel in test:
        want = answer(labels[rel])
        r = S.read(ROOT / rel)
        t0 = time.time()
        text, conf = S.solve_image(ROOT / rel)
        spent += time.time() - t0
        rows.append(dict(path=rel, label=labels[rel], expression=r and r[0],
                         prediction=text, answer=want, confidence=conf,
                         correct=text == want))
        groups[rel.rsplit("_", 1)[0]].append(rows[-1])
    ms = 1000 * spent / len(test)
    acc = [r for r in rows if r["confidence"] >= gate]
    first = [next((r for r in g if r["confidence"] >= gate), None) for g in groups.values()]
    voted = []
    for key, g in groups.items():
        t, c = S.solve_group([ROOT / r["path"] for r in g])
        voted.append(t == g[0]["answer"])
    out = dict(kind="sci", split="test", n=len(rows),
               exact=sum(r["correct"] for r in rows),
               expression_exact=sum(r["expression"] == r["label"] for r in rows),
               gate=gate, accepted=len(acc),
               accepted_correct=sum(r["correct"] for r in acc),
               groups=len(groups),
               group_first_accepted_correct=sum(bool(f and f["correct"]) for f in first),
               group_none_accepted=sum(f is None for f in first),
               group_voted_correct=sum(voted),
               ms_per_solve=round(ms, 1), rows=rows)
    (ROOT / "sci_eval.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}))
    for r in rows:
        if not r["correct"]:
            print("wrong", r["path"], r["label"], r["expression"], round(r["confidence"], 3))


if __name__ == "__main__":
    main()
