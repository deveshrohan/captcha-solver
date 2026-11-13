"""Score the MCA v2 reader on the pinned TEST split of mca2_split.json.

Reads through solver.api (the routing a caller gets), so this checks the
200x80 router as well as the model.

    python3 eval_mca2.py [validation|test]
"""
import json
import sys

from solver.api import GATES, solve_bytes

which = sys.argv[1] if len(sys.argv) > 1 else "test"
labels = json.load(open("mca2_labels.json"))
paths = json.load(open("mca2_split.json"))[which]
gate = GATES["mca2"]
exact = chars = acc = acc_ok = 0
rows = []
for p in paths:
    text, conf, kind = solve_bytes(open(p, "rb").read())
    assert kind == "mca2", (p, kind)
    lab = labels[p]
    ok = text == lab
    exact += ok
    chars += sum(a == b for a, b in zip(text, lab))
    if conf >= gate:
        acc += 1
        acc_ok += ok
    rows.append((conf, ok, p, lab, text))
n = len(paths)
print("%s: %d/%d exact (%.1f%%), %d/%d chars (%.1f%%)"
      % (which, exact, n, 100 * exact / n, chars, 6 * n, 100 * chars / (6 * n)))
print("gate %.2f: %d/%d accepted, %d of them correct" % (gate, acc, n, acc_ok))
for conf, ok, p, lab, text in sorted(rows):
    if not ok:
        print("  miss %s  label %s  read %s  conf %.3f" % (p, lab, text, conf))
print("lowest-confidence correct read: %.3f" % min(c for c, ok, *_ in rows if ok))
