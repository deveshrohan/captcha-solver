"""Adjudicate two BLIND visual passes and pin train/validation/test splits.

    python3 finalize_hc_labels.py madhc

Inputs: KIND_pass1.json and KIND_pass2.json (two independent labellers, no
model output consulted) and optionally KIND_adjudication.json
({path: {"final": label-or-null, "why": ...}}) for re-viewed disagreements.
Unreadable images (null, or containing "?") are dropped before splitting.

The split is pinned by filename. Once KIND_split.json exists, its test and
validation sets never change: newly labelled images only ever join train.
"""
import argparse
import json
import random
from pathlib import Path

LENGTH = {"gujhc": 5, "madhc": 6, "calhc": 6, "karhc": 6}
SEED = 20261027
N_TEST, N_VAL = 30, 10


def main(kind):
    one = json.loads(Path(kind + "_pass1.json").read_text())
    two = json.loads(Path(kind + "_pass2.json").read_text())
    adj_path = Path(kind + "_adjudication.json")
    decisions = json.loads(adj_path.read_text()) if adj_path.exists() else {}
    if one.keys() != two.keys():
        raise ValueError("each image needs both passes")
    labels, dropped = {}, []
    for path, value in one.items():
        if path in decisions:
            final = decisions[path]["final"]
        elif value == two[path]:
            final = value
        else:
            raise ValueError("unadjudicated disagreement: %s %r vs %r"
                             % (path, value, two[path]))
        if final is None or "?" in final:
            dropped.append(path)
        elif len(final) != LENGTH[kind]:
            raise ValueError("length: %s %r" % (path, final))
        else:
            labels[path] = final
    split_path = Path(kind + "_split.json")
    paths = sorted(labels)
    if split_path.exists():
        old = json.loads(split_path.read_text())
        if not set(old["test"] + old["validation"]) <= set(labels):
            raise ValueError("a pinned eval image lost its label")
        test, val = old["test"], old["validation"]
    else:
        if len(paths) < N_TEST + N_VAL + 10:
            raise ValueError("need >= %d readable images" % (N_TEST + N_VAL + 10))
        shuffled = paths[:]
        random.Random(SEED).shuffle(shuffled)
        test, val = sorted(shuffled[:N_TEST]), sorted(shuffled[N_TEST:N_TEST + N_VAL])
    held = set(test) | set(val)
    train = [p for p in paths if p not in held]
    split = {"seed": SEED, "test": test, "validation": val, "train": train,
             "dropped": dropped,
             "rule": "two independent blind visual passes; disagreements "
                     "re-viewed at pixel scale; unreadable dropped; no model labels"}
    Path(kind + "_labels.json").write_text(json.dumps(labels, indent=1, sort_keys=True) + "\n")
    split_path.write_text(json.dumps(split, indent=1) + "\n")
    print(kind, "labelled", len(labels), "dropped", len(dropped), "train",
          len(train), "val", len(val), "test", len(test))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("kind", choices=sorted(LENGTH))
    main(p.parse_args().kind)
