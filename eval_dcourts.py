"""Score the district-courts reader on the pinned test split.

    python3 eval_dcourts.py

The gate comes from the val split: the lowest confidence at which voted
reads (over 3 renders) were all correct, with a 0.5 floor. The test split
is then scored per render (one fetch) and per code voted over the first k
renders (k = 2..4). Writes dcourts_eval.json.
"""
import json

from download_dcourts import load_manifest
from solver import dcourts as D


def rows(split, man, L, S):
    out = []
    for code in S[split]:
        paths = ["dcourts_raw/" + n for n in man[code]["renders"]]
        out.append((code, L[code], paths))
    return out


def main():
    man = load_manifest()
    L = json.load(open("dcourts_labels.json"))
    S = json.load(open("dcourts_split.json"))

    # gate on val: voted over 3 renders
    vals = []
    for code, lab, paths in rows("val", man, L, S):
        t, c = D.solve_group(paths[:3])
        vals.append((c, t == lab))
    wrong = [c for c, ok in vals if not ok]
    gate = max(0.5, round(max(wrong) + 0.01, 2)) if wrong else 0.5

    test = rows("test", man, L, S)
    single = []
    for code, lab, paths in test:
        for p in paths:
            t, c = D.solve_image(p)
            single.append(dict(path=p, label=lab, prediction=t, confidence=round(c, 4),
                               correct=t == lab))
    voted = {}
    for k in (2, 3, 4):
        res = []
        for code, lab, paths in test:
            t, c = D.solve_group(paths[:k])
            res.append(dict(code=code, label=lab, prediction=t, confidence=round(c, 4),
                            correct=t == lab, accepted=c >= gate))
        voted[k] = res
    char = sum(sum(a == b for a, b in zip(r["prediction"], r["label"])) for r in single)
    out = {
        "kind": "dcourts", "split": "test", "codes": len(test), "renders": len(single),
        "gate": gate, "gate_from": "val split, voted over 3 renders",
        "single_render_exact": sum(r["correct"] for r in single),
        "single_render_char_accuracy": round(char / (5 * len(single)), 4),
        "voted": {str(k): dict(exact=sum(r["correct"] for r in v),
                               accepted=sum(r["accepted"] for r in v),
                               accepted_correct=sum(r["accepted"] and r["correct"] for r in v))
                  for k, v in voted.items()},
        "single_rows": single, "voted_rows": voted[3],
    }
    json.dump(out, open("dcourts_eval.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if not k.endswith("rows")}, indent=1))


if __name__ == "__main__":
    main()
