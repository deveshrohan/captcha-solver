#!/usr/bin/env python3
"""Build glyph files for the RERA readers from the TRAIN split only.

    python3 mkglyphs_rera.py tg      # -> solver/tg_rera_glyphs.json
    python3 mkglyphs_rera.py ga      # -> solver/ga_rera_glyphs.json
    python3 mkglyphs_rera.py up      # -> solver/up_rera_glyphs.json
"""
import json
import sys
from collections import defaultdict

import numpy as np
from PIL import Image


def split(name):
    labels = json.load(open("%s_labels.json" % name))
    train = json.load(open("%s_split.json" % name))["train"]
    return [(p, labels[p]) for p in train]


def build_tg():
    from solver import tg_rera as T
    rows = split("tg_rera")
    A = np.stack([np.asarray(Image.open(p).convert("RGB")).astype(int) for p, _ in rows])
    # dot lattice: navy pixels present in (nearly) every training image
    lat = (A == [0, 0, 128]).all(3).mean(0) > 0.9
    D = 1.0 - A.min(3) / 255.0
    D[:, lat] = 0.0
    wins = defaultdict(list)
    for k, (_, label) in enumerate(rows):
        assert len(label) == T.LENGTH, label
        for i, c in enumerate(label):
            x = T.SLOT0 + T.STEP * i
            w = np.zeros((T.H, T.WIN))
            seg = D[k][:, x:x + T.WIN]
            w[:, :seg.shape[1]] = seg
            wins[c].append(w)
    # median over contexts removes neighbours' overhang; then one
    # refinement: subtract the other slots' current templates and re-median
    tpl = {c: np.median(v, axis=0) for c, v in wins.items()}
    for _ in range(2):
        wins = defaultdict(list)
        for k, (_, label) in enumerate(rows):
            for i, c in enumerate(label):
                other = np.zeros((T.H, T.W))
                for j, cj in enumerate(label):
                    if j != i:
                        other = np.maximum(other, T._place(tpl[cj].astype(np.float32), j))
                resid = np.where(other > 0.25, np.nan, D[k])
                x = T.SLOT0 + T.STEP * i
                w = np.full((T.H, T.WIN), np.nan)
                seg = resid[:, x:x + T.WIN]
                w[:, :seg.shape[1]] = seg
                wins[c].append(w)
        new = {}
        for c, v in wins.items():
            m = np.nanmedian(np.stack(v), axis=0)
            new[c] = np.where(np.isnan(m), tpl[c], m)
        tpl = new
    out = {"_meta": {"source": "tg_rera_split.json train (%d server-accepted captures)" % len(rows),
                     "slot0": T.SLOT0, "step": T.STEP, "win": T.WIN},
           "lattice": [[int(y), int(x)] for y, x in zip(*np.nonzero(lat))],
           "glyphs": {c: np.rint(np.clip(t, 0, 1) * 255).astype(int).tolist() for c, t in sorted(tpl.items())}}
    json.dump(out, open("solver/tg_rera_glyphs.json", "w"), separators=(",", ":"))
    print("tg:", len(tpl), "letters:", "".join(sorted(tpl)), "from", len(rows), "captures")


if __name__ == "__main__":
    {"tg": build_tg}[sys.argv[1]]()
