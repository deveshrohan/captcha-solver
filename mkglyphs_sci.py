#!/usr/bin/env python3
"""Build solver/sci_glyphs.json from the TRAIN split of sci_labels.json.

    python3 mkglyphs_sci.py

Each labelled render is split into ink pieces (solver.sci.components). A
render is used only when its pieces line up one-to-one with the label's
tokens. The pieces are stored bit-packed; solver.sci adds the rotated
copies (+-6, +-12, +-18 degrees) when it loads them. If the operands' digits touch, the pieces and tokens don't line
up, and the render is skipped rather than guessed at. Each kept piece adds
its own exemplar plus copies rotated by +-6, +-12 and +-18 degrees, since every
glyph gets its own rotation.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from solver import sci as S

ROOT = Path(__file__).resolve().parent


def main():
    labels = json.loads((ROOT / "sci_labels.json").read_text())
    train = json.loads((ROOT / "sci_split.json").read_text())["train"]
    glyphs = []
    used = skipped = 0
    for rel in train:
        label = labels[rel]
        comps = S.components(S.ink(ROOT / rel))
        tokens = list(label)
        if len(comps) == len(tokens) - 1 and label.startswith("10"):
            tokens = ["10"] + tokens[2:]          # the 1 and 0 touch
        if len(comps) != len(tokens):
            skipped += 1
            continue
        used += 1
        for ch, (_, _, bm) in zip(tokens, comps):
            glyphs.append([ch] + S.pack(bm))
    out = {"_meta": {"built_from": "sci_split.json train, human labels",
                     "renders_used": used, "renders_skipped_merged": skipped,
                     "exemplars": len(glyphs),
                     "note": "bit-packed piece bitmaps [class, h, w, base64]; "
                             "solver.sci rotates and scales them at load"},
           "glyphs": glyphs}
    (ROOT / "solver" / "sci_glyphs.json").write_text(json.dumps(out, separators=(",", ":")))
    print("used", used, "skipped", skipped, "exemplars", len(glyphs),
          "classes", sorted({g[0] for g in glyphs}))


if __name__ == "__main__":
    main()
