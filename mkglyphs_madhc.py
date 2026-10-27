"""Build solver/madhc_glyphs.json: AHGBold digits, refined on the TRAIN split.

    python3 mkglyphs_madhc.py

Step 1 renders the ten digits from fonts/AHGBold.ttf at 32 px (securimage's
24pt at GD's 96 dpi) in the reader's additive L domain. That alone reads
every labelled image, but GD hints `1` a pixel differently from FreeType, so
step 2 re-estimates each template from train images: with every glyph's
position fixed by a forced-label cover, a digit's template is the median, over
its instances, of the observed L minus every other glyph's contribution.
Held-out (validation/test) images are never opened.
"""
import json
from pathlib import Path

import numpy as np
from PIL import ImageFont

from solver import madhc as M

PAD = 2          # template margin each side, so refinement can move ink


def rendered():
    font = ImageFont.truetype("fonts/AHGBold.ttf", 32)
    out = {}
    for d in M.CHARSET:
        mask = font.getmask(d)
        a = np.asarray(mask, np.float32).reshape(mask.size[1], mask.size[0]) / 255
        # the mask spans the glyph's box relative to (pen x, baseline)
        ox, oy = font.getbbox(d, anchor="ls")[:2]
        L = -np.log(1 - M.OPACITY * a)
        L = np.pad(L, PAD)
        out[d] = (M.BASELINE + oy - PAD, ox - PAD, L.astype(np.float32))
    return out


WIDE = {d: tuple(M.STEP) for d in M.CHARSET}


def refine(tp, train, labels, seen_steps):
    acc = {d: [] for d in M.CHARSET}
    for p in train:
        L = M.observed(p)
        found = M.cover(L, tp, label=labels[p], beam=8, step=WIDE)
        _, word, xs = found[0]
        for d, a, b in zip(word, xs, xs[1:]):
            seen_steps[d].add(b - a)
        for i, (d, x) in enumerate(zip(word, xs)):
            others = np.zeros_like(L)
            for j, (e, xj) in enumerate(zip(word, xs)):
                if j != i:
                    M._add(others, tp[e], xj)
            top, left, g = tp[d]
            h, w = g.shape
            x0 = x + left
            if x0 < 0 or x0 + w > M.W:
                continue
            crop = L[top:top + h, x0:x0 + w] - others[top:top + h, x0:x0 + w]
            # pixels saturated by an overlap carry no information about this glyph
            sat = np.minimum(L, M.L_MAX)[top:top + h, x0:x0 + w] >= M.L_MAX - 0.5
            crop = np.where(sat, np.nan, crop)
            acc[d].append(crop)
    new = {}
    for d, (top, left, g) in tp.items():
        if len(acc[d]) < 3:
            raise ValueError("too few train instances of %r" % d)
        med = np.nanmedian(np.stack(acc[d]), 0)
        med = np.where(np.isnan(med), g, np.clip(med, 0, None))
        med[med < 0.03] = 0
        new[d] = (top, left, med.astype(np.float32))
    return new, {d: len(v) for d, v in acc.items()}


def main():
    labels = json.loads(Path("madhc_labels.json").read_text())
    split = json.loads(Path("madhc_split.json").read_text())
    train, held = split["train"], set(split["test"]) | set(split["validation"])
    if set(train) & held:
        raise ValueError("overlapping split")
    tp = rendered()
    for _ in range(2):
        seen = {d: set() for d in M.CHARSET}
        tp, counts = refine(tp, train, labels, seen)
    out = {"_meta": {"built_from": "train-split", "train_images": len(train),
                     "held_out": len(held), "font": "AHGBold.ttf @ 32px",
                     "method": "font render, median-refined on train covers",
                     "instances": counts},
           "glyphs": {d: {"top": int(top), "left": int(left),
                          "steps": sorted(seen[d]),
                          "L": np.round(g, 3).tolist()}
                      for d, (top, left, g) in tp.items()}}
    Path("solver/madhc_glyphs.json").write_text(json.dumps(out, separators=(",", ":")) + "\n")
    print("madhc train", len(train), "held out", len(held), "instances", counts,
          "steps", {d: sorted(v) for d, v in seen.items()})


if __name__ == "__main__":
    main()
