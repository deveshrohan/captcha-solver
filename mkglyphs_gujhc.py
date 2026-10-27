"""Build solver/gujhc_glyphs.json: DejaVu Sans Mono 40, refined on the TRAIN split.

    python3 mkglyphs_gujhc.py          (needs fontTools at build time only)

Advances are not fitted. DefaultWordRenderer advances by
`(int) glyphVector.getVisualBounds().getWidth()`, the width of the glyph's
OUTLINE at 40 px, truncated -- computed here from the font file. Measured:
truncation reads validation 10/10, rounding 1/10, so the pen model is exact.

Templates start as FreeType renders (Java rasterises a pixel or so
differently) and are re-estimated from train images. Since every pen
position follows from the label, no search is involved: a glyph's template is
the median over its instances of observed L minus every other glyph, using
only pixels that are neither under the noise line nor saturated by an overlap.
Held-out images are never opened.
"""
import json
from pathlib import Path

import numpy as np
from PIL import ImageFont

from solver import gujhc as G

PAD = 2
FONT = "fonts/DejaVuSansMono.ttf"
SIZE = 40


def advances():
    from fontTools.pens.boundsPen import BoundsPen
    from fontTools.ttLib import TTFont
    t = TTFont(FONT)
    gs, cmap, upm = t.getGlyphSet(), t.getBestCmap(), t["head"].unitsPerEm
    out = {}
    for c in G.CHARSET:
        pen = BoundsPen(gs)
        gs[cmap[ord(c)]].draw(pen)
        x0, _, x1, _ = pen.bounds
        out[c] = int((x1 - x0) * SIZE / upm)       # Java's (int) truncation
    return out


def rendered(adv):
    font = ImageFont.truetype(FONT, SIZE)
    out = {}
    for c in G.CHARSET:
        m = font.getmask(c)
        a = np.asarray(m, np.float32).reshape(m.size[1], m.size[0]) / 255
        l, t, _, _ = font.getbbox(c, anchor="ls")
        L = -np.log(np.clip(1 - a, 0.5 / G.BG_G, 1))
        out[c] = (G.BASELINE + t - PAD, l - PAD, np.pad(L, PAD).astype(np.float32), adv[c])
    return out


def pens(label, tp):
    xs = [G.X0]
    for c in label[:-1]:
        xs.append(xs[-1] + tp[c][3])
    return xs


def refine(tp, observed, labels):
    acc = {c: [] for c in G.CHARSET}
    for p, (L, valid) in observed.items():
        lab = labels[p]
        xs = pens(lab, tp)
        for i, (c, x) in enumerate(zip(lab, xs)):
            others = np.zeros_like(L)
            for j, (e, xj) in enumerate(zip(lab, xs)):
                if j != i:
                    G._add(others, tp[e], xj)
            top, left, g, _ = tp[c]
            h, w = g.shape
            win = (slice(top, top + h), slice(x + left, x + left + w))
            crop = L[win] - others[win]
            # An opaque pixel no other glyph touches says "this glyph is solid
            # here" -- keep it (at L_MAX). Only an opaque pixel that another
            # glyph also covers is uninformative about this one.
            sat = L[win] >= G.L_MAX - 0.05
            bad = ~valid[win] | (sat & (others[win] > 0.05))
            acc[c].append(np.where(bad, np.nan, crop))
    new = {}
    for c, (top, left, g, adv) in tp.items():
        if acc[c]:
            with np.errstate(all="ignore"):
                med = np.nanmedian(np.stack(acc[c]), 0)
            med = np.where(np.isnan(med), g, np.clip(med, 0, G.L_MAX))
            med[med < 0.02] = 0
        else:                           # unseen in train: keep the font render
            med = g
        new[c] = (top, left, med.astype(np.float32), adv)
    return new, {c: len(v) for c, v in acc.items()}


def main():
    labels = json.loads(Path("gujhc_labels.json").read_text())
    split = json.loads(Path("gujhc_split.json").read_text())
    train, held = split["train"], set(split["test"]) | set(split["validation"])
    if set(train) & held:
        raise ValueError("overlapping split")
    observed = {p: G.observed(p) for p in train}
    tp = rendered(advances())
    for _ in range(3):
        tp, counts = refine(tp, observed, labels)
    out = {"_meta": {"built_from": "train-split", "train_images": len(train),
                     "held_out": len(held), "font": "DejaVuSansMono.ttf @ 40px",
                     "advance": "int(outline width at 40px), from the font file",
                     "method": "font render, median-refined at label-implied pen positions",
                     "instances": counts},
           "glyphs": {c: {"top": int(top), "left": int(left), "adv": int(adv),
                          "L": np.round(g, 3).tolist()}
                      for c, (top, left, g, adv) in tp.items()}}
    Path("solver/gujhc_glyphs.json").write_text(json.dumps(out, separators=(",", ":")) + "\n")
    print("gujhc train", len(train), "held out", len(held),
          "unseen", [c for c, n in counts.items() if n == 0],
          "min instances", min(counts.values()))


if __name__ == "__main__":
    main()
