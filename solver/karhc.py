"""Karnataka High Court captcha (judiciary.karnataka.gov.in/captcha.php), 100x20.

A PHP GD script: six digits drawn with `imagestring(..., 5, ...)` -- GD's
built-in font 5, gdFontGiant, a 9x15 bitmap font -- in black at a FIXED
origin (25, 0), plus faint lavender noise lines and `+` specks, saved as a
libgd JPEG at default quality. Measured on every harvested image: the best
grid origin is (25, 0), never anything else.

So the read is six independent bitmap lookups, like NGT's, against the font's
own bitmaps transcribed from libgd's `src/gdfontg.c` -- external ground truth,
not templates learned from images, so there is no train split to leak.
JPEG blurs the edges, so no cell matches byte for byte; each cell takes the
digit with the least absolute difference on the ink map.

Ink is black and the noise is light (its darkest channel stays high), so the
ink map `(255 - max(r, g, b)) / 200` keeps the text and drops the noise.

Confidence is a measurement, not a softmax: per cell, how far the cell sits
from the runner-up digit RELATIVE to how far apart those two bitmaps are
(1.0 = on the winning glyph, 0.0 = exactly between the two), capped by how
much of the cell's ink the winner explains and by any ink outside the grid.
The image's confidence is its weakest cell.
"""
import io
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

W, H = 100, 20
LENGTH = 6
CHARSET = "0123456789"
CELL_W, CELL_H = 9, 15
ORIGIN = (25, 0)
SLACK = 1               # also try the origin +/- 1 px, in case the script moves
_HERE = Path(__file__).resolve().parent


def open_image(image):
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, (bytes, bytearray)):
        return Image.open(io.BytesIO(image))
    return Image.open(image)


@lru_cache(maxsize=1)
def glyphs():
    data = json.loads((_HERE / "karhc_glyphs.json").read_text())
    if data.get("_meta", {}).get("built_from") != "libgd-source":
        raise ValueError("karhc glyphs must come from libgd's font source")
    return {c: np.array([[int(b) for b in row] for row in rows], np.float32)
            for c, rows in data["glyphs"].items()}


def ink_map(image):
    im = open_image(image).convert("RGB")
    if im.size != (W, H):
        raise ValueError("karhc captcha must be %dx%d, got %r" % (W, H, im.size))
    a = np.asarray(im, np.float32)
    return np.clip((255 - a.max(2)) / 200, 0, 1)


def _cells(ink, x0, y0):
    out = []
    for k in range(LENGTH):
        x = x0 + CELL_W * k
        crop = np.zeros((CELL_H, CELL_W), np.float32)
        ys, xs = max(0, y0), max(0, x)
        sub = ink[ys:y0 + CELL_H, xs:x + CELL_W]
        crop[ys - y0:ys - y0 + sub.shape[0], xs - x:xs - x + sub.shape[1]] = sub
        out.append(crop)
    return out


def read(ink):
    """-> (word, total_error, per-cell [(err, runner_up_err, ink, bitmap
    distance winner<->runner-up)], origin)."""
    tp = glyphs()
    best = None
    for dy in range(-SLACK, SLACK + 1):
        for dx in range(-SLACK, SLACK + 1):
            x0, y0 = ORIGIN[0] + dx, ORIGIN[1] + dy
            word, tot, cells = "", 0.0, []
            for crop in _cells(ink, x0, y0):
                errs = sorted((float(np.abs(crop - t).sum()), c) for c, t in tp.items())
                word += errs[0][1]
                tot += errs[0][0]
                gap = float(np.abs(tp[errs[0][1]] - tp[errs[1][1]]).sum())
                cells.append((errs[0][0], errs[1][0], float(crop.sum()), gap))
            if best is None or tot < best[1]:
                best = (word, tot, cells, (x0, y0))
    return best


def solve_image(image):
    """-> (text, confidence)."""
    ink = ink_map(image)
    if ink.sum() < 20:                       # blank: six digits carry far more
        return "", 0.0
    word, _, cells, (x0, y0) = read(ink)
    # ink the grid does not account for (a moved or longer text)
    inside = sum(c.sum() for c in _cells(ink, x0, y0))
    stray = max(0.0, float(ink.sum()) - inside) / max(float(ink.sum()), 1.0)
    conf = 1.0 - stray
    for err, second, mass, gap in cells:
        fit = 1.0 - err / max(mass, 20.0)      # a digit has >= ~25 ink pixels
        margin = (second - err) / gap          # 1 on the glyph, 0 halfway
        conf = min(conf, fit, margin)
    return word, float(max(0.0, min(1.0, conf)))
