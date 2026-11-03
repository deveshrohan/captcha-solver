"""MCA CDM export captcha (mcacdm.nic.in): `captcha.php?context=company|llp`,
on the company and LLP master-data export pages. Served since ~2026-09-08.

120x38 RGB PNG with exactly two colours: ink #cc0000 on #e0e0e0, no noise.
Six characters from [A-Za-z0-9], all 62 seen across the company and llp
contexts, drawn by PHP GD `imagestring()` in built-in font 5 (gdFontGiant,
9x15 cells) at pen origin (30, 10), one cell per character.

The digit and lowercase cells match libgd's `gdfontg.c` bitmaps (in
tests/gdfontgiant_alnum.json) byte for byte. The uppercase cells come from
the same BDF font but are not in that excerpt, so they were cut from the
train split of mcacdm_split.json.

No model. Each of the six cells is matched exactly against 62 bitmaps
(`mcacdm_glyphs.json`). The closest pair, `I`/`l`, is 5 pixels apart, so a
pixel-exact match is never ambiguous. To allow for a moved pen, the origin is
searched within +-2 px. Confidence is fidelity: 1 - (mismatched pixels, plus
ink outside the six cells) / total ink. A clean read scores exactly 1.0.
"""
import io
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

W, H = 120, 38
LENGTH = 6
ORIGIN = (30, 10)
CELL_H, CELL_W = 15, 9
SLACK = 2
_GLYPHS = Path(__file__).resolve().with_name("mcacdm_glyphs.json")


def _open(image):
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, (bytes, bytearray)):
        return Image.open(io.BytesIO(image))
    return Image.open(image)


@lru_cache(maxsize=1)
def glyphs():
    """-> (chars, (n, 15, 9) uint8 stack)"""
    data = json.loads(_GLYPHS.read_text())["glyphs"]
    chars = sorted(data)
    return chars, np.stack([np.array([[int(b) for b in row] for row in data[c]], np.uint8)
                            for c in chars])


def ink(image):
    """-> (H, W) uint8 mask of red ink, or None for a wrong size."""
    im = _open(image).convert("RGB")
    if im.size != (W, H):
        return None
    a = np.asarray(im).astype(np.int16)
    return ((a[:, :, 0] - a[:, :, 1] > 80) & (a[:, :, 0] - a[:, :, 2] > 80)).astype(np.uint8)


def read(mask, origin=ORIGIN):
    """-> (word, mismatched pixels inside the cells, ink inside the cells)"""
    chars, stack = glyphs()
    x0, y0 = origin
    word, err, inside = "", 0, 0
    for k in range(LENGTH):
        cell = mask[y0:y0 + CELL_H, x0 + CELL_W * k:x0 + CELL_W * (k + 1)]
        if cell.shape != (CELL_H, CELL_W):
            return "", 1 << 30, 0
        d = np.abs(stack.astype(np.int16) - cell).sum(axis=(1, 2))
        i = int(d.argmin())
        word += chars[i]
        err += int(d[i])
        inside += int(cell.sum())
    return word, err, inside


def solve_image(image):
    """-> (text, confidence). ('', 0.0) for a wrong size or a blank image."""
    m = ink(image)
    if m is None:
        return "", 0.0
    total = int(m.sum())
    if total < 30:
        return "", 0.0
    best = None
    for dy in range(-SLACK, SLACK + 1):
        for dx in range(-SLACK, SLACK + 1):
            word, err, inside = read(m, (ORIGIN[0] + dx, ORIGIN[1] + dy))
            cost = err + (total - inside)           # stray ink counts against
            if best is None or cost < best[0]:
                best = (cost, word)
    cost, word = best
    return word, float(max(0.0, 1.0 - cost / total))
