"""Andhra Pradesh High Court (digi-courts.aphc.ap.gov.in/csis_ap) captcha:
`loadCaptcha?date=...`.

110x33 JPEG with white text on a dark background. The text is a sum written
as a word, e.g. `77plus8`, and the portal wants the result (`85`). Operands
are one to three digits each.

No model. The reader binarises at 200, splits the text on blank columns, and
matches each glyph against 14 fixed 16x13 bitmaps (`0`-`9` plus `l p s u`),
allowing a shift of one pixel either way. It first finds the four-letter run
"plus". A noise line through the `p` can make it read as `u`, which still
counts as a letter. Then it reads the glyphs on either side as digits.

Confidence comes from the worst digit mismatch, measured in pixels:
1 - worst/60. Live reads had a worst mismatch of 7 or less, which gives a
confidence of at least 0.88. The original reader refused anything above 30,
which is the 0.5 gate here.

Ported from the litigation scraper (`crawlers/aphc_site.py:read_captcha`).
That reader was built from 100 live captchas, all of which it read correctly.
"""
import io

import numpy as np
from PIL import Image

W, H = 110, 33
_GH, _GW = 16, 13
# 16 rows x 13 columns per glyph; each row is a 13-bit mask, MSB leftmost.
_GLYPHS = {
    '0': "0780 0fc0 1ce0 1860 1860 1860 1860 1860 1860 1860 0ce0 0fc0 0380 0000 0000 0000",
    '1': "0300 0700 0f00 1b00 1300 0300 0300 0300 0300 0300 0300 0300 0300 0000 0000 0000",
    '2': "0780 1fc0 08e0 0060 0060 00e0 00c0 0180 0300 0600 0c00 1fe0 1fe0 0000 0000 0000",
    '3': "0780 1fc0 00e0 0060 00c0 0780 07c0 00e0 0060 0060 00e0 1fc0 0f00 0000 0000 0000",
    '4': "00c0 01c0 01c0 02c0 06c0 0cc0 08c0 18c0 1ff0 1ff0 00c0 00c0 00c0 0000 0000 0000",
    '5': "0fc0 0fc0 0c00 0c00 0c00 0f80 1fe0 00e0 0060 0060 00e0 1fc0 0f00 0000 0000 0000",
    '6': "01c0 07e0 0e00 0c00 1800 1bc0 1fe0 1c60 1860 1860 1c60 0fe0 0380 0000 0000 0000",
    '7': "1fe0 1fe0 0060 00e0 00c0 00c0 0180 0180 0380 0300 0700 0600 0e00 0000 0000 0000",
    '8': "0780 0fc0 1c60 1860 0ce0 07c0 0780 0ce0 1860 1860 1860 1fe0 0780 0000 0000 0000",
    '9': "0700 0fc0 1ce0 1860 1860 1860 1fe0 0f60 0060 0060 00c0 0f80 0e00 0000 0000 0000",
    'l': "1800 1800 1800 1800 1800 1800 1800 1800 1800 1800 1800 1800 1800 1800 0000 0000",
    'p': "1180 1fc0 1ce0 1860 1860 1860 1860 1ce0 1fc0 1b80 1800 1800 1800 1800 0000 0000",
    's': "0780 1fc0 1800 1c00 0f00 03c0 00c0 00c0 1fc0 0f00 0000 0000 0000 0000 0000 0000",
    'u': "0c30 0c30 0c30 0c30 0c30 1c30 0c30 0c70 0ff0 0730 0000 0000 0000 0000 0000 0000",
}
_TEMPLATES = {k: np.array([[(int(h, 16) >> (_GW - 1 - c)) & 1 for c in range(_GW)]
                           for h in v.split()]) for k, v in _GLYPHS.items()}
_SCALE = 60.0           # confidence = 1 - worst/_SCALE; the old refusal at 30 = 0.5


def templates():
    return _TEMPLATES


def _open(image):
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, (bytes, bytearray)):
        return Image.open(io.BytesIO(image))
    return Image.open(image)


def _best(g, keys):
    c = np.zeros((_GH, _GW), dtype=int)
    h, w = min(g.shape[0], _GH), min(g.shape[1], _GW)
    c[:h, :w] = g[:h, :w]
    out = None
    for k in keys:
        t = _TEMPLATES[k]
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                d = int(np.abs(np.roll(np.roll(c, dy, 0), dx, 1) - t).sum())
                if out is None or d < out[0]:
                    out = (d, k)
    return out


def read(image):
    """-> (expression like '77plus8', answer int, worst digit distance), or
    None when the image is not a readable `<a>plus<b>`."""
    im = np.asarray(_open(image).convert("L")) > 200
    cols, glyphs, start = im.any(axis=0), [], None
    for x, on in enumerate(list(cols) + [False]):
        if on and start is None:
            start = x
        elif not on and start is not None:
            sub = im[:, start:x]
            rows = np.where(sub.any(axis=1))[0]
            g = sub[rows[0]:rows[-1] + 1]
            if g.sum() >= 6:                 # specks left by a noise line
                glyphs.append(g)
            start = None
    labels = [_best(g, _GLYPHS)[1] for g in glyphs]
    i = next((j for j in range(1, len(glyphs) - 4)
              if all(x in "plus" for x in labels[j:j + 4])), None)
    if i is None or not 1 <= len(glyphs) - i - 4 <= 3 or i > 3:
        return None
    digits = [_best(g, "0123456789") for g in glyphs[:i] + glyphs[i + 4:]]
    a = "".join(k for _, k in digits[:i])
    b = "".join(k for _, k in digits[i:])
    return "%splus%s" % (a, b), int(a) + int(b), max(d for d, _ in digits)


def solve_image(image):
    """-> (answer as text, confidence). ('', 0.0) when unreadable."""
    pil = _open(image)
    if pil.size != (W, H):
        return "", 0.0
    r = read(pil)
    if r is None:
        return "", 0.0
    _, answer, worst = r
    return str(answer), float(max(0.0, 1.0 - worst / _SCALE))
