"""CESTAT (cestat.gov.in) captcha: `/captcha?rand=<random>`, served after a
GET of `/casestatus` (which sets the ci_session and csrf cookies).

100x35 RGB JPEG. Black glyphs on cyan (#40-60 F0 F0), drawn by PHP GD
`imagettftext` in Latin Modern Roman 10 Bold at 18 pt (24 px at GD's
96 dpi). Measured on 160 labelled glyphs: pen x = 5 + 23*i, baseline y = 26,
upright, with no jitter at all.

**The image cannot hold the whole answer.** The code is six characters
(the form field is `maxlength=6`), but at a 23 px pitch only cells 0-3 fit
in 100 px. Cell 4's pen is at x = 97, so only a 3 px sliver of it shows, and
cell 5 starts at x = 120, off the image. This reader returns the four
visible characters. When it was measured (2026-11-03), the portal's own page
script filled the field with `111111` and hid the captcha, so the site did
not appear to check it at all. That would explain the scraper's "~99%"
Tesseract figure.

No model. The 62 glyph renders are baked into `cestat_glyphs.json`, so the
font is not needed at runtime. The reader does a beam cover over the four
cells, scoring each prefix only on columns that no later glyph can reach.
Confidence is the margin to the best different word, capped by fidelity
(the share of visible ink the cover explains).
"""
import io
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

W, H = 100, 35
LENGTH = 6            # what the site's field takes
VISIBLE = 4           # what the image shows in full
PITCH, PEN0, LIMIT = 23, 5, 97   # cell i's pen is PEN0 + PITCH*i; cell 4 starts at LIMIT
BEAM = 30
_GLYPHS = Path(__file__).resolve().with_name("cestat_glyphs.json")


def _open(image):
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, (bytes, bytearray)):
        return Image.open(io.BytesIO(image))
    return Image.open(image)


# The pool the portal draws from: 0-9, A-Z, a-z without O and o. Neither
# appeared once in 600 harvested glyphs (about 10 of each were expected).
CHARSET = "0123456789ABCDEFGHIJKLMNPQRSTUVWXYZabcdefghijklmnpqrstuvwxyz"


@lru_cache(maxsize=1)
def templates():
    """-> (charset, {char: (left offset from pen, alpha float32 HxW)})"""
    data = json.loads(_GLYPHS.read_text())
    return CHARSET, {
        c: (g["left"], np.array(g["alpha"], np.float32) / 255.0)
        for c, g in data["glyphs"].items() if c in CHARSET}


@lru_cache(maxsize=1)
def _cells():
    """Per cell: (chars, stacked glyph alphas placed on the full width)."""
    charset, tp = templates()
    out = []
    for i in range(VISIBLE):
        stack = np.zeros((len(charset), H, W), np.float32)
        for k, c in enumerate(charset):
            left, g = tp[c]
            x0 = PEN0 + PITCH * i + left
            a, b = max(0, x0), min(W, x0 + g.shape[1])
            stack[k, :, a:b] = g[:, a - x0:b - x0]
        out.append(stack)
    return charset, out


def ink(image):
    """-> (H, W) ink coverage in [0, 1] from the green channel, or None for
    a wrong size. The cyan background is G~240 and black ink is G~0."""
    im = _open(image).convert("RGB")
    if im.size != (W, H):
        return None
    g = np.asarray(im, np.float32)[:, :, 1]
    bg = float(np.median(g))
    if bg < 120:                           # not this captcha's background
        return None
    return np.clip((bg - g) / bg, 0.0, 1.0)


def cover(A, beam=BEAM):
    """-> [(error, word)] best first, over the VISIBLE cells.

    Glyph i reaches no further left than pen_i + spill, so the columns
    before that are final once glyphs 0..i-1 are placed. Each step adds only
    the error of the newly final columns [start, end)."""
    charset, cells = _cells()
    _, tp = templates()
    spill = min(left for left, _ in tp.values())
    states = [(0.0, "", np.zeros((H, W), np.float32))]
    start = 0
    for i in range(VISIBLE):
        end = LIMIT if i == VISIBLE - 1 else PEN0 + PITCH * (i + 1) + spill
        stack = cells[i]
        cand = []
        for err, word, canvas in states:
            win = np.maximum(canvas[None, :, start:end], stack[:, :, start:end])
            es = err + np.abs(win - A[None, :, start:end]).sum(axis=(1, 2))
            for k in np.argsort(es)[:beam]:
                cand.append((float(es[k]), word + charset[k], canvas, k))
        cand.sort(key=lambda s: s[0])
        states = [(e, w, np.maximum(canvas, stack[k])) for e, w, canvas, k in cand[:beam]]
        start = end
    best = {}
    for e, w, _ in states:
        best.setdefault(w, e)
    return sorted((e, w) for w, e in best.items())


def solve_image(image):
    """-> (the four visible characters, confidence); ('', 0.0) if unreadable."""
    A = ink(image)
    if A is None:
        return "", 0.0
    total = float(A[:, :LIMIT].sum())
    if total < 60:
        return "", 0.0
    found = cover(A)
    err, word = found[0]
    fidelity = 1.0 - err / total
    margin = (found[1][0] - err) / total if len(found) > 1 else 1.0
    if fidelity < 0.6:
        return word, 0.0
    return word, float(min(1.0, margin / 0.05))
