"""CERSAI (cersai.org.in) captcha: `/CERSAI/captcha.jpg`, served after
`home.prg` sets the session.

200x45 RGB JPEG. Six characters, mixed case plus digits, drawn in green ink on
a light background. 60 glyph classes have been seen; `I` and `l` never have.
The portal checks case.

No model. Ink is the dark-green pixel set (R < 100, G < 170, B < 100). The
reader searches left to right over six glyph templates cut at native height,
so c/C and s/S stay distinct by size. Templates come from the 60 training
captures in `cersai_split.json` and are stored in `cersai_glyphs.json`.

Confidence is the smaller of two numbers: the fit (one minus mismatched ink
over total ink) and the score margin to the best different word.

Ported from the blrMent records scraper (`scripts/local/records/captcha.py`),
where it read 40/40 human-labelled held-out captures exactly. 39 of them cleared
the 0.95 gate, and all 39 were correct.
"""
import io
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

W, H = 200, 45
LENGTH = 6
_GLYPHS = Path(__file__).resolve().with_name("cersai_glyphs.json")


def _open(image):
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, (bytes, bytearray)):
        return Image.open(io.BytesIO(image))
    return Image.open(image)


def ink(image):
    """-> (H, W) float32 mask of green ink, or None if the size is wrong."""
    im = _open(image).convert("RGB")
    if im.size != (W, H):
        return None
    a = np.asarray(im)
    return ((a[:, :, 0] < 100) & (a[:, :, 1] < 170) & (a[:, :, 2] < 100)).astype(np.float32)


@lru_cache(maxsize=1)
def templates():
    """-> {width: (chars, stacked templates)}"""
    data = json.loads(_GLYPHS.read_text())
    grouped = {}
    for char, variants in data["glyphs"].items():
        for v in variants:
            a = np.array(v, dtype=np.float32)
            grouped.setdefault(a.shape[1], []).append((char, a))
    return {w: ([c for c, _ in pairs], np.stack([a for _, a in pairs]))
            for w, pairs in grouped.items()}


def solve_image(image):
    """-> (text, confidence). ('', 0.0) for a wrong size or too little ink."""
    a = ink(image)
    if a is None or a.sum() < 100:
        return "", 0.0
    groups = templates()
    cols = a.any(axis=0)
    first, last = np.flatnonzero(cols)[[0, -1]]
    total = float(a.sum())
    # Search over (word, x). Keep runner-up words so the margin can be measured.
    states = [(0.0, "", int(first))]
    for index in range(LENGTH):
        candidates = {}
        for score, word, x in states:
            for width, (chars, glyphs) in groups.items():
                end = x + width
                if end > W:
                    continue
                errors = np.abs(glyphs - a[:, x:end]).sum(axis=(1, 2))
                best = {}
                for c, e in zip(chars, errors):
                    best[c] = min(best.get(c, float("inf")), float(e))
                for c, e in sorted(best.items(), key=lambda p: p[1])[:3]:
                    if e > max(30, width * H * .22):
                        continue
                    # skip only blank gap columns; touching letters are read
                    # as two templates, never one wide one
                    nx = end
                    while nx < W and not cols[nx]:
                        nx += 1
                    if nx - end > 10 and index < LENGTH - 1:
                        continue
                    key = (word + c, nx)
                    candidates[key] = min(candidates.get(key, float("inf")), score + e)
        states = sorted((s, w, x) for (w, x), s in candidates.items())[:50]
        if not states:
            return "", 0.0
    valid = sorted((s, w) for s, w, x in states if x > last)
    if not valid:
        return "", 0.0
    score, word = valid[0]
    runner = next((s for s, w in valid if w != word), score + total)
    fit = max(0.0, 1 - score / total)
    margin = max(0.0, (runner - score) / max(1.0, total))
    return word, float(min(fit, min(1.0, margin / .015)))
