"""Telangana RERA (rerait.telangana.gov.in) search captcha:
`/SearchList/SearchCaptcha`, fetched after the search page in the same session.

158x48 RGBA PNG (fully opaque). Five uppercase letters in navy Times Italic,
ClearType-rendered, on white with a fixed lattice of navy dots. Measured on
70 server-accepted captures:

* the dot lattice is identical in every image (`lattice` in the glyph file);
* each letter's origin is exactly x = 12 + 25*i. A letter seen in two slots
  differs by 0 px of shift, so pen positions don't depend on the text;
* a letter renders identically wherever it appears, so a template is a
  per-letter darkness patch.

No model. Darkness is 1 - min(R, G, B)/255, with lattice pixels set to 0. A
read places five templates at their slot origins, composes them by per-pixel
max, and descends one slot at a time to the lowest L1 error. Italic
overhangs reach into the next slot, so slots are fitted jointly, not one by
one. Confidence is the smaller of two numbers: fidelity (1 - error/ink) and
the worst slot's margin to its runner-up letter.

Templates come from the train split only (`tg_rera_split.json`) and are
stored in `tg_rera_glyphs.json`.
"""
import io
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

W, H = 158, 48
LENGTH = 5
SLOT0, STEP, WIN = 12, 25, 34
_GLYPHS = Path(__file__).resolve().with_name("tg_rera_glyphs.json")


def _open(image):
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, (bytes, bytearray)):
        return Image.open(io.BytesIO(image))
    return Image.open(image)


@lru_cache(maxsize=1)
def templates():
    """-> (chars, stack (n, H, WIN) float32, lattice mask (H, W) bool)"""
    data = json.loads(_GLYPHS.read_text())
    lat = np.zeros((H, W), bool)
    for y, x in data["lattice"]:
        lat[y, x] = True
    chars = sorted(data["glyphs"])
    stack = np.stack([np.array(data["glyphs"][c], np.float32) / 255.0 for c in chars])
    return chars, stack, lat


def darkness(image, lattice=None):
    """-> (H, W) float32 darkness with the dot lattice zeroed, or None."""
    im = _open(image)
    if im.size != (W, H):
        return None
    a = np.asarray(im.convert("RGB")).astype(np.float32)
    d = 1.0 - a.min(axis=2) / 255.0
    if lattice is None:
        lattice = templates()[2]
    d[lattice] = 0.0
    return d


def _place(t, i):
    """Template t (H, WIN) placed at slot i on an (H, W) canvas."""
    out = np.zeros((H, W), np.float32)
    x = SLOT0 + STEP * i
    w = min(WIN, W - x)
    out[:, x:x + w] = t[:, :w]
    return out


@lru_cache(maxsize=1)
def _placed():
    chars, stack, _ = templates()
    return np.stack([np.stack([_place(t, i) for t in stack]) for i in range(LENGTH)])


def cover(d, rounds=3):
    """-> (word, error, per-slot margins) for darkness d."""
    chars, stack, _ = templates()
    P = _placed()                                   # (slot, char, H, W)
    # start: each slot alone against the image
    pick = [int(np.abs(P[i] - d).sum(axis=(1, 2)).argmin()) for i in range(LENGTH)]
    margins = [0.0] * LENGTH
    for _ in range(rounds):
        changed = False
        for i in range(LENGTH):
            rest = np.zeros((H, W), np.float32)
            for j in range(LENGTH):
                if j != i:
                    rest = np.maximum(rest, P[j, pick[j]])
            errs = np.abs(np.maximum(P[i], rest) - d).sum(axis=(1, 2))
            order = np.argsort(errs)
            margins[i] = float(errs[order[1]] - errs[order[0]])
            if order[0] != pick[i]:
                pick[i], changed = int(order[0]), True
        if not changed:
            break
    pred = np.zeros((H, W), np.float32)
    for i in range(LENGTH):
        pred = np.maximum(pred, P[i, pick[i]])
    err = float(np.abs(pred - d).sum())
    return "".join(chars[k] for k in pick), err, margins


def solve_image(image):
    """-> (text, confidence). ('', 0.0) for a wrong size or a blank image."""
    d = darkness(image)
    if d is None:
        return "", 0.0
    ink = float(d.sum())
    if ink < 50:
        return "", 0.0
    word, err, margins = cover(d)
    fidelity = max(0.0, 1.0 - err / ink)
    margin = min(margins) / ink
    return word, float(min(fidelity, min(1.0, margin / 0.02)))
