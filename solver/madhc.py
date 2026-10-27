"""Madras High Court captcha (hcmadras.tn.gov.in securimage_show.php), 120x60.

Securimage (PHP) configured with six digits, no angles, no random baseline,
no perturbation, no noise lines. Everything that varies is the start x and
the -2..0 px between characters, so the image is a deterministic sum of known
parts and the read is an exact cover, not a classification. No model.

Measured composition:

* font: securimage's default AHGBold at 24pt (GD renders at 96 dpi -> 32 px),
  baseline at y=41, every digit advance 18 px (tabular figures). FreeType via
  Pillow reproduces GD's glyphs to ~0.4% residual -- except `1`, whose stem GD
  hints a pixel differently, which is why the shipped templates are refined on
  the train split rather than taken straight from the font.
* ink is #323232 at 80% opacity (securimage's default 20% transparency):
  one layer reads 91, two overlapping layers ~58. Opacity multiplies, so in
  `L = -log((I - 50) / 205)` overlapping glyphs simply ADD. That is what lets
  overlapped digits be decomposed instead of guessed.

Confidence is the cover's fidelity (how much of the ink the six templates
explain), capped by the margin to the best *different* digit string. Both
are measurements, not a softmax.
"""
import io
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

W, H = 120, 60
LENGTH = 6
CHARSET = "0123456789"
INK, OPACITY = 50.0, 0.8
L_MAX = 7.0                      # -log of the clip floor; a fully dark pixel
BASELINE = 41
FIRST_X = range(0, 50)           # securimage: x = rand(5, 2*cx - 5)
# Securimage advances by each glyph's INK box (imagettfbbox) plus a random
# -2..0, not by the font's 18 px figure advance, so a `1` steps less than an
# `8`. This wide range is only for DISCOVERING the steps (mkglyphs_madhc.py);
# the reader uses the per-digit step sets measured on the train split and
# stored with the templates (13/15/17 px; 11/13/15 after a `1`). A random
# space (which securimage can insert) was never observed, and would show up
# as a low-fidelity read rather than a confident wrong one.
STEP = range(4, 34)
BEAM = 40
_HERE = Path(__file__).resolve().parent


def open_image(image):
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, (bytes, bytearray)):
        return Image.open(io.BytesIO(image))
    return Image.open(image)


def observed(image):
    """-> L map (H, W): additive darkness of the 80%-opaque ink."""
    im = open_image(image).convert("L")
    if im.size != (W, H):
        raise ValueError("madhc captcha must be %dx%d, got %r" % (W, H, im.size))
    I = np.asarray(im, np.float32)
    T = np.clip((I - INK) / (255.0 - INK), np.exp(-L_MAX), 1.0)
    return -np.log(T)


@lru_cache(maxsize=1)
def _library():
    data = json.loads((_HERE / "madhc_glyphs.json").read_text())
    if data.get("_meta", {}).get("built_from") != "train-split":
        raise ValueError("madhc templates must be built from the train split")
    return data


def templates():
    """digit -> (top_row, left_offset, L-template). Offsets are relative to
    (pen x, baseline)."""
    return {d: (g["top"], g["left"], np.array(g["L"], np.float32))
            for d, g in _library()["glyphs"].items()}


def steps():
    """digit -> allowed pen steps to the NEXT glyph."""
    return {d: tuple(g["steps"]) for d, g in _library()["glyphs"].items()}


def _cache_clear():
    _library.cache_clear()


def _add(R, tpl, x):
    """Add a template at pen position x into R (in place). -> (x0, x1) cols."""
    top, left, t = tpl
    h, w = t.shape
    x0 = x + left
    a, b = max(0, x0), min(W, x0 + w)
    if b > a:
        R[top:top + h, a:b] += t[:, a - x0:b - x0]
    return a, b


def _err(R, L, a, b):
    return float(np.abs(np.minimum(R[:, a:b], L_MAX) - L[:, a:b]).sum())


def cover(L, tp=None, label=None, beam=BEAM, step=None):
    """Beam search over (digit, pen x) left to right.

    After placing glyph k at x, no later glyph can reach columns < x + (the
    smallest step), so
    those columns are final and are scored then; the rest of the image is
    scored when the sixth glyph lands. Returns a list of
    (error, word, positions) for the best few DISTINCT words, best first.
    """
    tp = tp or templates()
    step = step or steps()
    lead = min(min(v) for v in step.values())
    states = [(0.0, "", (), np.zeros_like(L), 0)]       # err, word, xs, R, done
    for k in range(LENGTH):
        last = k == LENGTH - 1
        cand = []
        for err, word, xs, R, done in states:
            starts = FIRST_X if not xs else [xs[-1] + s for s in step[word[-1]]]
            for d in (label[k] if label else CHARSET):
                t = tp[d]
                for x in starts:
                    lim = W if last else min(W, x + lead)
                    if lim < done:
                        continue
                    # score only the newly finalised columns [done, lim)
                    top, left, g = t
                    a, b = max(done, x + left), min(lim, x + left + g.shape[1])
                    if b > a:
                        win = R[:, done:lim].copy()
                        win[top:top + g.shape[0], a - done:b - done] += \
                            g[:, a - x - left:b - x - left]
                        e = float(np.abs(np.minimum(win, L_MAX) - L[:, done:lim]).sum())
                    else:
                        e = _err(R, L, done, lim)
                    cand.append((err + e, word + d, xs + (x,), R, lim))
        cand.sort(key=lambda c: c[0])
        states = []
        for err, word, xs, R, lim in cand[:beam * 3]:
            R2 = R.copy()
            _add(R2, tp[word[-1]], xs[-1])
            states.append((err, word, xs, R2, lim))
            if len(states) == beam:
                break
    best = {}
    for err, word, xs, R, _ in states:
        if word not in best or err < best[word][0]:
            best[word] = (err, word, xs)
    return sorted(best.values())


def solve_image(image):
    """-> (text, confidence)."""
    L = observed(image)
    ink = float(L.sum())
    if ink < 50:                       # blank: six digits carry far more ink
        return "", 0.0
    found = cover(L)
    err, word, _ = found[0]
    fidelity = max(0.0, 1.0 - err / ink)
    # Measured on 258 real images: fidelity >= 0.997 and the best DIFFERENT
    # digit string costs >= 4% of the ink more. Confidence is the fidelity,
    # pulled down only when a rival string comes within 2% of the ink.
    margin = (found[1][0] - err) / ink if len(found) > 1 else 1.0
    return word, float(min(fidelity, margin / 0.02, 1.0))
