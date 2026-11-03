"""Khajane-II (k2.karnataka.gov.in) captcha: the treasury page every Kaveri
payment passes through, and K2's public challan pages
(`/wps/PA_*/CustomSimpleCaptchaServlet`).

200x50 RGBA PNG from Java SimpleCaptcha (`nl.captcha`):

* text: 5 characters from SimpleCaptcha's default set `abcdefghkmnprwxy2345678`
* DefaultWordRenderer: black DejaVu Sans 48 on a transparent background, pen
  starting at x = 10 on baseline y = 37, advancing by int(ink width), so glyphs
  touch and overlap
* CurvedLineNoiseProducer: one black curve, about 3 px, from x ~ 20 to ~ 180

The alpha channel is the ink, and overlaps compose as transmittance products.
Pen positions follow from the characters, so reading is a left-to-right beam
search over glyph renders (`k2_glyphs.json`), no model. The curve only adds
ink, so extra ink costs less than missing ink.

Measured 2026-10-28: 40/40 exact on public-page samples and 1/1 on a payment
page capture. Pass native pixels: a screenshot shifted by 1 px misread one
character.
"""
import io
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

W, H = 200, 50
LENGTH = 5
X0 = 10
BEAM = 40
W_EXTRA = 0.35          # weight of ink the glyphs don't explain (the curve)
_HERE = Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def templates():
    data = json.loads((_HERE / "k2_glyphs.json").read_text())
    return data["_meta"]["charset"], {
        c: (g["left"], np.array(g["alpha"], np.float32), g["adv"]) for c, g in data["glyphs"].items()}


def ink(image):
    """-> (H, W) float ink coverage in [0, 1]."""
    if isinstance(image, np.ndarray):
        a = image.astype(np.float32)
        return a / 255.0 if a.max() > 1.0 else a
    im = image if isinstance(image, Image.Image) else Image.open(
        io.BytesIO(image) if isinstance(image, (bytes, bytearray)) else image)
    if im.size != (W, H):
        raise ValueError("k2 captcha must be %dx%d, got %r" % (W, H, im.size))
    if im.mode in ("RGBA", "LA") and np.asarray(im.convert("RGBA"))[:, :, 3].min() < 255:
        return np.asarray(im.convert("RGBA"))[:, :, 3].astype(np.float32) / 255.0
    return 1.0 - np.asarray(im.convert("L")).astype(np.float32) / 255.0


def cover(A, beam=BEAM):
    """-> [(error, word)] best first."""
    charset, tp = templates()
    t_obs = 1.0 - A
    lead = min(t[0] for t in tp.values())
    states = [(0.0, "", X0, np.ones((H, W), np.float32), 0)]
    for k in range(LENGTH):
        last = k == LENGTH - 1
        cand = []
        for err, word, x, tpred, done in states:
            for c in charset:
                left, g, adv = tp[c]
                lim = W if last else max(done, min(W, x + adv + lead))
                win = tpred[:, done:lim].copy()
                x0 = x + left
                ca, cb = max(done, x0), min(lim, x0 + g.shape[1])
                if cb > ca:
                    win[:, ca - done:cb - done] *= 1 - g[:, ca - x0:cb - x0]
                d = win - t_obs[:, done:lim]
                e = float(np.where(d > 0, d * W_EXTRA, -d).sum())
                cand.append((err + e, word + c, x, x + adv, tpred, lim))
        cand.sort(key=lambda s: s[0])
        states = []
        for err, word, xp, xn, tpred, lim in cand[:beam]:
            t2 = tpred.copy()
            left, g, _ = tp[word[-1]]
            x0 = xp + left
            a, b = max(0, x0), min(W, x0 + g.shape[1])
            t2[:, a:b] *= 1 - g[:, a - x0:b - x0]
            states.append((err, word, xn, t2, lim))
    best = {}
    for err, word, *_ in states:
        best.setdefault(word, err)
    return sorted((e, w) for w, e in best.items())


def solve_image(image):
    """-> (text, confidence). Confidence 0 when the image doesn't look like this captcha."""
    A = ink(image)
    total = float(A.sum())
    if total < 200:
        return "", 0.0
    found = cover(A)
    err, word = found[0]
    fidelity = 1.0 - err / total
    margin = (found[1][0] - err) / total if len(found) > 1 else 1.0
    if fidelity < 0.6:
        return word, 0.0
    return word, float(min(1.0, margin / 0.002))
