"""Gujarat High Court captcha (gujarathc-casestatus.nic.in CaptchaServlet?ct=S).

200x50 RGB PNG from Java SimpleCaptcha (`nl.captcha`, the library behind the
GST reader too), built as:

    FlatColorBackgroundProducer(#dbf7ff)
    DefaultWordRenderer(colours = black/blue/red, font = DejaVu Sans Mono 40)
    StraightLineNoiseProducer(black)          -- drawn OVER the text
    addBorder()                               -- 1 px black

DefaultWordRenderer draws each character upright at pen x, starting at
round(0.05 * W) = 10 on baseline H - round(0.25 * H) = 37, then advances by
int(glyph visual-bounds width) -- the INK width, so glyphs touch and overlap.
Every pen position therefore follows from the characters themselves: the
only unknowns are the five characters and their colours.

The read needs no colour: all three ink colours have green == 0, so in the
green channel every glyph is a pure transmittance mask and overlaps compose
by multiplication, whichever glyph is on top:

    L = -log(G / 247) = sum_k -log(1 - alpha_k)

which makes the image an additive cover of known glyphs (as for Madras). The
noise line is an aliased 4 px polygon from x=0 to x=W; it is fitted from the
text-free margins and its pixels are masked out of the comparison. No model.

Confidence is cover fidelity (fraction of the unmasked ink the five glyphs
explain), capped by the margin to the best different word.
"""
import io
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

W, H = 200, 50
LENGTH = 5
#  never occurs: 0 of 400 hand-labelled characters against ~11 expected
# from a uniform 36 (P = 1e-5), so this deployment's word list excludes it.
CHARSET = "0123456789abcdefghijklmnopqrstuvwxy"
BG_G = 247.0
L_MAX = float(-np.log(0.5 / BG_G))      # an interior (G == 0) pixel
X0 = 10                                  # round(0.05 * 200)
BASELINE = 37                            # 50 - round(0.25 * 50)
TEXT_X = (10, 112)                       # measured: ink never leaves 10..110
BEAM = 30
_HERE = Path(__file__).resolve().parent


def open_image(image):
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, (bytes, bytearray)):
        return Image.open(io.BytesIO(image))
    return Image.open(image)


def line_mask(rgb):
    """Boolean (H, W): pixels the noise line (or the border) may occupy.

    The line is the only pure-black thing outside the text columns, so its
    top and bottom edges are fitted on the margins and interpolated across
    the text, padded a pixel each side for the polygon's aliased steps."""
    blk = (rgb == 0).all(2)
    xs, tops, bots = [], [], []
    for x in list(range(1, TEXT_X[0] - 1)) + list(range(TEXT_X[1] + 4, W - 1)):
        ys = np.where(blk[1:H - 1, x])[0] + 1
        if len(ys):
            xs.append(x)
            tops.append(ys.min())
            bots.append(ys.max())
    m = np.zeros((H, W), bool)
    m[0, :] = m[-1, :] = m[:, 0] = m[:, -1] = True
    if len(xs) >= 4:
        xs = np.array(xs, float)
        pt = np.polyfit(xs, tops, 1)
        pb = np.polyfit(xs, bots, 1)
        cols = np.arange(W)
        top = np.floor(np.polyval(pt, cols)) - 1
        bot = np.ceil(np.polyval(pb, cols)) + 1
        yy = np.arange(H)[:, None]
        m |= (yy >= top[None]) & (yy <= bot[None])
    return m


def observed(image):
    """-> (L, valid): additive green-channel darkness, and the pixels that
    are not under the line/border."""
    im = open_image(image).convert("RGB")
    if im.size != (W, H):
        raise ValueError("gujhc captcha must be %dx%d, got %r" % (W, H, im.size))
    rgb = np.asarray(im)
    G = rgb[:, :, 1].astype(np.float32)
    L = -np.log(np.clip(G, 0.5, BG_G) / BG_G)
    return L, ~line_mask(rgb)


@lru_cache(maxsize=1)
def _library():
    data = json.loads((_HERE / "gujhc_glyphs.json").read_text())
    if data.get("_meta", {}).get("built_from") != "train-split":
        raise ValueError("gujhc templates must be built from the train split")
    return data


def templates():
    """char -> (top, left, L-template, advance); offsets rel. (pen x, baseline)."""
    return {c: (g["top"], g["left"], np.array(g["L"], np.float32), g["adv"])
            for c, g in _library()["glyphs"].items()}


def _add(R, tpl, x):
    top, left, t, _ = tpl
    h, w = t.shape
    x0 = x + left
    a, b = max(0, x0), min(W, x0 + w)
    if b > a:
        R[top:top + h, a:b] += t[:, a - x0:b - x0]


def cover(L, valid, tp=None, label=None, beam=BEAM, adv=None):
    """Beam search over characters; pen x follows from them.

    `adv` optionally maps char -> candidate advances (used only while
    discovering advances on the train split); normally each char has one.
    After glyph k, no later glyph can reach columns left of the next pen x
    plus the smallest left bearing, so those columns are final and scored.
    Returns [(error, word, pens)] for the best distinct words, best first."""
    tp = tp or templates()
    adv = adv or {c: (t[3],) for c, t in tp.items()}
    lead = min(t[1] for t in tp.values())           # smallest left offset
    # Compare in transmittance (G / 247), not in L: an opaque interior pixel
    # clips L at L_MAX, and an L-domain error there would outweigh a whole
    # edge. In T every pixel's error lies in [0, 1].
    Tc = np.where(valid, np.exp(-L), 1.0)
    states = [(0.0, "", (X0,), np.zeros_like(L), 0)]
    for k in range(LENGTH):
        last = k == LENGTH - 1
        cand = []
        for err, word, pens, R, done in states:
            x = pens[-1]
            for c in (label[k] if label else CHARSET):
                top, left, g, _ = tp[c]
                for a in adv[c]:
                    lim = W if last else max(done, min(W, x + a + lead))
                    win = R[:, done:lim].copy()
                    x0 = x + left
                    ca, cb = max(done, x0), min(lim, x0 + g.shape[1])
                    if cb > ca:
                        win[top:top + g.shape[0], ca - done:cb - done] += \
                            g[:, ca - x0:cb - x0]
                    d = np.abs(np.exp(-np.minimum(win, L_MAX)) - Tc[:, done:lim])
                    e = float((d * valid[:, done:lim]).sum())
                    cand.append((err + e, word + c, pens + (x + a,), R, lim))
        cand.sort(key=lambda s: s[0])
        states = []
        for err, word, pens, R, lim in cand[:beam]:
            R2 = R.copy()
            _add(R2, tp[word[-1]], pens[-2])
            states.append((err, word, pens, R2, lim))
    best = {}
    for err, word, pens, R, _ in states:
        if word not in best or err < best[word][0]:
            best[word] = (err, word, pens[:-1])
    return sorted(best.values())


def solve_image(image):
    """-> (text, confidence)."""
    L, valid = observed(image)
    ink = float(((1 - np.exp(-L)) * valid).sum())
    if ink < 50:
        return "", 0.0
    found = cover(L, valid)
    err, word, _ = found[0]
    fidelity = max(0.0, 1.0 - err / ink)
    margin = (found[1][0] - err) / ink if len(found) > 1 else 1.0
    return word, float(min(fidelity, margin / 0.02, 1.0))
