"""Goa RERA (rera.goa.gov.in/reraApp) search captcha: `captcha.jpg?x=<ns>`,
fetched after the app's base page in the same session.

200x45 JPEG. Six glyphs in black, mixed case plus digits, on flat grey (~208).
The font is Arial Italic, drawn without antialiasing. Each glyph gets its own
rotation (about +-25 degrees), a small vertical jitter and a random gap, and
glyphs rarely touch. Validation is case-sensitive.

Read: threshold the ink, take 8-connected components, and merge the pieces of
one glyph (the dots of i/j, broken strokes) by horizontal overlap. If that
leaves fewer than six, split the widest component at its thinnest column.
Each glyph is then classified by a small CNN (`ga_rera_model.pt`) that sees it
at native scale and native height on a 48x40 canvas. Scale matters: o/O, s/S,
c/C, v/V, w/W, x/X and z/Z differ only in size and in where they sit relative
to the baseline. The CNN was pretrained on Arial Italic renders of the same
composition, then fine-tuned on eye-labelled real glyphs.

`I` and `l` are the same bar in Arial at this size, so the reader reads
every bar as `l`, summing the two classes' probabilities. The portal
accepted `l` for all 4 bars it was shown: 3 live answers and 1 gold. So the
generator very likely never draws `I`, and no AMBIGUOUS entry is needed.
Measured 2026-11-03: 53/53 on held-out captures, including 10 fresh
captures whose reader answers the portal accepted.

Confidence is the product of the six per-glyph softmax maxima.

`route_200x45(pil)` tells this captcha from CERSAI's, which is the same size:
CERSAI's ink is green, GA's is neutral black on grey.
"""
import io
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .glyphcnn import GlyphCNN, components, merge, predict

W, H = 200, 45
LENGTH = 6
CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
CH, CW = 48, 40                      # glyph canvas
# I and l are the same bar here; the server accepted "l" for one (gold),
# so every bar is read as l. Pair with AMBIGUOUS["garera"] = "l".
FOLD = {"I": "l"}
INK = 128                            # L below this is ink
_MODEL = Path(__file__).resolve().with_name("ga_rera_model.pt")


def _open(image):
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, (bytes, bytearray)):
        return Image.open(io.BytesIO(image))
    return Image.open(image)


def route_200x45(pil):
    """'cersai' or 'garera' for a 200x45 image.

    Measured 2026-11-03: every CERSAI capture has at least 1053 strongly
    green pixels (G - R > 40 and G - B > 40; 300 images), and every GA
    capture has 0 (121 images). GA's ink is neutral. The cut is at 100."""
    a = np.asarray(pil.convert("RGB")).astype(np.int16)
    green = ((a[:, :, 1] - a[:, :, 0] > 40) & (a[:, :, 1] - a[:, :, 2] > 40)).sum()
    return "cersai" if green >= 100 else "garera"


def segment(image, n=LENGTH):
    """-> list of glyph components (see glyphcnn.components), left to right."""
    g = np.asarray(_open(image).convert("L")) < INK
    cs = [c for c in components(g) if c["n"] >= 3]
    # pieces of one glyph overlap horizontally (i/j dots, broken strokes)
    merged = []
    for c in cs:
        if merged:
            p = merged[-1]
            ov = min(p["x1"], c["x1"]) - max(p["x0"], c["x0"]) + 1
            if ov >= 0.5 * min(p["x1"] - p["x0"] + 1, c["x1"] - c["x0"] + 1):
                merged[-1] = merge(p, c)
                continue
        merged.append(c)
    # specks too small to be a glyph join their nearest neighbour
    while len(merged) > n:
        k = min(range(len(merged)), key=lambda i: merged[i]["n"])
        if len(merged) == 1:
            break
        j = k - 1 if k == len(merged) - 1 else (
            k + 1 if k == 0 else (k - 1 if merged[k]["x0"] - merged[k - 1]["x1"]
                                  <= merged[k + 1]["x0"] - merged[k]["x1"] else k + 1))
        a, b = sorted((k, j))
        merged[a:b + 1] = [merge(merged[a], merged[b])]
    # touching glyphs: split the widest at its thinnest middle column
    while 0 < len(merged) < n:
        k = max(range(len(merged)), key=lambda i: merged[i]["x1"] - merged[i]["x0"])
        c = merged[k]
        w = c["x1"] - c["x0"] + 1
        if w < 16:
            break
        ys, xs = c["pix"]
        cols = np.bincount(xs - c["x0"], minlength=w)
        lo, hi = int(w * 0.3), int(w * 0.7) + 1
        cut = c["x0"] + lo + int(cols[lo:hi].argmin())
        left, right = xs < cut, xs >= cut
        parts = []
        for m in (left, right):
            py, px = ys[m], xs[m]
            parts.append(dict(x0=int(px.min()), x1=int(px.max()), y0=int(py.min()),
                              y1=int(py.max()), n=int(m.sum()), pix=(py, px)))
        merged[k:k + 1] = parts
    return merged


def canvas(c):
    """Glyph at native scale and native rows, centred horizontally."""
    out = np.zeros((CH, CW), np.float32)
    ys, xs = c["pix"]
    cx = (c["x0"] + c["x1"]) // 2
    xx = xs - cx + CW // 2
    keep = (xx >= 0) & (xx < CW) & (ys < CH)
    out[ys[keep], xx[keep]] = 1.0
    return out


@lru_cache(maxsize=1)
def model():
    m = GlyphCNN(len(CHARSET), CH, CW)
    m.load_state_dict(torch.load(_MODEL, map_location="cpu"))
    m.eval()
    return m


def solve_image(image):
    """-> (text, confidence). ('', 0.0) for a wrong size or a blank image."""
    pil = _open(image)
    if pil.size != (W, H):
        return "", 0.0
    segs = segment(pil)
    if not segs or sum(c["n"] for c in segs) < 150:
        return "", 0.0
    idx, p = predict(model(), np.stack([canvas(c) for c in segs]))
    # fold I into l: sum their probabilities so the bar is not split two ways
    li, ii = CHARSET.index("l"), CHARSET.index("I")
    p[:, li] += p[:, ii]
    p[:, ii] = 0.0
    idx = p.argmax(1)
    text = "".join(CHARSET[i] for i in idx)
    conf = float(np.prod(p.max(1))) if len(segs) == LENGTH else 0.0
    return text, conf
