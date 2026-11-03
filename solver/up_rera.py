"""Uttar Pradesh RERA (up-rera.in) captcha: MSCaptcha (`CaptchaImage.axd`),
linked from `frm_Rc_Report.aspx` and fetched in the same session.

150x35 JPEG. Five green glyphs from `2346789ACDEFGHJKLNPQRTUVXYZ`. Each glyph has
its own random font (serif or bold sans), size and baseline, and some are
squeezed horizontally. Black and blue noise dots are scattered over
everything, including on top of the glyphs. The glyphs never touch. MSCaptcha
also rejects an answer submitted within about 4 s of the fetch, whether or
not it is right.

Read:
1. Take the green ink and fill the holes the noise dots punch in it.
2. Take 8-connected components, merge the pieces of one glyph by horizontal
   overlap, and fold away specks.
3. Scale each glyph to fit a 32x32 box (case does not exist here, so size
   carries no information) and classify it with a small CNN
   (`up_rera_model.pt`). The CNN was pretrained on renders of the system
   fonts MSCaptcha draws from, then fine-tuned on eye-labelled real glyphs.

Confidence is the product of the five per-glyph softmax maxima.
"""
import io
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .glyphcnn import GlyphCNN, components, merge, predict

W, H = 150, 35
LENGTH = 5
# census of 270 eye-labelled characters: 0 1 5 B I M O S W never occur
CHARSET = "2346789ACDEFGHJKLNPQRTUVXYZ"
CH = CW = 40
BOX = 32
_MODEL = Path(__file__).resolve().with_name("up_rera_model.pt")


def _open(image):
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, (bytes, bytearray)):
        return Image.open(io.BytesIO(image))
    return Image.open(image)


def ink(image):
    """-> (H, W) bool green ink with noise-dot holes filled."""
    a = np.asarray(_open(image).convert("RGB")).astype(np.int16)
    r, g, b = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    green = (g - r > 25) & (g - b > 40) & (g < 235)
    dark = a.max(2) < 120
    # a dark pixel with green on both sides (row or column) is a hole
    pad = np.pad(green, 1)
    lr = pad[1:-1, :-2] & pad[1:-1, 2:]
    ud = pad[:-2, 1:-1] & pad[2:, 1:-1]
    return green | (dark & (lr | ud))


SLOT = W // LENGTH           # MSCaptcha draws glyph i inside x in [30i, 30i+30)


def segment(image, n=LENGTH):
    """-> one merged component per occupied slot, left to right. A glyph
    broken apart by noise (a thin W, a split K) is rejoined by its slot."""
    slots = {}
    pieces = []
    for c in components(ink(image)):
        if c["n"] < 4:
            continue
        if c["x1"] - c["x0"] + 1 > 1.2 * SLOT:      # two glyphs touching
            ys, xs = c["pix"]
            for k in np.unique(xs // SLOT):
                m = xs // SLOT == k
                pieces.append(dict(x0=int(xs[m].min()), x1=int(xs[m].max()), y0=int(ys[m].min()),
                                   y1=int(ys[m].max()), n=int(m.sum()), pix=(ys[m], xs[m])))
        else:
            pieces.append(c)
    for c in pieces:
        k = min(n - 1, ((c["x0"] + c["x1"]) // 2) // SLOT)
        slots[k] = merge(slots[k], c) if k in slots else c
    return [slots[k] for k in sorted(slots) if slots[k]["n"] >= 15]


def canvas_from_mask(a):
    """bool glyph mask (any size) -> (CH, CW) float canvas, scaled to BOX."""
    ys, xs = np.nonzero(a)
    out = np.zeros((CH, CW), np.float32)
    if not len(xs):
        return out
    crop = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.uint8) * 255
    h, w = crop.shape
    s = BOX / max(h, w)
    nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
    g = np.asarray(Image.fromarray(crop).resize((nw, nh), Image.BILINEAR), np.float32) / 255.0
    y0, x0 = (CH - nh) // 2, (CW - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = g
    return out


def canvas(c):
    ys, xs = c["pix"]
    a = np.zeros((c["y1"] - c["y0"] + 1, c["x1"] - c["x0"] + 1), bool)
    a[ys - c["y0"], xs - c["x0"]] = True
    return canvas_from_mask(a)


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
    if not segs:
        return "", 0.0
    idx, p = predict(model(), np.stack([canvas(c) for c in segs]))
    text = "".join(CHARSET[i] for i in idx)
    conf = float(np.prod(p.max(1))) if len(segs) == LENGTH else 0.0
    return text, conf
