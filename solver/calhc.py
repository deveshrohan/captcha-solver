"""Calcutta High Court captcha (calcuttahighcourt.gov.in/captcha/default), 120x36.

Laravel `mews/captcha` (the URL is `captcha_src('default')`'s shape: the
config name plus 8 random characters to defeat caching), an older release
whose default character list is `2346789abcdefghjmnpqrtuxyz` + capitals.
The site keeps `sensitive` off, so mews LOWERCASES the text before drawing it
and lowercases the submission before checking: 26 classes, no case.

Measured on real images (and matching the PHP source):

* 6 characters; none of `0 1 5 i k l o s v w` occurs in 80 hand labels,
  exactly mews' exclusions, so the homoglyph pairs never arise.
* each character: random font from mews' six bundled faces, random size
  `rand(H-10, H)` = 26..36 px, random angle `rand(-15, 15)`, at
  `(4 + k*(W-4)/6, H/6)` with the upper-left corner of GD's `imagettfbbox`
  (taken at angle 0, then rotated by hand) pinned there -- Intervention v2's
  `align left, valign top`.
* that box's top is NOT the glyph's own ink top: x-height letters sit 5 px
  lower than ascenders and every class shares a baseline (real ink bottoms
  26.8 vs 26.6 px). The top matches each font's cap height, so glyphs hang
  from the cap line; with em = size this reproduces the measured per-class
  tops, bottoms and heights to within ~0.5 px.
* background flat #c4c4c4 (no background image), ink pure blue, NO noise
  lines, libgd JPEG quality 75 with 4:2:0 chroma subsampling.

Per-character font, size and rotation are random, so no template library can
cover it; the reader is a CRNN+CTC trained on a port of that renderer. The
input is the red channel inverted: blue ink has red ~0 against 196.
"""
import io
import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont

CHARSET = "2346789abcdefghjmnpqrtuxyz"
CH2I = {c: i for i, c in enumerate(CHARSET)}
I2CH = {i: c for c, i in CH2I.items()}
N_CLASSES = len(CHARSET)
BLANK = N_CLASSES
LENGTH = 6
W, H = 120, 36
IN_W, IN_H = 192, 64
BG = 196
LEFT_PAD = 4
ANGLE = 15

_HERE = os.path.dirname(os.path.abspath(__file__))
_FONTS_DIR = os.path.join(os.path.dirname(_HERE), "fonts", "mews")
FONT_FILES = [os.path.join(_FONTS_DIR, f) for f in (
    "ABeeZee_regular.ttf", "Asap_700.ttf", "Khand_500.ttf",
    "Open_Sans_regular.ttf", "Roboto_regular.ttf", "Ubuntu_regular.ttf")]


def open_image(image):
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, (bytes, bytearray)):
        return Image.open(io.BytesIO(image))
    return Image.open(image)


def _to_input(rgb):
    """RGB uint8 (H, W, 3) -> uint8 (IN_H, IN_W) ink map (0 = background)."""
    r = rgb[:, :, 0].astype(np.float32)
    ink = np.clip((BG - r) / BG, 0, 1) * 255
    im = Image.fromarray(ink.astype(np.uint8)).resize((IN_W, IN_H), Image.BILINEAR)
    return np.asarray(im, np.uint8)


def load_real(image):
    im = open_image(image).convert("RGB")
    if im.size != (W, H):
        raise ValueError("calhc captcha must be %dx%d, got %r" % (W, H, im.size))
    return _to_input(np.asarray(im))


# --------------------------------------------------------------------------
# synthetic renderer: a port of mews/captcha text() over Intervention v2 GD
# --------------------------------------------------------------------------
_font_cache = {}


def _font(path, size):
    key = (path, size)
    if key not in _font_cache:
        _font_cache[key] = ImageFont.truetype(path, size)
    return _font_cache[key]


def random_label(rng):
    return "".join(CHARSET[i] for i in rng.integers(0, N_CLASSES, LENGTH))


def encode_label(label):
    return [CH2I[c] for c in label]


SS = 3   # supersampling: GD rasterises rotated outlines; we rotate a fine mask

# OS/2 sCapHeight / unitsPerEm of each bundled face (read with fontTools).
CAP_HEIGHT = {"ABeeZee_regular.ttf": 0.700, "Asap_700.ttf": 0.700,
              "Khand_500.ttf": 0.678, "Open_Sans_regular.ttf": 0.714,
              "Roboto_regular.ttf": 0.711, "Ubuntu_regular.ttf": 0.693}
# Residual offset between GD's and this renderer's placement, measured on the
# labelled train+validation reals (synthetic ink sat ~0.5 px low).
RAISE = 0.6
# Per-glyph placement jitter (px) covering what the port does not model.
JITTER = 0.75


def _glyph(ch, path, size, angle):
    """-> (alpha float32 (h, w) at 1x, ox, oy): rotated glyph mask whose
    pixel (0, 0) sits at (ox, oy) relative to the rotated box's upper-left
    corner -- the point Intervention pins to (marginLeft, marginTop)."""
    # Intervention v2 asks GD for ceil(size * 0.75) pt, which at 96 dpi would
    # be a touch over `size` px, but em = size matched the real glyph heights
    # better (22.1/17.2/22.9 px vs real 22.1/16.9/22.5 by class).
    em = size
    f = _font(path, int(round(em * SS)))
    l, t, r, b = f.getbbox(ch, anchor="ls")          # ink box rel. pen/baseline
    box_top = -(CAP_HEIGHT[os.path.basename(path)] * em + RAISE) * SS
    pad = int(em * SS)
    cw, chh = (r - l) + 2 * pad, (b - t) + 2 * pad
    canvas = Image.new("L", (cw, chh), 0)
    # pen at (pad - l, pad - t): the ink box's upper-left lands at (pad, pad)
    px, py = pad - l, pad - t
    ImageDraw.Draw(canvas).text((px, py), ch, font=f, fill=255, anchor="ls")
    # GD angles are counter-clockwise about the pen origin, like PIL's rotate
    rot = canvas.rotate(angle, resample=Image.BICUBIC, center=(px, py))
    # rotated position of every corner of the unrotated ink box (GD's bbox)
    a = np.deg2rad(angle)
    # imagettfbbox's points 6,7 = the box's upper-left, rotated by hand
    ulx = l * np.cos(a) + box_top * np.sin(a)
    uly = -l * np.sin(a) + box_top * np.cos(a)
    small = rot.resize((cw // SS, chh // SS), Image.BOX)
    alpha = np.asarray(small, np.float32) / 255
    # canvas origin relative to the rotated upper-left corner, at 1x
    ox = (-px - ulx) / SS
    oy = (-py - uly) / SS
    return alpha, ox, oy


def make_rgb(label, rng, jitter=JITTER):
    canvas = np.empty((H, W, 3), np.float32)
    canvas[:] = BG
    blue = float(rng.uniform(225, 255))
    colour = np.array([0, 0, blue], np.float32)
    top = H / LENGTH                                   # marginTop = H / length
    for k, ch in enumerate(label):
        left = LEFT_PAD + k * (W - LEFT_PAD) / LENGTH
        path = FONT_FILES[rng.integers(len(FONT_FILES))]
        size = int(rng.integers(H - 10, H + 1))
        angle = int(rng.integers(-ANGLE, ANGLE + 1))
        alpha, ox, oy = _glyph(ch, path, size, angle)
        # PHP casts to int on the way into imagettftext
        jx, jy = rng.uniform(-jitter, jitter, 2) if jitter else (0.0, 0.0)
        x0 = int(round(int(left) + ox + jx))
        y0 = int(round(int(top) + oy + jy))
        h, w = alpha.shape
        ya, yb = max(0, y0), min(H, y0 + h)
        xa, xb = max(0, x0), min(W, x0 + w)
        if yb <= ya or xb <= xa:
            continue
        a = alpha[ya - y0:yb - y0, xa - x0:xb - x0, None]
        canvas[ya:yb, xa:xb] = canvas[ya:yb, xa:xb] * (1 - a) + colour * a
    im = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=75, subsampling=2)   # libgd defaults
    return np.asarray(Image.open(buf).convert("RGB"))


def make_input(label, rng):
    return _to_input(make_rgb(label, rng))


# --------------------------------------------------------------------------
# model: the repo's single-channel CRNN+CTC (same body as ITAT's)
# --------------------------------------------------------------------------
class CalhcCRNN(nn.Module):
    def __init__(self, n_classes=N_CLASSES):
        super().__init__()

        def cb(cin, cout, pool):
            return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1),
                                 nn.BatchNorm2d(cout), nn.ReLU(), nn.MaxPool2d(pool))

        self.cnn = nn.Sequential(
            cb(1, 64, (2, 2)),      # 32 x 96
            cb(64, 128, (2, 2)),    # 16 x 48
            cb(128, 256, (2, 2)),   # 8 x 24
            cb(256, 256, (2, 1)),   # 4 x 24
            cb(256, 384, (2, 1)),   # 2 x 24
            cb(384, 384, (2, 1)),   # 1 x 24
        )
        self.rnn = nn.LSTM(384, 256, num_layers=2, bidirectional=True,
                           batch_first=True, dropout=0.2)
        self.head = nn.Linear(512, n_classes + 1)

    def forward(self, x):
        f = self.cnn(x)
        f = f.squeeze(2).permute(0, 2, 1).contiguous()
        f, _ = self.rnn(f)
        return self.head(f)


def _to_tensor(imgs, device):
    x = np.asarray(imgs, dtype=np.float32) / 255.0
    return torch.tensor(x, device=device)[:, None]


def ctc_greedy_decode(logits_btc):
    probs = torch.softmax(logits_btc, dim=2)
    conf, idx = probs.max(dim=2)
    idx, conf = idx.cpu().numpy(), conf.cpu().numpy()
    out, confs = [], []
    for b in range(idx.shape[0]):
        chars, cs, prev = [], [], -1
        for t in range(idx.shape[1]):
            c = int(idx[b, t])
            if c != prev and c != BLANK:
                chars.append(I2CH[c])
                cs.append(float(conf[b, t]))
            prev = c
        out.append("".join(chars))
        confs.append(min(cs) if cs else 0.0)
    return out, confs


def predict_greedy(model, imgs, device="cpu"):
    model.eval()
    with torch.no_grad():
        return ctc_greedy_decode(model(_to_tensor(imgs, device)))


def predict(model, imgs, device="cpu", beam_width=20):
    """Length-6 CTC beam decode. -> (texts, confidences); confidence is the
    geometric-mean per-character probability of the chosen string."""
    model.eval()
    with torch.no_grad():
        lps = model(_to_tensor(imgs, device)).log_softmax(2)
    out, confs = [], []
    for b in range(lps.shape[0]):
        s, c = _beam_decode(lps[b], beam_width)
        out.append(s)
        confs.append(c)
    return out, confs


def _beam_decode(log_probs_tc, beam_width=20, force_len=LENGTH):
    from collections import defaultdict
    p = np.exp(log_probs_tc.detach().cpu().numpy())
    T, C = p.shape
    beams = {(): [1.0, 0.0]}
    for t in range(T):
        pt = p[t]
        nxt = defaultdict(lambda: [0.0, 0.0])
        for prefix, (pb, pnb) in beams.items():
            ptot = pb + pnb
            nxt[prefix][0] += ptot * pt[BLANK]
            if prefix:
                nxt[prefix][1] += pnb * pt[prefix[-1]]
            for c in range(C):
                if c == BLANK:
                    continue
                if prefix and c == prefix[-1]:
                    nxt[prefix + (c,)][1] += pb * pt[c]
                else:
                    nxt[prefix + (c,)][1] += ptot * pt[c]
        beams = dict(sorted(nxt.items(), key=lambda kv: kv[1][0] + kv[1][1],
                            reverse=True)[:beam_width])
        s = sum(v[0] + v[1] for v in beams.values())
        if s > 0:
            for k in beams:
                beams[k][0] /= s
                beams[k][1] /= s
    scored = [(pref, pb + pnb) for pref, (pb, pnb) in beams.items() if pref]
    if not scored:
        return "", 0.0
    cand = [(pref, sc) for pref, sc in scored if len(pref) == force_len]
    if cand:
        scored = cand
    pref, sc = max(scored, key=lambda x: x[1])
    return "".join(I2CH[c] for c in pref), float(sc ** (1.0 / max(len(pref), 1)))
