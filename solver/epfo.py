"""EPFO employer-portal captcha support (unifiedportal-emp.epfindia.gov.in):
150x50 RGB, 5 uppercase-alphanumeric characters.

Composition (measured, not assumed):

* the image is entirely grayscale, and the background is a **fixed vertical
  gradient that is byte-identical in every captcha** — 255 at y=0 falling to 128
  at y=25 and rising back to 249 at y=49, constant along each row. Verified by
  taking the per-row modal value across many samples: zero variation.
* so ink extraction is exact arithmetic, not a heuristic: any pixel that differs
  from `BACKGROUND[y]` is ink. Roughly 330 ink pixels per image.
* the text is upright black with no warp, no rotation and **no noise lines at
  all** — the only variation is per-character horizontal spacing, vertical
  offset and size.

That makes this the cleanest of the supported captchas: preprocessing is
subtraction, and the generator only has to model glyph placement.
"""
import io
import json
import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# 32 characters. The portal excludes 0, I, N and O: across 280 hand-labelled
# characters those four never appear, and the remaining 32 are uniformly
# distributed (chi2 p = 0.52). P(three given characters absent by chance from a
# 36-char set) is 5e-11, so this is by design, not sampling luck. Excluding
# 0/O and 1/I removes the homoglyph problem that caps the MCA reader.
CHARSET = "123456789ABCDEFGHJKLMPQRSTUVWXYZ"
CH2I = {c: i for i, c in enumerate(CHARSET)}
I2CH = {i: c for i, c in enumerate(CHARSET)}
N_CLASSES = len(CHARSET)
BLANK = N_CLASSES
LENGTH = 5

W, H = 150, 50
IN_W, IN_H = 160, 64

# The exact background, measured off real captchas (identical in all of them).
BACKGROUND = np.array(
    [255, 250, 245, 240, 235, 229, 224, 219, 214, 209, 204, 199, 194, 189, 184,
     179, 174, 168, 163, 158, 153, 148, 143, 138, 133, 128, 132, 137, 142, 147,
     153, 158, 163, 168, 173, 178, 183, 188, 193, 198, 203, 208, 214, 219, 224,
     229, 234, 239, 244, 249], dtype=np.int16)
assert BACKGROUND.shape[0] == H

_HERE = os.path.dirname(os.path.abspath(__file__))
_FONTS_DIR = os.path.join(os.path.dirname(_HERE), "fonts")
# Regular weight only: measured stroke density on real glyphs is 4.45 px per row
# of glyph height, versus 5.72 for a bold-inclusive pool — the real text is
# uniformly light. Size is effectively FIXED, not jittered: real glyph height is
# 12.09 +/- 0.41 px (that spread is just per-character overshoot, not size
# variation), and DejaVuSans at 16pt gives 12.14 x 8.36 against a measured
# 12.09 x 8.12.
_FACES = ["DejaVuSansCondensed.ttf", "DejaVuSans.ttf"]
FONT_SIZE = 16
FONT_FILES = [os.path.join(_FONTS_DIR, f) for f in _FACES
              if os.path.exists(os.path.join(_FONTS_DIR, f))]

_font_cache = {}


def _font(path, size):
    key = (path, int(size))
    if key not in _font_cache:
        _font_cache[key] = ImageFont.truetype(path, int(size))
    return _font_cache[key]


_inkw_cache = {}


def _ink_width(path, size, ch):
    key = (path, int(size), ch)
    if key not in _inkw_cache:
        img = Image.new("L", (90, 90), 255)
        ImageDraw.Draw(img).text((25, 65), ch, font=_font(path, size), fill=0, anchor="ls")
        m = np.asarray(img) < 190
        xs = np.where(m.any(0))[0]
        _inkw_cache[key] = int(xs.max() - xs.min() + 1) if len(xs) else int(size * 0.5)
    return _inkw_cache[key]


def random_label(rng):
    return "".join(CHARSET[i] for i in rng.integers(0, N_CLASSES, size=LENGTH))


def encode_label(label):
    return [CH2I[c] for c in label]


# --------------------------------------------------------------------------
# preprocessing: exact, because the background is a known constant
# --------------------------------------------------------------------------
def ink_strength(rgb):
    """(H,W,3)|(H,W) -> float32 in [0,1]: how much darker than the background."""
    a = np.asarray(rgb)
    if a.ndim == 3:
        a = a[..., 0]
    diff = BACKGROUND[:, None].astype(np.int16) - a.astype(np.int16)
    return np.clip(diff, 0, None).astype(np.float32) / 255.0


def _to_input(strength):
    im = Image.fromarray((np.clip(strength, 0, 1) * 255).astype(np.uint8))
    return np.asarray(im.resize((IN_W, IN_H), Image.BILINEAR), dtype=np.uint8)


def load_real(path_or_bytes):
    if isinstance(path_or_bytes, (bytes, bytearray)):
        im = Image.open(io.BytesIO(path_or_bytes))
    elif isinstance(path_or_bytes, Image.Image):
        im = path_or_bytes
    else:
        im = Image.open(path_or_bytes)
    return _to_input(ink_strength(im.convert("RGB")))


# --------------------------------------------------------------------------
# synthetic generation — the renderer uses a SPRITE FONT, so we blit its glyphs
# --------------------------------------------------------------------------
# The captcha is not drawn with TrueType text: every instance of a character is
# a byte-identical bitmap. Segmenting 280 labelled glyphs and normalising to
# alpha (`pixel = bg*(1-alpha)`) gives a max deviation of 0.004 across all 32
# characters — they are fixed sprites blitted at varying x/y. Widths are
# quantised accordingly (most glyphs 8px, W and Y 10, R and K 9, Q 14 tall with
# its tail), which no scalable font reproduces.
#
# So the generator does not approximate the font: it replays the extracted
# sprites in solver/epfo_glyphs.json, which makes the synthetic images
# essentially exact rather than merely close.
_GLYPHS = None
_LAYOUT = None


def _glyphs():
    global _GLYPHS, _LAYOUT
    if _GLYPHS is None:
        with open(os.path.join(_HERE, "epfo_glyphs.json")) as f:
            _GLYPHS = {k: np.asarray(v, dtype=np.float32) for k, v in json.load(f).items()}
        with open(os.path.join(_HERE, "epfo_layout.json")) as f:
            _LAYOUT = json.load(f)
    return _GLYPHS, _LAYOUT


def make_image(label, rng, jitter=True):
    """-> (H, W) uint8. Blits the real sprites over the real background."""
    G, LO = _glyphs()
    alpha = np.zeros((H, W), np.float32)
    x = LO["x0_mean"] + (rng.normal(0, LO["x0_std"]) if jitter else 0.0)
    for ch in label:
        g = G.get(ch)
        if g is None:
            continue
        gh, gw = g.shape
        top = LO["top_mean"] + (rng.normal(0, LO["top_std"]) if jitter else 0.0)
        xi, yi = int(round(x)), int(round(top))
        xi = max(0, min(W - gw, xi))
        yi = max(0, min(H - gh, yi))
        np.maximum(alpha[yi:yi + gh, xi:xi + gw], g, out=alpha[yi:yi + gh, xi:xi + gw])
        gap = rng.normal(LO["gap_mean"], LO["gap_std"]) if jitter else LO["gap_mean"]
        x += gw + max(LO["gap_min"], min(LO["gap_max"], gap))
    bg = np.repeat(BACKGROUND[:, None], W, axis=1).astype(np.float32)
    return np.rint(bg * (1.0 - alpha)).astype(np.uint8)


def make_input(label, rng):
    return _to_input(ink_strength(make_image(label, rng)))


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
class EpfoCRNN(nn.Module):
    """CRNN+CTC over the background-subtracted ink map."""

    def __init__(self, n_classes=N_CLASSES):
        super().__init__()

        def cb(cin, cout, pool):
            return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1),
                                 nn.BatchNorm2d(cout), nn.ReLU(), nn.MaxPool2d(pool))

        self.cnn = nn.Sequential(
            cb(1, 64, (2, 2)),      # 32 x 80
            cb(64, 128, (2, 2)),    # 16 x 40
            cb(128, 256, (2, 2)),   # 8 x 20
            cb(256, 256, (2, 1)),   # 4 x 20
            cb(256, 384, (2, 1)),   # 2 x 20
            cb(384, 384, (2, 1)),   # 1 x 20  (T = 20)
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
    idx = idx.cpu().numpy()
    conf = conf.cpu().numpy()
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


def ctc_beam_decode(log_probs_tc, beam_width=20, force_len=LENGTH):
    from collections import defaultdict
    lp = log_probs_tc.detach().cpu().numpy() if hasattr(log_probs_tc, "detach") else np.asarray(log_probs_tc)
    p = np.exp(lp)
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
        beams = dict(sorted(nxt.items(), key=lambda kv: kv[1][0] + kv[1][1], reverse=True)[:beam_width])
        s = sum(v[0] + v[1] for v in beams.values())
        if s > 0:
            for k in beams:
                beams[k][0] /= s
                beams[k][1] /= s
    scored = [(pref, pb + pnb) for pref, (pb, pnb) in beams.items() if pref]
    if not scored:
        return "", 0.0
    if force_len is not None:
        cand = [(pref, sc) for pref, sc in scored if len(pref) == force_len]
        if cand:
            scored = cand
    pref, sc = max(scored, key=lambda x: x[1])
    return "".join(I2CH[c] for c in pref), float(sc ** (1.0 / max(len(pref), 1)))


def predict_greedy(model, imgs, device="cpu"):
    X = _to_tensor(imgs, device)
    model.eval()
    with torch.no_grad():
        return ctc_greedy_decode(model(X))


def predict(model, imgs, device="cpu", beam_width=20, force_len=LENGTH):
    X = _to_tensor(imgs, device)
    model.eval()
    with torch.no_grad():
        lps = model(X).log_softmax(2)
    out, confs = [], []
    for b in range(X.shape[0]):
        s, c = ctc_beam_decode(lps[b], beam_width, force_len)
        out.append(s)
        confs.append(c)
    return out, confs
