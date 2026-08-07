"""MCA portal captcha support (mca.gov.in): 200x80 RGB, 6 mixed-case alnum chars.

Composition (measured over the real corpus):

* the image uses **exactly three colours** — background 230, noise lines 150,
  text 0 — with no antialiasing anywhere;
* the noise lines are drawn *under* the text, so an exact match on pure black
  recovers every glyph whole, unbroken. Noise therefore cannot affect
  recognition at all and the generator does not need to model it;
* text sits on a fixed baseline (y≈50) starting at x≈66, but **each character
  gets its own size and weight/slant** — cap heights range over 12..21px within
  a single captcha, so the reader must be scale-invariant.

That makes this the easiest of the three captchas: preprocessing is a colour
equality test, and the only domain gap left is the font pool.
"""
import io
import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont

CHARSET = ("0123456789"
           "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
           "abcdefghijklmnopqrstuvwxyz")
CH2I = {c: i for i, c in enumerate(CHARSET)}
I2CH = {i: c for i, c in enumerate(CHARSET)}
N_CLASSES = len(CHARSET)      # 62
BLANK = N_CLASSES
LENGTH = 6

W, H = 200, 80
# Text window measured over the corpus (x 62..165, y 29..55) plus margin.
CROP = (56, 22, 176, 62)      # left, top, right, bottom -> 120 x 40
IN_W, IN_H = 192, 64

BG, LINE, INK = 230, 150, 0

BASELINE = 50
START_X = 66

_HERE = os.path.dirname(os.path.abspath(__file__))
_FONTS_DIR = os.path.join(os.path.dirname(_HERE), "fonts")
_FACES = [
    "DejaVuSans.ttf", "DejaVuSans-Bold.ttf",
    "DejaVuSans-Oblique.ttf", "DejaVuSans-BoldOblique.ttf",
    "DejaVuSansCondensed.ttf", "DejaVuSansCondensed-Bold.ttf",
    "DejaVuSansCondensed-Oblique.ttf", "DejaVuSansCondensed-BoldOblique.ttf",
]
FONT_FILES = [os.path.join(_FONTS_DIR, f) for f in _FACES
              if os.path.exists(os.path.join(_FONTS_DIR, f))]

_font_cache = {}


def _font(path, size):
    key = (path, int(size))
    if key not in _font_cache:
        _font_cache[key] = ImageFont.truetype(path, int(size))
    return _font_cache[key]


def random_label(rng):
    return "".join(CHARSET[i] for i in rng.integers(0, N_CLASSES, size=LENGTH))


def encode_label(label):
    return [CH2I[c] for c in label]


# --------------------------------------------------------------------------
# preprocessing — exact, because ink is the only pure-black thing in the image
# --------------------------------------------------------------------------
def text_mask(rgb):
    """(H,W,3) uint8 -> bool mask of glyph pixels."""
    a = np.asarray(rgb)
    return (a[..., 0] == 0) & (a[..., 1] == 0) & (a[..., 2] == 0)


def _mask_to_input(mask):
    """bool (H,W) -> (IN_H, IN_W) uint8, ink white on black."""
    l, t, r, b = CROP
    sub = mask[t:b, l:r]
    im = Image.fromarray((sub * 255).astype(np.uint8))
    return np.asarray(im.resize((IN_W, IN_H), Image.BILINEAR), dtype=np.uint8)


def load_real(path_or_bytes):
    if isinstance(path_or_bytes, (bytes, bytearray)):
        im = Image.open(io.BytesIO(path_or_bytes))
    elif isinstance(path_or_bytes, Image.Image):
        im = path_or_bytes
    else:
        im = Image.open(path_or_bytes)
    return _mask_to_input(text_mask(im.convert("RGB")))


# --------------------------------------------------------------------------
# synthetic generation — only the text matters, so only the text is modelled
# --------------------------------------------------------------------------
def _ink_box(fnt, ch):
    img = Image.new("L", (120, 120), 0)
    ImageDraw.Draw(img).text((30, 90), ch, font=fnt, fill=255, anchor="ls")
    a = np.asarray(img) > 127
    if not a.any():
        return 0, 0
    xs = np.where(a.any(0))[0]
    return int(xs.min() - 30), int(xs.max() - 30)


def make_mask(label, rng):
    """Render one synthetic captcha's glyph mask, matching the real geometry."""
    canvas = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(canvas)
    x = START_X + float(rng.uniform(-2, 2))
    base = BASELINE + float(rng.uniform(-1, 1))
    # Font size is essentially CONSTANT within an image. Measured on 60 real
    # captchas, the height spread among full-height glyphs (capitals, digits,
    # ascenders) is only (max-min)/mean = 0.044. What looks like per-character
    # size randomisation is mostly cap-height vs x-height.
    #
    # This matters far more than it looks: because every glyph shares one
    # baseline and one size, *relative height disambiguates case* for the
    # x-height letters. Measured on real labelled glyphs, o/O, c/C, x/X and u/U
    # separate at 100% and s/S at 95% on relative height alone. An over-jittered
    # generator destroys that cue and teaches the model to ignore it — which is
    # exactly what an earlier version did (spread 0.208, ~4.8x too wide).
    base_size = float(rng.uniform(17, 28))
    for ch in label:
        path = FONT_FILES[int(rng.integers(0, len(FONT_FILES)))]
        size = int(round(np.clip(base_size * (1.0 + rng.normal(0, 0.015)), 15, 30)))
        fnt = _font(path, size)
        # PIL antialiases; the real renderer does not, so threshold hard below.
        d.text((x, base + rng.uniform(-0.6, 0.6)), ch, font=fnt, fill=255, anchor="ls")
        lo, hi = _ink_box(fnt, ch)
        x += (hi - lo + 1) + rng.uniform(0.0, 2.5)
    return np.asarray(canvas) > 127


def make_input(label, rng):
    return _mask_to_input(make_mask(label, rng))


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
class McaCRNN(nn.Module):
    """Single-channel CRNN+CTC over the binarised glyph mask."""

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
