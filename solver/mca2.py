"""MCA portal captcha, format served since ~Nov 2026 (kind `mca2`).

Same endpoint as `mca` (`www.mca.gov.in/bin/mca/generateCaptchaWithHMAC`, a
multipart/mixed body with a PNG and a WAV part) and the same 200x80 size, but
a different generator. The old one drew exactly three flat colours; this one
draws ~1700. Measured on the real corpus:

* background (242,244,246), a pale rounded border, ~200 pale speckles;
* three wavy polylines ~2 px wide across the full width, one in each band
  (top / middle / bottom), each a palette colour;
* six characters, each in its own palette colour, DejaVu-Sans-like face in
  plain / bold / oblique, cap height ~13-16 px, centred on slots at x ~24 +
  30*i (jitter ~+-7 px) with
  a random height (y centre ~24..58) and a small tilt;
* a soft blur over the whole picture (a 1-px line reads 184/138/184);
* ink palette (darkest cores): blue, green, dark green, crimson, navy, purple.
  Lines and speckles draw from the same palette, so colour does NOT separate
  text from noise, but every hue has min(R,G,B) ~ 50..80, so the darkness map
  `1 - min(R,G,B)/255` puts all six at the same weight.

Charset from the labelled census: 2-9 and A-Z minus 0, 1, 5, 8, I, O (30
symbols; each present one appears ~12x per 60 captchas, the absent six never).

The accessibility WAV is six spoken characters, but the clips are not
byte-identical across instances (two '3's correlate at 0.21), so the audio is
not an exact-match shortcut; this reader uses the image.

Reader: the glyphs never overlap and sit in fixed slots, so one small CNN
classifies a window around each slot (train_mca2.py: renderer below plus
hand-labelled reals). A CRNN+CTC over the same input sat at the blank-only
plateau for 13 epochs; the slot classifier needs no alignment.
"""
import io
import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFilter, ImageFont

CHARSET = "234679ABCDEFGHJKLMNPQRSTUVWXYZ"
CH2I = {c: i for i, c in enumerate(CHARSET)}
I2CH = {i: c for i, c in enumerate(CHARSET)}
N_CLASSES = len(CHARSET)
BLANK = N_CLASSES
LENGTH = 6

W, H = 200, 80
CROP = (0, 8, 200, 72)          # glyphs live in y ~12..68; native resolution
IN_W, IN_H = CROP[2] - CROP[0], CROP[3] - CROP[1]

BG = (242, 244, 246)
BORDER = (176, 190, 214)
# Ink palette: darkest cores measured on reals, pushed slightly darker because
# the blur lifts every stroke core toward the background.
PALETTE = [(52, 104, 180), (52, 136, 92), (56, 92, 64), (168, 48, 72),
           (56, 68, 120), (92, 68, 144)]

# Slot geometry (native px), measured on the labelled reals.
SLOT_X0, SLOT_DX = 24.0, 30.0
Y_CENTRE = (24.0, 58.0)
FONT_PX = (18, 23)               # DejaVu cap height ~0.73 em -> 13..17 px
ROT_DEG = 18.0

_HERE = os.path.dirname(os.path.abspath(__file__))
_FONTS_DIR = os.path.join(os.path.dirname(_HERE), "fonts")
# Java's `SansSerif` logical font resolves to DejaVu Sans on a typical Linux
# server; the letterforms (J, Q, G) match it. Plain / bold / oblique all occur.
_FACES = ["DejaVuSans.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans-Oblique.ttf",
          "DejaVuSans-BoldOblique.ttf"]
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
# preprocessing
# --------------------------------------------------------------------------
def darkness(rgb):
    a = np.asarray(rgb, dtype=np.float32)
    return 1.0 - a.min(2) / 255.0


def _to_input(rgb):
    d = darkness(rgb)
    l, t, r, b = CROP
    return np.clip(d[t:b, l:r] * 255.0, 0, 255).astype(np.uint8)


def _open(image):
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, (bytes, bytearray)):
        return Image.open(io.BytesIO(image))
    return Image.open(image)


def load_real(image):
    return _to_input(_open(image).convert("RGB"))


def looks_like_mca2(pil):
    """The old `mca` format is exactly three flat colours; this one has
    hundreds. Used by the 200x80 router."""
    a = np.asarray(pil.convert("RGB"))
    return len(np.unique(a.reshape(-1, 3)[::7], axis=0)) > 16


# --------------------------------------------------------------------------
# synthetic renderer
# --------------------------------------------------------------------------
SS = 2   # supersample, then downscale: gives antialiased strokes and sub-px tilt


def _mix(c, t):
    """Blend colour c toward the background by t (0 = c, 1 = bg)."""
    return tuple(int(round(ci * (1 - t) + bi * t)) for ci, bi in zip(c, BG))


def _wave(rng, y0):
    amp = rng.uniform(2.0, 5.5)
    period = rng.uniform(90, 140)
    phase = rng.uniform(0, 2 * np.pi)
    tilt = rng.uniform(-0.03, 0.03)
    xs = np.arange(-4, W + 5, 3.0)
    ys = (y0 + amp * np.sin(2 * np.pi * xs / period + phase) + tilt * (xs - W / 2)
          + rng.normal(0, 0.35, len(xs)))
    return [(float(x) * SS, float(y) * SS) for x, y in zip(xs, ys)]


def make_rgb(label, rng):
    im = Image.new("RGB", (W * SS, H * SS), BG)
    d = ImageDraw.Draw(im)

    # speckle (under everything else; pale)
    for _ in range(int(rng.integers(150, 260))):
        c = _mix(PALETTE[int(rng.integers(len(PALETTE)))], rng.uniform(0.35, 0.75))
        x, y = rng.uniform(0, W) * SS, rng.uniform(0, H) * SS
        s = SS * (1 if rng.random() < 0.6 else 2)
        d.rectangle([x, y, x + s - 1, y + s - 1], fill=c)

    # border: rounded rectangle, slightly ragged
    inset = rng.uniform(2, 4)
    d.rounded_rectangle([inset * SS, inset * SS, (W - inset) * SS, (H - inset) * SS],
                        radius=int(rng.uniform(6, 10) * SS),
                        outline=_mix(BORDER, rng.uniform(-0.1, 0.2)), width=2 * SS)

    # glyphs, each on its own layer (rotated), composited over the speckle
    for i, ch in enumerate(label):
        fnt = _font(FONT_FILES[int(rng.integers(len(FONT_FILES)))],
                    int(rng.integers(FONT_PX[0], FONT_PX[1] + 1)) * SS)
        n = 40 * SS
        layer = Image.new("L", (n, n), 0)
        ImageDraw.Draw(layer).text((n / 2, n / 2), ch, font=fnt, fill=255, anchor="mm")
        layer = layer.rotate(float(rng.uniform(-ROT_DEG, ROT_DEG)),
                             resample=Image.BICUBIC)
        cx = (SLOT_X0 + SLOT_DX * i + rng.normal(0, 2.5)) * SS
        cy = rng.uniform(*Y_CENTRE) * SS
        col = _mix(PALETTE[int(rng.integers(len(PALETTE)))], rng.uniform(-0.1, 0.1))
        im.paste(Image.new("RGB", (n, n), col),
                 (int(cx - n / 2), int(cy - n / 2)), layer)

    # three wavy lines, one per band, drawn OVER the text: on reals a line
    # can hide a T's whole crossbar, leaving a bare stem
    d = ImageDraw.Draw(im)
    for lo, hi in [(14, 30), (34, 50), (52, 68)]:
        c = PALETTE[int(rng.integers(len(PALETTE)))]
        d.line(_wave(rng, rng.uniform(lo, hi)), fill=_mix(c, rng.uniform(0.0, 0.25)),
               width=int(round(rng.uniform(1.6, 2.4) * SS)), joint="curve")

    im = im.resize((W, H), Image.LANCZOS)
    return im.filter(ImageFilter.GaussianBlur(rng.uniform(0.45, 0.8)))


def make_input(label, rng):
    return _to_input(make_rgb(label, rng))


# --------------------------------------------------------------------------
# model: one shared CNN classifies each of the six slots
# --------------------------------------------------------------------------
# The glyphs sit in six fixed slots (centres ~24 + 30*i, jitter ~+-7 px) and
# never overlap, so the read is six independent 30-way classifications of a
# window around each slot -- no sequence alignment to learn.
HALF = 20                        # window = 2*HALF wide, full crop height
SHIFTS = (-4, 0, 4)              # test-time shifts, probabilities averaged


def slot_windows(x, shift=0):
    """(IN_H, IN_W) uint8 darkness -> (LENGTH, IN_H, 2*HALF) uint8."""
    pad = np.pad(x, ((0, 0), (HALF + 8, HALF + 8)))
    out = np.empty((LENGTH, x.shape[0], 2 * HALF), np.uint8)
    for i in range(LENGTH):
        c = int(round(SLOT_X0 + SLOT_DX * i)) + shift + HALF + 8 - CROP[0]
        out[i] = pad[:, c - HALF:c + HALF]
    return out


class Mca2SlotCNN(nn.Module):
    def __init__(self, n_classes=N_CLASSES):
        super().__init__()

        def cb(cin, cout):
            return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout),
                                 nn.ReLU(), nn.Conv2d(cout, cout, 3, padding=1),
                                 nn.BatchNorm2d(cout), nn.ReLU(), nn.MaxPool2d(2))

        self.cnn = nn.Sequential(cb(1, 32), cb(32, 64), cb(64, 128), cb(128, 192))  # 4 x 2
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3),
                                  nn.Linear(192 * 4 * (2 * HALF // 16), 256), nn.ReLU(),
                                  nn.Linear(256, n_classes))

    def forward(self, x):
        return self.head(self.cnn(x))


def _to_tensor(wins, device="cpu"):
    return torch.tensor(np.asarray(wins, np.float32) / 255.0, device=device)[:, None]


def predict(model, imgs, device="cpu"):
    """imgs: (N, IN_H, IN_W) uint8 -> (texts, confidences). Confidence is the
    lowest per-slot probability (after averaging the shifted windows)."""
    model.eval()
    imgs = np.asarray(imgs)
    probs = 0
    with torch.no_grad():
        for sh in SHIFTS:
            wins = np.concatenate([slot_windows(x, sh) for x in imgs])
            probs = probs + torch.softmax(model(_to_tensor(wins, device)), 1).cpu().numpy()
    probs = probs.reshape(len(imgs), LENGTH, N_CLASSES) / len(SHIFTS)
    idx = probs.argmax(2)
    texts = ["".join(I2CH[int(c)] for c in row) for row in idx]
    confs = [float(probs[n, np.arange(LENGTH), idx[n]].min()) for n in range(len(imgs))]
    return texts, confs
