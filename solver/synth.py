"""Synthetic single-digit generator matching the reverse-engineered captcha style.

Observed style (see analysis): 120x40 white canvas, digits in ~#333 with a
standard sans-serif font, mild rotation, and ~#ccc noise lines drawn *over* the
digits (so where a line crosses a glyph it erases ink and, after thresholding,
leaves a small notch). We reproduce a single digit with those properties and
push it through the SAME crop_digit normalization used at inference, so the
model trains on exactly the distribution it will later see.
"""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .preprocess import DEFAULT_THRESHOLD
from .segment import crop_digit

_FONT_FILES = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Verdana.ttf",
    "/System/Library/Fonts/Supplemental/Tahoma.ttf",
    "/System/Library/Fonts/Supplemental/Trebuchet MS.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Geneva.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
]
_FONT_FILES = [f for f in _FONT_FILES if os.path.exists(f)]

INK = (51, 51, 51)      # #333 digit color
LINE = (204, 204, 204)  # #ccc noise line color

# One cached (path,size)->font map to avoid re-opening fonts every sample.
_font_cache = {}


def _font(path, size):
    key = (path, size)
    if key not in _font_cache:
        _font_cache[key] = ImageFont.truetype(path, size)
    return _font_cache[key]


def make_digit(d, rng, out=28):
    """Render one digit `d` (0-9) with style augmentation -> out x out uint8 (1=ink)."""
    W, H = 48, 56
    img = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = _font(rng.choice(_FONT_FILES), int(rng.integers(30, 40)))
    ch = str(d)
    l, t, r, b = draw.textbbox((0, 0), ch, font=font)
    x = (W - (r - l)) / 2 - l + rng.integers(-2, 3)
    y = (H - (b - t)) / 2 - t + rng.integers(-2, 3)
    draw.text((x, y), ch, font=font, fill=INK)
    # rotate slightly about the center, keep white background
    ang = float(rng.uniform(-13, 13))
    img = img.rotate(ang, resample=Image.BILINEAR, fillcolor=(255, 255, 255))
    # draw 0-2 #ccc lines ON TOP -> creates realistic notches after threshold
    d2 = ImageDraw.Draw(img)
    for _ in range(int(rng.integers(0, 3))):
        y0 = rng.integers(0, H)
        y1 = np.clip(y0 + rng.integers(-14, 15), 0, H - 1)
        d2.line([(-2, y0), (W + 2, y1)], fill=LINE, width=int(rng.integers(1, 3)))
    a = np.asarray(img.convert("L"))
    ink = (a < DEFAULT_THRESHOLD).astype(np.uint8)
    return crop_digit(ink, 0, ink.shape[1], out=out)


def batch(n, rng, out=28):
    """Generate n random labeled digits -> (X: n x out x out float32, y: n int64)."""
    X = np.empty((n, out, out), np.float32)
    y = np.empty((n,), np.int64)
    for i in range(n):
        d = int(rng.integers(0, 10))
        X[i] = make_digit(d, rng, out)
        y[i] = d
    return X, y


# --- full-captcha rendering (for building a 1000-image corpus) -----------------
CAPTCHA_W, CAPTCHA_H = 120, 40
N_DIGITS = 6


def _rotated_glyph(ch, font, rng):
    """Render one digit as an RGBA sprite (transparent bg, #333 ink), rotated."""
    pad = 8
    tmp = Image.new("L", (48, 56), 0)
    d = ImageDraw.Draw(tmp)
    l, t, r, b = d.textbbox((0, 0), ch, font=font)
    d.text(((48 - (r - l)) / 2 - l, (56 - (b - t)) / 2 - t), ch, font=font, fill=255)
    tmp = tmp.rotate(float(rng.uniform(-13, 13)), resample=Image.BILINEAR, expand=True)
    bbox = tmp.getbbox()
    return tmp.crop(bbox) if bbox else tmp


def make_captcha(label, rng):
    """Render a full 6-digit captcha (120x40) in the gstat style -> RGB PIL image.

    Digits sit at a ~20px pitch with per-digit vertical jitter (as the real
    ones do), in #333, then 2-3 #ccc lines are drawn on top."""
    img = Image.new("RGB", (CAPTCHA_W, CAPTCHA_H), (255, 255, 255))
    pitch = CAPTCHA_W / N_DIGITS  # 20
    for k, ch in enumerate(label):
        font = _font(rng.choice(_FONT_FILES), int(rng.integers(26, 33)))
        glyph = _rotated_glyph(ch, font, rng)
        gw, gh = glyph.size
        cx = pitch * (k + 0.5) + rng.integers(-2, 3)
        cy = CAPTCHA_H / 2 + rng.integers(-7, 8)   # vertical jitter
        x = int(cx - gw / 2)
        y = int(cy - gh / 2)
        sprite = Image.new("RGB", glyph.size, INK)
        img.paste(sprite, (x, y), glyph)           # glyph is the alpha mask
    draw = ImageDraw.Draw(img)
    for _ in range(int(rng.integers(2, 4))):
        y0 = rng.integers(0, CAPTCHA_H)
        y1 = int(np.clip(y0 + rng.integers(-16, 17), 0, CAPTCHA_H - 1))
        draw.line([(-2, y0), (CAPTCHA_W + 2, y1)], fill=LINE, width=int(rng.integers(1, 3)))
    return img


def random_label(rng):
    return "".join(str(int(d)) for d in rng.integers(0, 10, size=N_DIGITS))
