"""District courts (`<district>.dcourts.gov.in`) captcha: the securimage-wp
WordPress plugin, served at `/?_siwp_captcha&id=<scid>`.

The S3WaaS district sites run the plugin's defaults, which were checked
against its source (securimage-wp, Securimage 4.0.0-RC2/WP):

* 225x85, five characters from Securimage's charset
  `abcdefghijkmnopqrstuvwxzyABCDEFGHJKLMNPQRSTUVWXZY0123456789` (no l, I, O)
* AHGBold at font_ratio 0.4 on a 2x canvas, a smooth per-character angle
  sweep between +-10..20 degrees, random spaces between two groups, and a
  bouncing baseline
* a 3-pole radial warp at perturbation 0.75; 3 sine lines; noise level 5
* background #F2F2F2, with text, lines and noise all #7D7D7D, so the image is
  two-tone

Validation is case-insensitive. The model reads 36 case-folded classes,
A-Z plus 0-9, and the answer is returned in upper case.

The code belongs to the PHP session and is sticky. Re-fetching the image
re-renders the same code with a fresh warp, noise and lines, and a wrong
guess does not change it. `solve_group(images)` votes over several renders
of one code. `captcha_solver.solve_with_retry` does this for you when
`fetch` takes a `new` argument.

Reader: a CRNN + CTC trained on `render()`, a port of Securimage 4's
`doImage`, then fine-tuned on eye-labelled real renders. Weights are in
`dcourts_model.pt`.
"""
import io
import math
import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont

W, H = 225, 85
LENGTH = 5
DRAW_CHARSET = "abcdefghijkmnopqrstuvwxzyABCDEFGHJKLMNPQRSTUVWXZY0123456789"
CLASSES = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"      # case-folded
C2I = {c: i for i, c in enumerate(CLASSES)}
BLANK = len(CLASSES)
IN_H, IN_W = 64, 176                                 # model input (T = 22)
BG, INK = 242, 125

ISCALE = 2
FONT_RATIO = 0.4
PT_TO_PX = 96 / 72                                   # GD renders points at 96 dpi
PERTURBATION = 0.75
NUM_LINES = 3
NOISE_LEVEL = 5

_HERE = os.path.dirname(os.path.abspath(__file__))
_FONT = os.path.join(os.path.dirname(_HERE), "fonts", "AHGBold.ttf")
MODEL = os.path.join(_HERE, "dcourts_model.pt")


# ---------------------------------------------------------------- renderer
def _mt(rng, a, b):
    """PHP mt_rand(a, b): integer arguments, inclusive."""
    return int(rng.integers(int(a), int(b) + 1))


def _frand(rng):
    return 0.0001 * _mt(rng, 0, 9999)


_fonts = {}


def _font(px):
    if px not in _fonts:
        _fonts[px] = ImageFont.truetype(_FONT, px)
    return _fonts[px]


def _dims(font, ch, angle):
    """imagettfbbox -> (width, height, baseline) for a rotated character."""
    if ch == " ":
        return font.getlength(" "), 0.0, 0.0
    x0, y0, x1, y1 = font.getbbox(ch, anchor="ls")      # y up is negative
    a = math.radians(angle)
    ca, sa = math.cos(a), math.sin(a)
    rot = lambda x, y: (x * ca + y * sa, -x * sa + y * ca)
    ll, lr, ur, ul = rot(x0, y1), rot(x1, y1), rot(x1, y0), rot(x0, y0)
    width = lr[0] - ll[0]
    height = max(ll[1] - ul[1], ur[1] - lr[1])
    return width, height, ll[1]


def _draw_char(canvas, font, ch, angle, x, y):
    """imagettftext: pen at (x, y) on the baseline, rotated CCW about it."""
    if ch == " ":
        return
    size = font.size
    pad = 2 * size
    tile = Image.new("L", (2 * pad, 2 * pad), 0)
    ImageDraw.Draw(tile).text((pad, pad), ch, font=font, fill=255, anchor="ls")
    tile = tile.rotate(angle, resample=Image.NEAREST, center=(pad, pad))
    mask = tile.point(lambda v: 255 if v >= 128 else 0)
    canvas.paste(INK, (int(x) - pad, int(y) - pad), mask)


def _word(code, rng, font_scale=1.0):
    """Securimage::drawWord on the ISCALE canvas -> (H*s, W*s) uint8."""
    s = ISCALE
    width, height = W * s, H * s
    font = _font(int(round(height * FONT_RATIO * PT_TO_PX * font_scale)))
    text = code
    if _mt(rng, 1, 100) % 5 > 0:                          # random spaces
        i = _mt(rng, 1, len(text) - 1)
        text = text[:i] + " " * _mt(rng, 1, 3) + text[i:]
    angle0, angleN = _mt(rng, 10, 20), _mt(rng, -20, 10)
    if _mt(rng, 0, 99) % 2 == 0:
        angle0 = -angle0
    if _mt(rng, 0, 99) % 2 == 1:
        angleN = -angleN
    step = abs(angle0 - angleN) / (len(text) - 1)
    step = -step if angle0 > angleN else step
    angle, angles, dims, txt = angle0, [], [], 0.0
    for ch in text:
        angles.append(angle)
        d = _dims(font, ch, angle)
        dims.append(d)
        txt += d[0]
        angle += step
        if angle > 20:
            angle, step = 20, -step
        elif angle < -20:
            angle, step = -20, -step
    canvas = Image.new("L", (width, height), BG)
    cx = math.floor(width / 2 - txt / 2)
    x = _mt(rng, 5 * s, max(cx * 2 - 5 * s, 5 * s))
    y = _mt(rng, dims[0][1], height - 10)
    st = s * _mt(rng, 5, 10)
    up = True
    for i, ch in enumerate(text):
        if y + st + dims[i][2] + 10 * s > height:
            up = False
        elif y - st - dims[i][2] < dims[i][1] + dims[i][2] + 5 * s:
            up = True
        y = y + st if up else y - st
        _draw_char(canvas, font, ch, angles[i], x, y)
        x += dims[i][0]
    return np.asarray(canvas)


def _distorted_copy(tmp, out, rng):
    """Securimage::distortedCopy: three radial poles, nearest sampling of the
    ISCALE canvas, copying only non-background pixels."""
    x = W / 4
    maxX = W - x
    dx = _mt(rng, x / 10, x)
    y = _mt(rng, 20, H - 20)
    dy = _mt(rng, 20, H * 0.7)
    minY, maxY = 20, H - 20
    px, py, rad, amp = [], [], [], []
    for i in range(3):
        px.append(int(x + dx * i) % int(maxX))
        py.append(int(y + dy * i) % int(maxY) + minY)
        rad.append(_mt(rng, H * 0.4, H * 0.8))
        amp.append(PERTURBATION * ((-_frand(rng)) * 0.15 - .15))
    iy, ix = np.mgrid[0:H, 0:W].astype(np.float64)
    X, Y = ix.copy(), iy.copy()
    for i in range(3):
        ddx, ddy = ix - px[i], iy - py[i]
        r = np.sqrt(ddx * ddx + ddy * ddy)
        m = (r <= rad[i]) & (r > 0)
        rs = amp[i] * np.sin(3.14 * r / rad[i])
        X += np.where(m, ddx * rs, 0)
        Y += np.where(m, ddy * rs, 0)
    X, Y = X * ISCALE, Y * ISCALE
    ok = (X >= 0) & (X < W * ISCALE) & (Y >= 0) & (Y < H * ISCALE)
    v = np.full((H, W), BG, np.uint8)
    v[ok] = tmp[Y[ok].astype(int), X[ok].astype(int)]
    out[v != BG] = v[v != BG]


def _noise(out, rng):
    """Securimage::drawNoise: small filled pie arcs, before the text."""
    img = Image.fromarray(out)
    d = ImageDraw.Draw(img)
    level = NOISE_LEVEL * math.log2(math.e)
    for x in range(1, W, 20):
        for y in range(1, H, 20):
            i = 0
            while i < level:
                x1, y1 = _mt(rng, x, x + 20), _mt(rng, y, y + 20)
                size = _mt(rng, 1, 3)
                end = _mt(rng, 180, 360)
                i += 1
                if x1 - size <= 0 and y1 - size <= 0:
                    continue
                r = size / 2
                if size == 1:
                    d.point((x1, y1), fill=INK)
                else:
                    d.pieslice([x1 - r, y1 - r, x1 + r, y1 + r], 0, end, fill=INK)
    out[:] = np.asarray(img)


def _lines(out, rng):
    """Securimage::drawLines: sine-perturbed strokes, after the warp."""
    for line in range(NUM_LINES):
        x = W * (1 + line) / (NUM_LINES + 1)
        x += (0.5 - _frand(rng)) * W / NUM_LINES
        y = _mt(rng, H * 0.1, H * 0.9)
        theta = (_frand(rng) - 0.5) * math.pi * 0.33
        length = _mt(rng, W * 0.4, W * 0.7)
        lwid = _mt(rng, 0, 2)
        k = _frand(rng) * 0.6 + 0.2
        k = k * k * 0.5
        phi = _frand(rng) * 6.28
        step = 0.5
        dx, dy = step * math.cos(theta), step * math.sin(theta)
        amp = 1.5 * _frand(rng) / (k + 5.0 / length)
        x0 = x - 0.5 * length * math.cos(theta)
        y0 = y - 0.5 * length * math.sin(theta)
        i = np.arange(int(length / step + 0.999))
        xs = x0 + i * dx + amp * dy * np.sin(k * i * step + phi)
        ys = y0 + i * dy - amp * dx * np.sin(k * i * step + phi)
        for a, b in zip(xs.astype(int), ys.astype(int)):
            out[max(0, b):max(0, b + lwid + 1), max(0, a):max(0, a + lwid + 1)] = INK


def random_code(rng):
    return "".join(DRAW_CHARSET[_mt(rng, 0, len(DRAW_CHARSET) - 1)] for _ in range(LENGTH))


def render(code, rng, font_scale=1.0):
    """Full captcha for `code` (as drawn, mixed case) -> (H, W) uint8.
    `font_scale` jitters the glyph size in training (GD's point-to-pixel
    conversion is not pinned down exactly)."""
    out = np.full((H, W), BG, np.uint8)
    _noise(out, rng)
    _distorted_copy(_word(code, rng, font_scale), out, rng)
    _lines(out, rng)
    return out


# ---------------------------------------------------------------- model
def to_input(gray):
    """(H, W) uint8 native -> (IN_H, IN_W) float32 in [0, 1], ink bright."""
    im = Image.fromarray(gray).resize((IN_W, IN_H), Image.BILINEAR)
    return (BG - np.asarray(im, np.float32)).clip(0, None) / (BG - INK)


def load(image):
    """bytes / path / PIL -> native (H, W) uint8 grey, or None if not 225x85."""
    im = image if isinstance(image, Image.Image) else Image.open(
        io.BytesIO(image) if isinstance(image, (bytes, bytearray)) else image)
    if im.size != (W, H):
        return None
    return np.asarray(im.convert("L"))


class DcourtsCRNN(nn.Module):
    def __init__(self):
        super().__init__()

        def cb(cin, cout, pool):
            return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1),
                                 nn.BatchNorm2d(cout), nn.ReLU(), nn.MaxPool2d(pool))
        self.cnn = nn.Sequential(
            cb(1, 64, (2, 2)), cb(64, 128, (2, 2)), cb(128, 256, (2, 2)),   # 8 x 22
            cb(256, 256, (2, 1)), cb(256, 384, (2, 1)), cb(384, 384, (2, 1)))  # 1 x 22
        # No dropout inside the LSTM: on MPS (torch 2.8) the LSTM's
        # inter-layer dropout makes its outputs differ from CPU's for the same
        # weights, by up to 0.6, even in eval mode. A model trained on MPS then
        # reads ~95% there and ~1% on CPU, which is where the package runs.
        # A plain Dropout after the LSTM behaves the same on both devices.
        self.rnn = nn.LSTM(384, 256, num_layers=2, bidirectional=True,
                           batch_first=True, dropout=0.0)
        self.drop = nn.Dropout(0.2)
        self.head = nn.Linear(512, len(CLASSES) + 1)

    def forward(self, x):
        f = self.cnn(x).squeeze(2).permute(0, 2, 1)
        f, _ = self.rnn(f)
        return self.head(self.drop(f))                       # [B, T, C+1]


_model = None


def model():
    global _model
    if _model is None:
        m = DcourtsCRNN()
        m.load_state_dict(torch.load(MODEL, map_location="cpu"))
        m.eval()
        _model = m
    return _model


def _beam(lp, beam=25, force_len=LENGTH):
    """Prefix beam search over [T, C] log-probs restricted to LENGTH.
    -> list of (word, prob) best first."""
    from collections import defaultdict
    p = np.exp(lp)
    beams = {(): [1.0, 0.0]}
    for t in range(p.shape[0]):
        pt = p[t]
        nxt = defaultdict(lambda: [0.0, 0.0])
        for pre, (pb, pnb) in beams.items():
            tot = pb + pnb
            nxt[pre][0] += tot * pt[BLANK]
            if pre:
                nxt[pre][1] += pnb * pt[pre[-1]]
            if len(pre) >= force_len:
                continue
            for c in np.argsort(pt[:BLANK])[-6:]:
                c = int(c)
                if pre and c == pre[-1]:
                    nxt[pre + (c,)][1] += pb * pt[c]
                else:
                    nxt[pre + (c,)][1] += tot * pt[c]
        beams = dict(sorted(nxt.items(), key=lambda kv: -sum(kv[1]))[:beam])
        z = sum(sum(v) for v in beams.values())
        for v in beams.values():
            v[0] /= z
            v[1] /= z
    full = [(pre, sum(v)) for pre, v in beams.items() if len(pre) == force_len]
    full.sort(key=lambda kv: -kv[1])
    return ["".join(CLASSES[c] for c in pre) for pre, _ in full], [s for _, s in full]


def log_probs(grays):
    x = torch.tensor(np.stack([to_input(g) for g in grays])[:, None])
    with torch.no_grad():
        return model()(x).log_softmax(2).numpy()


def solve_image(image):
    """-> (text, confidence) for one render. ('', 0.0) if not 225x85."""
    g = load(image)
    if g is None:
        return "", 0.0
    words, probs = _beam(log_probs([g])[0])
    if not words:
        return "", 0.0
    return words[0], float(probs[0] / max(sum(probs), 1e-12))


def rank_group(images):
    """Several renders of ONE code -> [(word, score)] best first. Each
    render's beam posterior is summed per word, so independent noise cancels.
    Score is the mean posterior over renders, in [0, 1]."""
    grays = [g for g in (load(i) for i in images) if g is not None]
    if not grays:
        return []
    score = {}
    for lp in log_probs(grays):
        words, probs = _beam(lp)
        z = max(sum(probs), 1e-12)
        for w, p in zip(words, probs):
            score[w] = score.get(w, 0.0) + p / z
    return sorted(((w, v / len(grays)) for w, v in score.items()), key=lambda kv: -kv[1])


def solve_group(images):
    """Several renders of ONE code -> (text, confidence)."""
    ranked = rank_group(images)
    return (ranked[0][0], float(ranked[0][1])) if ranked else ("", 0.0)
