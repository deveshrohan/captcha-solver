"""GST portal captcha support (services.gst.gov.in): 182x50 RGB, 6 digits.

Composition (reverse-engineered, then confirmed against the library source):
the image is produced by **SimpleCaptcha** (Java, nl.captcha). Evidence:

* `FishEyeGimpyRenderer` computes its hatch spacing as
  `hspace = height/(height/7 + 1)` and `vspace = width/(width/7 + 1)`, which for
  182x50 gives exactly 6 and 6 — the measured grid pitch. Grid lines land on
  x = 6,12,...,180 and y = 6,12,...,48, exactly as measured.
* The same renderer then applies a fisheye inside a circle of radius
  `ranInt(width/4, width/3)` = 45..60 centred on (91, 25). Real captchas have
  perfectly straight grid lines outside that circle and warped ones inside.
* `GradiatedBackgroundProducer` paints a Java `GradientPaint` from (0,0)
  DARK_GRAY to (width,height) WHITE. Predicted vs measured background matches to
  a mean absolute error of ~0.9 gray levels at every radius, which also proves
  the background is composited *behind* the (already-fisheyed) ink layer rather
  than being distorted with it — matching `Captcha.Builder.build()`.

Two pieces are NOT stock SimpleCaptcha and are fitted from real samples instead:

* the noise line is red, spans the full width and has no antialiasing, whereas
  stock `CurvedLineNoiseProducer` spans 0.1w..0.9w with antialiasing on;
* the font is a Verdana-lineage bold face (DejaVu Sans Condensed Bold fits the
  real glyphs best), not the stock Arial/Courier — see the font notes below.

Layer order (from Captcha.Builder): transparent layer <- text <- red curve <-
grid + fisheye; then the gradient behind it; then a 1px border on top.
"""
import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont

CHARSET = "0123456789"
CH2I = {c: i for i, c in enumerate(CHARSET)}
I2CH = {i: c for i, c in enumerate(CHARSET)}
N_CLASSES = len(CHARSET)
BLANK = N_CLASSES
LENGTH = 6

W, H = 182, 50                 # native size
IN_W, IN_H = 192, 64           # model input (T = 24 after the CNN)

# FishEyeGimpyRenderer's own arithmetic, kept as integer division on purpose.
HSPACE = H // (H // 7 + 1)     # 6
VSPACE = W // (W // 7 + 1)     # 6
FISH_MIN, FISH_MAX = W // 4, W // 3   # ranInt bounds -> 45..60 inclusive

# Java's Color.DARK_GRAY -> Color.WHITE
BG_FROM, BG_TO = 64.0, 255.0

# The exact face is the one thing that could not be pinned down: the server is
# Linux, so the Java font name resolves through fontconfig to whatever is
# installed. Matching real glyph shapes (IoU on the pristine strip left of the
# fisheye circle) ranks DejaVu Sans Condensed Bold first (0.775), then DejaVu
# Sans Bold / Tahoma Bold / Verdana Bold (~0.74) — all Verdana-lineage humanist
# sans faces. Rather than bet on one, training randomises over the shortlist;
# the *geometry* below is measured and is what actually has to be right.
_HERE = os.path.dirname(os.path.abspath(__file__))
_FONTS_DIR = os.path.join(os.path.dirname(_HERE), "fonts")
_FONT_CANDIDATES = [
    os.path.join(_FONTS_DIR, "DejaVuSansCondensed-Bold.ttf"),
    os.path.join(_FONTS_DIR, "DejaVuSans-Bold.ttf"),
    "/System/Library/Fonts/Supplemental/Tahoma Bold.ttf",
    "/System/Library/Fonts/Supplemental/Verdana Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]
FONT_FILES = [p for p in _FONT_CANDIDATES if os.path.exists(p)]
FONT_PATH = FONT_FILES[0] if FONT_FILES else None

# Measured on 150 real captchas after undoing the fisheye and masking the grid:
# ink spans x 13..125 (width 112) for 6 digits, ink height 28, baseline ~37.
TARGET_MEAN_INK_W = 112.0 / LENGTH     # 18.67 px per digit
BASELINE = 37
START_X = 9

_font_cache = {}
_inkw_cache = {}
_size_cache = {}


def _font(size, path=None):
    key = (path or FONT_PATH, int(size))
    if key not in _font_cache:
        _font_cache[key] = ImageFont.truetype(key[0], key[1])
    return _font_cache[key]


def _ink_widths(path, size):
    """Per-digit ink (visual) width — SimpleCaptcha advances the pen by this,
    not by the advance width, which is why real digits sit tightly packed."""
    key = (path, int(size))
    if key not in _inkw_cache:
        fnt = _font(size, path)
        widths = {}
        for d in CHARSET:
            img = Image.new("L", (160, 160), 0)
            ImageDraw.Draw(img).text((40, 120), d, font=fnt, fill=255, anchor="ls")
            a = np.asarray(img) > 100
            xs = np.where(a.any(0))[0]
            widths[d] = int(xs.max() - xs.min() + 1) if len(xs) else int(size * 0.5)
        _inkw_cache[key] = widths
    return _inkw_cache[key]


def _base_size(path):
    """Pick the pt size whose mean digit ink width matches the measured 18.67px,
    so every face in the pool reproduces the real text geometry."""
    if path not in _size_cache:
        best, berr = 36, 1e9
        for s in range(26, 50):
            mw = float(np.mean(list(_ink_widths(path, s).values())))
            err = abs(mw - TARGET_MEAN_INK_W)
            if err < berr:
                best, berr = s, err
        _size_cache[path] = best
    return _size_cache[path]


def random_label(rng):
    return "".join(CHARSET[i] for i in rng.integers(0, N_CLASSES, size=LENGTH))


# --------------------------------------------------------------------------
# background: Java GradientPaint((0,0), DARK_GRAY, (W,H), WHITE)
# --------------------------------------------------------------------------
_ys, _xs = np.mgrid[0:H, 0:W]
_T = np.clip((_xs * W + _ys * H) / float(W * W + H * H), 0.0, 1.0)
GRADIENT = np.rint(BG_FROM + _T * (BG_TO - BG_FROM)).astype(np.uint8)


def background():
    return np.repeat(GRADIENT[:, :, None], 3, axis=2).copy()


# --------------------------------------------------------------------------
# ink layer: text, then the red curve, then the hatch grid
# --------------------------------------------------------------------------
def _render_text(label, rng, jitter=True, font_path=None, font_size=None):
    """Black antialiased digits on a transparent layer.

    SimpleCaptcha's DefaultWordRenderer advances the pen by each glyph's
    *visual* (ink) width rather than its advance width, so the digits sit
    tightly packed — which is what real GST captchas show.
    """
    path = font_path or (rng.choice(FONT_FILES) if jitter else FONT_PATH)
    size = font_size if font_size is not None else _base_size(path) + (
        int(rng.integers(-1, 2)) if jitter else 0)
    fnt = _font(size, path)
    widths = _ink_widths(path, size)

    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    x = float(START_X + (rng.uniform(-1.5, 1.5) if jitter else 0.0))
    base = BASELINE + (rng.uniform(-1.0, 1.0) if jitter else 0.0)
    for ch in label:
        d.text((x, base), ch, font=fnt, fill=(0, 0, 0, 255), anchor="ls")
        x += widths[ch] + (rng.uniform(-0.4, 0.4) if jitter else 0.0)
    return layer


def _flatten_cubic(p0, p1, p2, p3, n=60):
    t = np.linspace(0.0, 1.0, n)[:, None]
    pts = ((1 - t) ** 3 * np.array(p0) + 3 * (1 - t) ** 2 * t * np.array(p1)
           + 3 * (1 - t) * t ** 2 * np.array(p2) + t ** 3 * np.array(p3))
    return pts


def _draw_noise(layer, rng):
    """Red curve, full width, hard-edged (no antialiasing, exactly one colour)."""
    d = ImageDraw.Draw(layer)
    p0 = (0.0, H * rng.random())
    p1 = (W * 0.10, H * rng.random())
    p2 = (W * 0.25, H * rng.random())
    p3 = (float(W), H * rng.random())
    pts = _flatten_cubic(p0, p1, p2, p3)
    ipts = [(int(px), int(py)) for px, py in pts]
    for a, b in zip(ipts[:-1], ipts[1:]):
        d.line([a, b], fill=(255, 0, 0, 255), width=3)
    return layer


def _draw_grid(layer):
    d = ImageDraw.Draw(layer)
    for i in range(HSPACE, H, HSPACE):
        d.line([(0, i), (W, i)], fill=(0, 0, 0, 255), width=1)
    for i in range(VSPACE, W, VSPACE):
        d.line([(i, 0), (i, H)], fill=(0, 0, 0, 255), width=1)
    return layer


def _fish_formula(s):
    return -0.75 * s ** 3 + 1.5 * s ** 2 + 0.25 * s


def _fisheye(arr, distance):
    """Port of FishEyeGimpyRenderer's distortion, including Java's (int) truncation."""
    wmid, hmid = W // 2, H // 2
    relx = _xs - wmid
    rely = _ys - hmid
    d1 = np.sqrt(relx.astype(np.float64) ** 2 + rely.astype(np.float64) ** 2)
    inside = d1 < distance
    with np.errstate(divide="ignore", invalid="ignore"):
        fac = (_fish_formula(d1 / distance) * distance) / d1
    fac = np.where(d1 == 0, 0.0, fac)            # Java: (int)NaN == 0 -> maps to centre
    j2 = wmid + np.trunc(fac * relx).astype(np.int64)
    k2 = hmid + np.trunc(fac * rely).astype(np.int64)
    np.clip(j2, 0, W - 1, out=j2)
    np.clip(k2, 0, H - 1, out=k2)
    out = arr.copy()
    out[inside] = arr[k2[inside], j2[inside]]
    return out


def _add_border(rgb):
    rgb[0, :] = 0
    rgb[H - 1, :] = 0
    rgb[:, 0] = 0
    rgb[:, W - 1] = 0
    return rgb


def make_image(label, rng, jitter=True, font_path=None, font_size=None):
    """Render one synthetic GST captcha as an (H, W, 3) uint8 array."""
    layer = _render_text(label, rng, jitter, font_path, font_size)
    _draw_noise(layer, rng)
    _draw_grid(layer)

    lay = np.asarray(layer).astype(np.uint8)
    distance = int(rng.integers(FISH_MIN, FISH_MAX + 1))
    lay = _fisheye(lay, distance)

    rgb = background()
    alpha = lay[..., 3:4].astype(np.float32) / 255.0
    out = np.rint(lay[..., :3] * alpha + rgb * (1 - alpha)).astype(np.uint8)
    return _add_border(out)


# --------------------------------------------------------------------------
# I/O helpers
# --------------------------------------------------------------------------
def load_real(path_or_bytes):
    """-> (IN_H, IN_W, 3) uint8, the model's input format."""
    if isinstance(path_or_bytes, (bytes, bytearray)):
        import io
        im = Image.open(io.BytesIO(path_or_bytes))
    elif isinstance(path_or_bytes, Image.Image):
        im = path_or_bytes
    else:
        im = Image.open(path_or_bytes)
    im = im.convert("RGB").resize((IN_W, IN_H), Image.BILINEAR)
    return np.asarray(im, dtype=np.uint8)


def to_input(arr_hw3):
    """(H,W,3) uint8 native -> (IN_H, IN_W, 3) uint8."""
    return np.asarray(Image.fromarray(arr_hw3).resize((IN_W, IN_H), Image.BILINEAR), dtype=np.uint8)


def encode_label(label):
    return [CH2I[c] for c in label]


# --------------------------------------------------------------------------
# model: same CRNN+CTC recipe that took the securimage reader to ~99%
# --------------------------------------------------------------------------
class GstCRNN(nn.Module):
    """CNN -> collapse height -> width sequence -> BiLSTM -> per-step logits.

    Input is 3-channel RGB on purpose: the noise line is pure red and the ink is
    pure black, so a grayscale conversion would map the line to luminance 76 and
    make it indistinguishable from ink against the darker end of the gradient.
    """

    def __init__(self, n_classes=N_CLASSES):
        super().__init__()

        def cb(cin, cout, pool):
            return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1),
                                 nn.BatchNorm2d(cout), nn.ReLU(), nn.MaxPool2d(pool))

        # 64 x 192 -> height collapses to 1, width 192 -> 24
        self.cnn = nn.Sequential(
            cb(3, 64, (2, 2)),      # 32 x 96
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
        f = f.squeeze(2).permute(0, 2, 1).contiguous()   # cuDNN/MPS LSTM needs contiguous
        f, _ = self.rnn(f)
        return self.head(f)


def _to_tensor(imgs, device):
    x = np.asarray(imgs, dtype=np.float32) / 255.0        # [B,H,W,3]
    x = np.ascontiguousarray(np.transpose(x, (0, 3, 1, 2)))   # [B,3,H,W]
    return torch.tensor(x, device=device)


def ctc_beam_decode(log_probs_tc, beam_width=25, force_len=LENGTH):
    """Prefix beam search restricted to exactly `force_len` characters."""
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
    if not scored:                       # early in training the net emits only blanks
        return "", 0.0
    if force_len is not None:
        cand = [(pref, sc) for pref, sc in scored if len(pref) == force_len]
        if cand:
            scored = cand
    pref, sc = max(scored, key=lambda x: x[1])
    conf = float(sc ** (1.0 / max(len(pref), 1)))
    return "".join(I2CH[c] for c in pref), conf


def ctc_greedy_decode(logits_btc):
    """Fast argmax-collapse decode — used for per-epoch validation."""
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


def predict_greedy(model, imgs, device="cpu"):
    X = _to_tensor(imgs, device)
    model.eval()
    with torch.no_grad():
        logits = model(X)
    return ctc_greedy_decode(logits)


def predict(model, imgs, device="cpu", beam_width=25, force_len=LENGTH):
    """imgs: [B, IN_H, IN_W, 3] uint8 -> (texts, confidences)."""
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


# --------------------------------------------------------------------------
# optional: undo the fisheye (used by the v2 reader; the distortion is
# invertible because f(s) is monotonic on [0,1])
# --------------------------------------------------------------------------
_INV_S = np.linspace(0.0, 1.0, 20001)
_INV_F = _fish_formula(_INV_S)


def undistort(arr, distance):
    """Inverse of _fisheye for a known radius."""
    wmid, hmid = W // 2, H // 2
    relx = _xs - wmid
    rely = _ys - hmid
    r_in = np.sqrt(relx.astype(np.float64) ** 2 + rely.astype(np.float64) ** 2)
    inside = (r_in < distance) & (r_in > 0)
    s_out = np.interp(np.clip(r_in / distance, 0, 1), _INV_F, _INV_S)
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.where(inside, s_out * distance / np.maximum(r_in, 1e-9), 1.0)
    sx = np.clip(np.rint(wmid + k * relx).astype(int), 0, W - 1)
    sy = np.clip(np.rint(hmid + k * rely).astype(int), 0, H - 1)
    out = arr.copy()
    out[inside] = arr[sy[inside], sx[inside]]
    return out


def _grid_score(arr):
    """Grid lines are black and land on a known 6px lattice *only* when the
    fisheye has been correctly undone, so this scores a candidate radius."""
    g = (arr[..., 0] == arr[..., 1]) & (arr[..., 1] == arr[..., 2])
    black = g & (arr[..., 0] < 40)
    on = sum(black[:, x].sum() for x in range(VSPACE, W, VSPACE))
    on += sum(black[y, :].sum() for y in range(HSPACE, H, HSPACE))
    off = sum(black[:, x + d].sum() for x in range(VSPACE, W - 2, VSPACE) for d in (-2, 2))
    return on - 0.5 * off


def estimate_distance(arr):
    """Recover the per-image fisheye radius by maximising grid alignment."""
    best, bs = FISH_MIN, -1e18
    for D in range(FISH_MIN, FISH_MAX + 1):
        s = _grid_score(undistort(arr, D))
        if s > bs:
            best, bs = D, s
    return best
