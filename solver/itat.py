"""ITAT portal captcha support (itat.gov.in): 150x42 RGB, 6 mixed-case alnum chars.

Composition (measured over the real corpus, see README for the derivation):

* the stack is **CodeIgniter + PHP GD** — the PNG carries libgd's `pHYs` = 3780
  ppm (96 DPI) fingerprint and the endpoint sets `ci_session` cookies. It is a
  heavily customised `create_captcha()`: truecolor, six mixed-case alnum glyphs
  in **one thin sans face**, each at its own **random rotation**, drawn near-black
  and antialiased on white;
* the noise is **light-blue straight lines plus colour speckle, drawn OVER the
  text** (632 blue pixels sit flanked by ink on both sides across 20 images, the
  signature of a line cutting through a stroke). So this is Udyam's layering, not
  MCA's — but here the ink is near-black (0,0,0) while every noise colour is much
  brighter, so a **darkness projection** `d = 1 - max(r,g,b)/255` separates them:
  ink -> ~1, white -> 0, light-blue line -> ~0.01, grey speck -> ~0.37.

That projection is the whole preprocessing story, and it is applied identically
to real and synthetic images (`_to_input`), so there is no train/test skew. What
it does NOT do is repair the fragmentation: where an opaque blue line crosses a
stroke the darkness there drops to ~0, punching a hole. Rather than try to fill
those holes, the generator *reproduces* them — it draws the same blue-lines-over
-text before projecting — so the model is trained on the same broken strokes it
will read. That is the Securimage lesson (model the noise that touches the ink),
applied through a colour projection that also throws away the parts that don't.

Random per-character rotation is why this is a CRNN and not a template cover like
Kaveri/Udyam/EPFO: those need pixel-identical fixed-pose glyphs, and a random
angle destroys that. It is a whole-image reader over the darkness map.
"""
import io
import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

# The real captcha renders MIXED CASE (clear lowercase p/u/w/v with descenders and
# x-heights appear alongside capitals). Case is read **case-insensitively** here:
# every glyph is folded to one uppercase class. Two reasons — (1) for full-height
# letters (K/k, S/s at cap size) case is not recoverable from the glyph at all, the
# same wall that caps the case-sensitive MCA reader; (2) the submit contract could
# not be verified (that would mean posting to a live government form), so whether
# ITAT even validates case is unknown, and folding is the choice that fails safe if
# it is case-insensitive. See the README for the caveat and how to make it
# case-sensitive if a portal check later shows it matters.
#
# The class set is then the homoglyph-free subset — `0 1 I O L` are excluded:
# none occur in the labelled census (666 chars, zero `L`; P(L absent by chance)
# ~5e-10), so ITAT avoids them exactly as EPFO/Kaveri/Udyam avoid their own. `L`
# matters especially because a lowercase `l` is a bare vertical bar; dropping the
# class removes that trap. Folding case removes the case-homoglyphs (c/C, s/S, ...)
# and dropping 0/1/I/O/L removes the digit/letter ones, so NO ambiguous pair
# remains and there is no `AMBIGUOUS` entry — the property that lets EPFO and
# Udyam reach ~100%.
CHARSET = "23456789ABCDEFGHJKMNPQRSTUVWXYZ"
CH2I = {c: i for i, c in enumerate(CHARSET)}
I2CH = {i: c for i, c in enumerate(CHARSET)}
N_CLASSES = len(CHARSET)
BLANK = N_CLASSES
LENGTH = 6

W, H = 150, 42
# The glyphs fill almost the whole frame (ink bbox ~116x31, baseline ~y40), and
# rotation swings them further, so the crop is the whole image; the darkness
# projection makes the noise-only margins near-black anyway.
CROP = (0, 0, W, H)
IN_W, IN_H = 192, 64

# --- geometry, measured on the real corpus (darkness > 0.6) ---
#   ink bbox height  median 31   (p10 29, p90 33)  -> cap height ~31px
#   ink bbox width   median 116  for 6 glyphs
#   ink pixels       median 414
#   stroke width     ~2.2px  (stroke/cap ~0.07)    -> a THIN face, not bold
#   baseline         y ~40        cap top  y ~10
BASELINE = 40.0
START_X = 6.0
GAP = 1.0               # px between one glyph's ink and the next (advance = width+gap)
FONT_PX = 68            # em-size for the initial per-glyph render (before bbox-pin)
ROT_DEG = 15.0          # per-character rotation range (+/-)
# Geometry is PINNED, not left to the font (the same choice the GST reader makes):
# after laying the glyphs out with proportional advance, the whole word's ink bbox
# is scaled to the measured real target below. That makes the synthetic word the
# right size and shape regardless of DejaVu's native proportions, which are wider
# and heavier than the real (unidentified) thin face. Measured on the real corpus:
#   ink bbox ~116 x 31, baseline ~y40, stroke ~2.2px, inkpx ~410.
TARGET_W = 116.0        # real ink-bbox width  (jittered per image)
TARGET_H = 31.0         # real ink-bbox height
THIN = 2                # light erosion toward the measured stroke width

# Measured light-blue noise-line palette (the brightest, most saturated blues
# seen in the real images); speckle is scattered saturated dots.
_LINE_BLUES = [(151, 206, 252), (155, 215, 241), (158, 196, 252),
               (148, 193, 253), (166, 230, 238), (167, 190, 244)]

_HERE = os.path.dirname(os.path.abspath(__file__))
_FONTS_DIR = os.path.join(os.path.dirname(_HERE), "fonts")
# The real face is one thin, narrow sans that could not be identified (as with
# MCA). DejaVu Sans Condensed is the narrowest bundled face and the closest in
# proportion; geometry is then PINNED to the measured real bbox (below), so the
# exact face matters less than it would for a reader that trusted the font's
# metrics. The residual letterform/weight gap is closed by the real-label
# fine-tune, not by adding more guessed faces here.
_FACES = ["DejaVuSansCondensed.ttf"]
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
# preprocessing — a darkness projection, identical for real and synthetic
# --------------------------------------------------------------------------
def darkness(rgb):
    """(H,W,3)|PIL -> float32 (H,W) in [0,1]: 1 where black ink, 0 where white
    or light-blue noise. Collapses the colour noise the model must ignore."""
    a = np.asarray(rgb, dtype=np.float32)
    return 1.0 - a.max(2) / 255.0


def _to_input(rgb):
    """RGB image -> (IN_H, IN_W) uint8, ink bright on dark."""
    d = darkness(rgb)
    l, t, r, b = CROP
    sub = (d[t:b, l:r] * 255.0).astype(np.uint8)
    im = Image.fromarray(sub)
    return np.asarray(im.resize((IN_W, IN_H), Image.BILINEAR), dtype=np.uint8)


def load_real(path_or_bytes):
    if isinstance(path_or_bytes, (bytes, bytearray)):
        im = Image.open(io.BytesIO(path_or_bytes))
    elif isinstance(path_or_bytes, Image.Image):
        im = path_or_bytes
    else:
        im = Image.open(path_or_bytes)
    return _to_input(im.convert("RGB"))


# --------------------------------------------------------------------------
# synthetic generation — render the full RGB (text, then noise over it) so the
# darkness projection sees the same fragmentation the real one does
# --------------------------------------------------------------------------
# Everything is drawn on a 2x supersampled canvas and downscaled at the end;
# that gives sub-pixel control over both rotation and stroke erosion (a 1px
# MinFilter at 2x thins by ~0.5px effective), which native-resolution filtering
# is too coarse to provide. Securimage's renderer samples from an upscaled
# canvas for the same reason.
SS = 2


def _glyph_alpha(ch, fnt, angle):
    """A rotated glyph as an L (0..255 ink coverage) layer, tight to its ink,
    rendered on the SSx supersampled grid. Returns (layer, ink_w, ink_h) in SS px,
    where ink_w/ink_h are the UNrotated glyph ink size (used for advance)."""
    n = FONT_PX * SS
    probe = Image.new("L", (n * 3, n * 3), 0)
    ImageDraw.Draw(probe).text((n, n * 2), ch, font=fnt, fill=255, anchor="ls")
    a = np.asarray(probe)
    ys, xs = np.where(a > 40)
    if len(xs) == 0:
        return None, 0, 0
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    glyph = probe.crop((x0, y0, x1 + 1, y1 + 1))
    rot = glyph.rotate(angle, expand=True, resample=Image.BICUBIC)
    return rot, (x1 - x0 + 1), (y1 - y0 + 1)


def make_ink(label, rng):
    """Render the label to a native-size L ink-coverage layer (255=ink), with the
    word's ink bbox pinned to the measured real geometry."""
    # lay glyphs out on a wide SS canvas with proportional advance and per-glyph
    # rotation + vertical jitter; exact placement is provisional — the bbox is
    # re-pinned afterwards, so only the RELATIVE layout matters here.
    cw, ch_ = 360 * SS, H * SS
    big = Image.new("L", (cw, ch_), 0)
    path = FONT_FILES[int(rng.integers(0, len(FONT_FILES)))]
    fnt = _font(path, FONT_PX * SS)
    x = 12 * SS
    midy = ch_ / 2.0
    for ch in label:
        # the label class is uppercase, but the captcha renders mixed case; render
        # each letter as lower- or upper-case at random so the case-insensitive
        # model learns BOTH glyph shapes map to the one class. Digits are unchanged.
        glyph = ch.lower() if (ch.isalpha() and rng.random() < 0.5) else ch
        angle = float(rng.uniform(-ROT_DEG, ROT_DEG))
        layer, gw, gh = _glyph_alpha(glyph, fnt, angle)
        if layer is None:
            x += 12 * SS
            continue
        cy = midy + float(rng.uniform(-3.0, 3.0)) * SS
        px = int(round(x - (layer.width - gw) / 2.0))
        py = int(round(cy - layer.height / 2.0))
        cur = big.crop((px, py, px + layer.width, py + layer.height))
        big.paste(ImageChops.lighter(cur, layer), (px, py))
        x += gw + (GAP + float(rng.uniform(-1.0, 2.0))) * SS
    for _ in range(THIN):
        big = big.filter(ImageFilter.MinFilter(3))

    # crop to ink and pin the bbox to the measured real target (with jitter)
    a = np.asarray(big)
    ys, xs = np.where(a > 40)
    if len(xs) == 0:
        return Image.new("L", (W, H), 0)
    word = big.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    tw = int(round(TARGET_W * (1.0 + rng.normal(0, 0.03))))
    th = int(round(TARGET_H * (1.0 + rng.normal(0, 0.03))))
    word = word.resize((max(tw, 1), max(th, 1)), Image.LANCZOS)

    out = Image.new("L", (W, H), 0)
    ox = int(round(START_X + rng.uniform(-2, 2)))
    oy = int(round(BASELINE - th + rng.uniform(-2, 2)))     # baseline ~y40
    out.paste(word, (ox, oy))
    return out


def make_rgb(label, rng):
    """Render one synthetic ITAT captcha as an RGB image (PIL)."""
    ink = np.asarray(make_ink(label, rng), dtype=np.float32) / 255.0   # (H,W)
    base = (255.0 * (1.0 - ink))[..., None] * np.ones(3, np.float32)    # black on white
    canvas = Image.fromarray(base.astype(np.uint8))
    _draw_noise(canvas, rng)
    return canvas


def _draw_noise(canvas, rng):
    """Light-blue straight lines + colour speckle, drawn OVER the text."""
    d = ImageDraw.Draw(canvas, "RGBA")
    for _ in range(int(rng.integers(16, 30))):
        col = _LINE_BLUES[int(rng.integers(0, len(_LINE_BLUES)))]
        a = int(rng.integers(150, 245))       # partial opacity -> some cut-through
        x0 = float(rng.uniform(-10, W + 10)); y0 = float(rng.uniform(0, H))
        x1 = float(rng.uniform(-10, W + 10)); y1 = float(rng.uniform(0, H))
        d.line([(x0, y0), (x1, y1)], fill=(*col, a), width=1)
    px = canvas.load()
    for _ in range(int(rng.integers(80, 200))):   # scattered speckle
        sx = int(rng.integers(0, W)); sy = int(rng.integers(0, H))
        if rng.random() < 0.5:
            px[sx, sy] = (int(rng.integers(120, 200)),) * 3    # grey speck
        else:
            px[sx, sy] = (int(rng.integers(120, 255)), int(rng.integers(120, 255)),
                          int(rng.integers(150, 255)))


def make_input(label, rng):
    return _to_input(make_rgb(label, rng))


# --------------------------------------------------------------------------
# model — single-channel CRNN+CTC over the darkness map (same shape as MCA's)
# --------------------------------------------------------------------------
class ItatCRNN(nn.Module):
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
