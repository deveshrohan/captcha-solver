"""Bharatkosh / NTRP captcha support (bharatkosh.gov.in): 150x40 RGBA, 6
mixed-case alnum chars, read as a VOTE across re-renders of one answer.

Composition (measured over the real corpus, see the README and
docs/superpowers/specs/2026-09-19-bharatkosh-captcha-design.md):

* the stack is **ASP.NET MVC + GDI+** (`System.Drawing`): the PNG carries
  `sRGB gAMA pHYs` chunks and zlib `78 5e`, and is served under a wrong
  `Content-Type: image/gif`;
* every glyph is drawn independently -- its own typeface, size, rotation (up to
  ~+/-30 deg) and two-colour gradient fill -- antialiased, neighbours overlapping.
  The same character in the same answer renders as visibly different shapes, so
  there is no bitmap library to cover with (the Kaveri/Udyam/NGT route is closed,
  for ITAT's reason and then some);
* ~340 pastel specks per image (min channel ~150-235), mostly isolated pixels;
* **two opaque pastel bars at rows 14-16 and 24-26, x=1..149, in every image**,
  drawn LAST: each bar row is 149/149 bar colour, nothing survives under them.

The bars are the only destructive noise and their position is a constant. So the
input ZEROES those six rows for real and synthetic alike: the model sees one
consistent blind band instead of hundreds of bar colours, and there is nothing to
learn about bars at all. What the band costs is real -- a rotated `A` lifts one
foot into the lower bar and what remains is a `4` -- and no single image can get
that back.

Several images can. `GenerateCaptcha?New=0` re-renders the session's SAME text
with a fresh draw of every random variable, so the unit of reading is a group of
K renders. `vote` scores each candidate word by its summed CTC log-likelihood
over all renders: a glyph the bar ruined in one render is whole in the next, and
because the draws are independent the errors are too.
"""
import io
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

# Case-sensitive alnum MINUS `0 1 I O i l o w` -- 54 classes. The exclusions
# are measured, not assumed, and by two independent routes:
#
# * the synthetic-only model's voted reads of 297 harvested groups (1782 chars,
#   ~29 expected per class) contain all eight exactly ONCE between them (one
#   `o`). A model can mislabel a class but cannot hide it: a real `O` has to
#   come out as SOMETHING, yet `0`, `O` and `o` sum to 1 against ~86 expected,
#   and `1 I i l` to 0 against ~115. `W` sits at 33 alone, i.e. `w` is not
#   being folded into it;
# * the 198 hand-read characters contain none of them either (after `b0005`,
#   whose one `O` the census flagged, was re-read and moved to ambiguous);
#   P(no draw from these 8 in 198 uniform draws over 62) = 1.3e-12.
#
# It is the classic confusable-glyph blocklist plus `w` (~ `W` once size is
# random per glyph). Carrying the impossible classes was not harmless: they
# were the top single-render confusions (`j->i` 14, `p->o` 13, `M->I` 3).
# Case is kept -- every other case pair appears in both cases at ~uniform
# rates -- and eval reports case-sensitive and folded scores, because whether
# the portal validates case could not be checked without posting a form.
CHARSET = ("23456789"
           "ABCDEFGHJKLMNPQRSTUVWXYZ"
           "abcdefghjkmnpqrstuvxyz")
EXCLUDED = "01IOilow"
CH2I = {c: i for i, c in enumerate(CHARSET)}
I2CH = {i: c for i, c in enumerate(CHARSET)}
N_CLASSES = len(CHARSET)
BLANK = N_CLASSES
LENGTH = 6

W, H = 150, 40
# half-open row ranges of the two bars: 200/200 images, never anywhere else
BAR_ROWS = ((14, 17), (24, 27))
# 1.6x upscale: rotation makes the fine distinctions finer (Securimage lesson)
IN_W, IN_H = 240, 64

# --- geometry, measured on the real corpus (components >= 15 px, bars excluded) ---
#   ink x0   median 14 (p2 11, p98 16)   -> the first glyph starts at a FIXED x
#   ink x1   median 128 (p2 114, p98 142)
#   ink y    7 .. 33 median; bbox height 22 / 26 / 31 (p2 / p50 / p98)
#   ink px   median 762 (outside the bars)
#   ink min-channel  p10 59, p50 106    -> mid-dark, saturated fills
#   speck min-channel p10 151, p50 170, p90 207; ~340 speck px per image
START_X = 14.0
PITCH = 19.0            # (128 - 14) / 6
PITCH_JIT = 3.0
MID_Y = 20.0
Y_JIT = 3.0
FONT_PX = (24, 30)      # per-glyph em size; (22,32) spread ink 1.9x too wide
ROT_DEG = 28.0
N_SPECK = (300, 380)
SS = 3

_HERE = os.path.dirname(os.path.abspath(__file__))
_FONTS_DIR = os.path.join(os.path.dirname(_HERE), "fonts")
# The server is Windows, so the faces are Windows core fonts; sans, serif and
# rounded faces all appear inside single images. macOS ships the same files, and
# they are loaded by path (training only -- inference never touches a font) and
# never copied into the repo. This is a shortlist, not an identification.
_SYS = "/System/Library/Fonts/Supplemental"
# Weighted toward bold: real strokes run ~3px, and with the regular serif faces
# in the list synthetic ink came out at 630 px/image against a real 833.
_FACES = ["Arial.ttf", "Arial Bold.ttf", "Arial Bold.ttf", "Verdana.ttf",
          "Verdana Bold.ttf", "Verdana Bold.ttf", "Tahoma.ttf", "Tahoma Bold.ttf",
          "Times New Roman Bold.ttf", "Georgia Bold.ttf", "Trebuchet MS.ttf",
          "Trebuchet MS Bold.ttf", "Comic Sans MS Bold.ttf", "Arial Black.ttf"]
FONT_FILES = [os.path.join(_SYS, f) for f in _FACES
              if os.path.exists(os.path.join(_SYS, f))]
if not FONT_FILES:                          # non-mac fallback: bundled DejaVu
    FONT_FILES = [os.path.join(_FONTS_DIR, f) for f in
                  ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf")]

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
# preprocessing -- identical for real and synthetic
# --------------------------------------------------------------------------
def ink_map(rgb):
    """(H,W,3) -> float32 (H,W) in [0,1]. `1 - min(r,g,b)/255`: white -> 0, a
    saturated glyph fill -> high, a pastel speck -> low. There is no ink COLOUR
    to key on (~500 colours per image), but every fill is far from white in at
    least one channel. Bar rows are zeroed; pixels with no 8-neighbour go too."""
    a = np.asarray(rgb, dtype=np.float32)[..., :3]
    d = 1.0 - a.min(2) / 255.0
    for y0, y1 in BAR_ROWS:
        d[y0:y1] = 0.0
    on = d > 0
    p = np.pad(on, 1)
    nb = sum(p[1 + dy:1 + dy + on.shape[0], 1 + dx:1 + dx + on.shape[1]]
             for dy in (-1, 0, 1) for dx in (-1, 0, 1)) - on
    d[on & (nb == 0)] = 0.0
    return d


def _to_input(rgb):
    """RGB image -> (IN_H, IN_W) uint8, ink bright on dark."""
    im = Image.fromarray((ink_map(rgb) * 255.0).astype(np.uint8))
    return np.asarray(im.resize((IN_W, IN_H), Image.BILINEAR), dtype=np.uint8)


def load_real(path_or_bytes):
    if isinstance(path_or_bytes, (bytes, bytearray)):
        im = Image.open(io.BytesIO(path_or_bytes))
    elif isinstance(path_or_bytes, Image.Image):
        im = path_or_bytes
    else:
        im = Image.open(path_or_bytes)
    if im.size != (W, H):
        raise ValueError("bharatkosh captcha must be %dx%d, got %dx%d"
                         % ((W, H) + im.size))
    return _to_input(im.convert("RGB"))


# --------------------------------------------------------------------------
# synthetic generation -- specks, then glyphs, then the bars LAST
# --------------------------------------------------------------------------
def _fill_colour(rng):
    return rng.integers(40, 215, size=3).astype(np.float32)


def _glyph(ch, rng):
    """One glyph as (RGB float array, alpha L image) on the SS grid: own face,
    size, rotation and two-colour linear gradient."""
    fnt = _font(FONT_FILES[int(rng.integers(0, len(FONT_FILES)))],
                int(rng.integers(FONT_PX[0], FONT_PX[1] + 1)) * SS)
    n = FONT_PX[1] * SS
    probe = Image.new("L", (n * 3, n * 3), 0)
    ImageDraw.Draw(probe).text((n, n * 2), ch, font=fnt, fill=255, anchor="ls")
    box = probe.getbbox()
    if box is None:
        return None, None
    alpha = probe.crop(box).rotate(float(rng.uniform(-ROT_DEG, ROT_DEG)),
                                   expand=True, resample=Image.BICUBIC)
    h, w = alpha.height, alpha.width
    c0, c1 = _fill_colour(rng), _fill_colour(rng)
    th = float(rng.uniform(0, 2 * np.pi))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    t = xx * np.cos(th) + yy * np.sin(th)
    t = (t - t.min()) / max(float(t.max() - t.min()), 1.0)
    return c0 + (c1 - c0) * t[..., None], alpha


def make_rgb(label, rng):
    """Render one synthetic Bharatkosh captcha as an RGB PIL image."""
    # specks go UNDER the text: real strokes are clean, never pitted
    small = np.full((H, W, 3), 255, np.uint8)
    n = int(rng.integers(N_SPECK[0], N_SPECK[1] + 1))
    small[rng.integers(0, H, n), rng.integers(0, W, n)] = \
        rng.integers(150, 256, size=(n, 3))
    big = small.repeat(SS, 0).repeat(SS, 1).astype(np.float32)
    for i, ch in enumerate(label):
        rgb, alpha = _glyph(ch, rng)
        if rgb is None:
            continue
        cx = (START_X + PITCH * (i + 0.5) + rng.uniform(-PITCH_JIT, PITCH_JIT)) * SS
        cy = (MID_Y + rng.uniform(-Y_JIT, Y_JIT)) * SS
        x0 = int(round(cx - alpha.width / 2.0))
        y0 = int(round(cy - alpha.height / 2.0))
        a = np.asarray(alpha, dtype=np.float32) / 255.0
        # clip the glyph to the canvas
        sx0, sy0 = max(0, -x0), max(0, -y0)
        dx0, dy0 = max(0, x0), max(0, y0)
        w = min(a.shape[1] - sx0, big.shape[1] - dx0)
        h = min(a.shape[0] - sy0, big.shape[0] - dy0)
        if w <= 0 or h <= 0:
            continue
        m = a[sy0:sy0 + h, sx0:sx0 + w, None]
        dst = big[dy0:dy0 + h, dx0:dx0 + w]
        dst[:] = dst * (1 - m) + rgb[sy0:sy0 + h, sx0:sx0 + w] * m
    im = Image.fromarray(big.clip(0, 255).astype(np.uint8)).resize(
        (W, H), Image.LANCZOS)
    a = np.array(im)
    for y0, y1 in BAR_ROWS:
        a[y0:y1, 1:150] = rng.integers(150, 256, size=3)
    return Image.fromarray(a)


def make_input(label, rng):
    return _to_input(make_rgb(label, rng))


# --------------------------------------------------------------------------
# model -- ITAT's single-channel CRNN+CTC, 62 classes, 30 timesteps
# --------------------------------------------------------------------------
class BharatkoshCRNN(nn.Module):
    def __init__(self, n_classes=N_CLASSES):
        super().__init__()

        def cb(cin, cout, pool):
            return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1),
                                 nn.BatchNorm2d(cout), nn.ReLU(), nn.MaxPool2d(pool))

        self.cnn = nn.Sequential(
            cb(1, 64, (2, 2)),      # 32 x 120
            cb(64, 128, (2, 2)),    # 16 x 60
            cb(128, 256, (2, 2)),   # 8 x 30
            cb(256, 256, (2, 1)),   # 4 x 30
            cb(256, 384, (2, 1)),   # 2 x 30
            cb(384, 384, (2, 1)),   # 1 x 30
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


def log_probs(model, imgs, device="cpu"):
    """(N, IN_H, IN_W) uint8 -> (N, T, C) CTC log-probs on the CPU."""
    model.eval()
    with torch.no_grad():
        return model(_to_tensor(imgs, device)).log_softmax(2).cpu()


def greedy(lps_ntc):
    """(N,T,C) log-probs -> best-path strings. Fast; for training-time scoring."""
    out = []
    for row in np.asarray(lps_ntc).argmax(2):
        chars, prev = [], -1
        for c in row:
            if c != prev and c != BLANK:
                chars.append(I2CH[int(c)])
            prev = c
        out.append("".join(chars))
    return out


def beam(log_probs_tc, beam_width=20, top=5, force_len=LENGTH):
    """CTC prefix beam search -> up to `top` (string, prob) pairs, best first,
    restricted to `force_len` characters when any such candidate exists."""
    from collections import defaultdict
    p = np.exp(np.asarray(log_probs_tc, dtype=np.float64))
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
            if len(prefix) >= (force_len or T):
                continue                      # never grow past the known length
            for c in np.argsort(pt)[-8:]:     # the rest cannot enter the beam
                c = int(c)
                if c == BLANK:
                    continue
                if prefix and c == prefix[-1]:
                    nxt[prefix + (c,)][1] += pb * pt[c]
                else:
                    nxt[prefix + (c,)][1] += ptot * pt[c]
        beams = dict(sorted(nxt.items(), key=lambda kv: kv[1][0] + kv[1][1],
                            reverse=True)[:beam_width])
    scored = [(pref, pb + pnb) for pref, (pb, pnb) in beams.items() if pref]
    if force_len is not None:
        scored = [s for s in scored if len(s[0]) == force_len] or scored
    scored.sort(key=lambda s: s[1], reverse=True)
    return [("".join(I2CH[c] for c in pref), float(sc)) for pref, sc in scored[:top]]


def _ctc_loglik(lp_tc, words):
    """log P(word | render) under CTC, for each word -> (len(words),) tensor."""
    T = lp_tc.shape[0]
    n = len(words)
    lp = lp_tc.float().unsqueeze(1).expand(T, n, lp_tc.shape[1])
    tg = torch.tensor([encode_label(w) for w in words], dtype=torch.long)
    return -F.ctc_loss(lp, tg, torch.full((n,), T, dtype=torch.long),
                       torch.tensor([len(w) for w in words], dtype=torch.long),
                       blank=BLANK, reduction="none", zero_infinity=False)


def vote(lps, beam_width=20, top=5):
    """K renders' (T,C) log-probs of ONE answer -> (word, confidence).

    Candidates are the union of every render's beam; each is scored by the SUM
    over renders of its exact CTC log-likelihood. Summing likelihoods rather
    than majority-voting strings is the point: three renders that are each
    unsure about a DIFFERENT glyph out-vote their own mistakes, because the true
    character is still the runner-up wherever it lost. Confidence is the winner's
    posterior over the candidate set."""
    cands = sorted({w for lp in lps for w, _ in beam(lp, beam_width, top)
                    if len(w) == LENGTH})
    if not cands:
        w = beam(lps[0], beam_width, 1, force_len=None)
        return (w[0][0] if w else ""), 0.0
    total = sum(_ctc_loglik(torch.as_tensor(lp), cands) for lp in lps)
    post = torch.softmax(total.double(), 0)
    i = int(post.argmax())
    return cands[i], float(post[i])


def predict(model, imgs, device="cpu"):
    """Independent single-render reads -> (strings, confidences)."""
    out, confs = [], []
    for lp in log_probs(model, imgs, device):
        w, c = vote([lp])
        out.append(w)
        confs.append(c)
    return out, confs


def predict_group(model, imgs, device="cpu"):
    """All `imgs` are renders of ONE answer -> (word, confidence)."""
    return vote(list(log_probs(model, imgs, device)))
