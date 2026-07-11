"""Securimage-style captcha support (eCourts portal): 215x80 grayscale,
6 lowercase-alphanumeric chars, heavy sinusoidal wave distortion + curvy
noise lines. Recognition is whole-image (no segmentation): a CNN reads the
full image and predicts all 6 characters via 6 classification heads.
"""
import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
# NOTE: scipy is only needed to *generate* synthetic training data (the 3-pole
# warp in _distort). Inference (load_real + predict_*) never imports it, so a
# production deploy only needs torch, numpy, pillow. The import is lazy below.

# Observed charset: a-z + 2-9 (0/1 never appear; commonly excluded as confusable).
CHARSET = "abcdefghijklmnopqrstuvwxyz23456789"
CH2I = {c: i for i, c in enumerate(CHARSET)}
I2CH = {i: c for i, c in enumerate(CHARSET)}
N_CLASSES = len(CHARSET)      # 34
LENGTH = 6
GEN_W, GEN_H = 215, 80        # render/native size
IN_W, IN_H = 192, 64          # model input: 64px tall preserves descenders (y/p/g)
                              # and stacked-loop detail (6/8, a/2); width 192 -> T=24

# Securimage's actual font is AHGBold (Alte Haas Grotesk Bold) — confirmed by
# matching letterforms against the live eCourts captcha. Train on the real font.
_AHGBOLD = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fonts", "AHGBold.ttf")
_FALLBACK = [f for f in [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Verdana Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
] if os.path.exists(f)]
_FONT_FILES = ([_AHGBOLD] if os.path.exists(_AHGBOLD) else _FALLBACK)

_BOLD_FONTS = _FONT_FILES

_font_cache = {}


def _font(path, size):
    key = (path, size)
    if key not in _font_cache:
        _font_cache[key] = ImageFont.truetype(path, size)
    return _font_cache[key]


def _pick_font_path(rng):
    # the real captcha is uniformly bold, so bias strongly toward bold faces
    pool = _BOLD_FONTS if rng.random() < 0.75 else _FONT_FILES
    return rng.choice(pool)


def random_label(rng):
    return "".join(CHARSET[i] for i in rng.integers(0, N_CLASSES, size=LENGTH))


# --- synthetic rendering (faithful port of Securimage's GD algorithm) ----------
# Securimage renders text upright and tightly packed onto an iscale-times-larger
# canvas, then warps it into the final image with a 3-pole radial distortion
# field, and finally overlays 5 sine-curved lines + noise arcs. Colours/opacity
# are the eCourts deployment's measured values:
#   text ink = flat gray50 (black at 20% transparency over white, opaque copy)
#   lines/noise = gray111 (#707070) at alpha 0.8 (semi-transparent, drawn over)
# The semi-transparency is critical: a line over text blends to ~99 (not an
# opaque 112 block), so descenders (y/p/g) remain readable *through* the line
# instead of being erased — which is exactly how the real captcha behaves.
ISCALE = 3                     # high-res render scale for the text (GD default 2)
INK = 50                       # measured text ink (gray50 / #323232)
LINE_COL = 111                 # measured line/noise colour (#707070) before alpha
LINE_ALPHA = 0.8               # measured opacity: over-white->140, over-text->99
_RATIO = 0.38                  # font_size / image_height (tuned to real ~31px cap)


def frand(rng):
    return float(rng.random())


def _render_text_hires(label, rng):
    """Draw the word upright + tightly packed on the iscale canvas (ink on white)."""
    TW, TH = GEN_W * ISCALE, GEN_H * ISCALE
    img = Image.new("L", (TW, TH), 255)
    d = ImageDraw.Draw(img)
    font_path = _pick_font_path(rng)
    size = int(round(GEN_H * ISCALE * _RATIO / 0.7))   # PIL em-size for ~ratio cap height
    font = _font(font_path, size)
    asc, desc = font.getmetrics()

    # per-char advances with Securimage's tight negative spacing dist=rand(-2,0)*scale
    widths, dists = [], []
    for ch in label:
        widths.append(font.getlength(ch))
        dists.append(int(rng.integers(-2, 1)) * ISCALE)   # mt_rand(-2,0)
    total = sum(w + dm for w, dm in zip(widths, dists))

    baseline = TH / 2 + asc / 2 - desc / 2                # vertically centred text block
    cx = TW / 2 - total / 2
    x = float(rng.integers(int(5 * ISCALE), max(int(cx * 2 - 5 * ISCALE), int(5 * ISCALE)) + 1))
    for ch, w, dm in zip(label, widths, dists):
        d.text((x, baseline), ch, font=font, fill=INK, anchor="ls")
        x += w + dm
    return np.asarray(img, np.float64)


def _distort(hires, rng):
    """3-pole radial warp (Securimage::distortedCopy), output GEN_H x GEN_W.

    The displacement field is computed in output-space (215x80); each output
    pixel is pulled toward up to 3 random poles, then sampled from the iscale
    hi-res text canvas."""
    W, H = GEN_W, GEN_H
    pert = rng.uniform(0.75, 0.85)                        # Securimage perturbation
    x_lo = W / 4.0
    maxX = int(W - x_lo)                                  # PHP integer modulo operands
    dx0 = int(rng.integers(int(x_lo / 10), int(x_lo) + 1))
    y0 = int(rng.integers(20, H - 20 + 1))
    dy0 = int(rng.integers(20, int(round(H * 0.7)) + 1))
    minY, maxY = 20, H - 20
    px, py, rad, amp = [], [], [], []
    for i in range(3):
        px.append(int(x_lo + dx0 * i) % maxX)
        py.append(int(y0 + dy0 * i) % maxY + minY)
        rad.append(float(rng.integers(int(H * 0.4), int(H * 0.8) + 1)))
        amp.append(pert * ((-frand(rng)) * 0.15 - 0.15))

    iy, ix = np.mgrid[0:H, 0:W].astype(np.float64)
    X = ix.copy()
    Y = iy.copy()
    for i in range(3):
        ddx = ix - px[i]
        ddy = iy - py[i]
        r = np.sqrt(ddx * ddx + ddy * ddy)
        with np.errstate(invalid="ignore"):
            rscale = amp[i] * np.sin(3.14 * r / rad[i])
        m = (r <= rad[i]) & (r > 0)
        X += np.where(m, ddx * rscale, 0.0)
        Y += np.where(m, ddy * rscale, 0.0)
    # sample the hi-res text canvas at (Y, X) * iscale
    from scipy.ndimage import map_coordinates       # generation-only dependency
    out = map_coordinates(hires, [Y * ISCALE, X * ISCALE], order=1,
                          mode="constant", cval=255.0)
    return out


def _noise_coverage(rng):
    """Securimage::drawNoise coverage mask — sparse small pie arcs (0..1)."""
    W, H = GEN_W, GEN_H
    m = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(m)
    for x in range(1, W, 20):
        for y in range(1, H, 20):
            if rng.random() > 0.5:                       # sparse: ~half the cells
                continue
            x1 = int(rng.integers(x, x + 21))
            y1 = int(rng.integers(y, y + 21))
            s = int(rng.integers(1, 3))                  # radius 1-2px
            if x1 - s <= 0 and y1 - s <= 0:
                continue
            a1 = int(rng.integers(180, 361))
            d.pieslice([x1 - s, y1 - s, x1 + s, y1 + s], 0, a1, fill=255)
    return np.asarray(m, np.float64) / 255.0


def _lines_coverage(rng):
    """Securimage::drawLines coverage mask — sine-perturbed strokes (0..1)."""
    W, H = GEN_W, GEN_H
    m = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(m)
    num = 5
    for line in range(num):
        x = W * (1 + line) / (num + 1)
        x += (0.5 - frand(rng)) * W / num
        y = float(rng.integers(int(H * 0.1), int(H * 0.9) + 1))
        theta = (frand(rng) - 0.5) * np.pi * 0.33
        length = float(rng.integers(int(W * 0.4), int(W * 0.7) + 1))
        lwid = int(rng.integers(0, 2))                   # thin strokes (measured real)
        k = frand(rng) * 0.6 + 0.2
        k = k * k * 0.5
        phi = frand(rng) * 6.28
        step = 0.5
        dx, dy = step * np.cos(theta), step * np.sin(theta)
        n = int(length / step)
        amp = 1.5 * frand(rng) / (k + 5.0 / length)
        x0 = x - 0.5 * length * np.cos(theta)
        y0 = y - 0.5 * length * np.sin(theta)
        for i in range(n):
            xi = x0 + i * dx + amp * dy * np.sin(k * i * step + phi)
            yi = y0 + i * dy - amp * dx * np.sin(k * i * step + phi)
            d.rectangle([round(xi), round(yi), round(xi + lwid), round(yi + lwid)], fill=255)
    return np.asarray(m, np.float64) / 255.0


def _blend(base, cov):
    """Alpha-composite gray LINE_COL over `base` with the measured 0.8 opacity."""
    return base * (1 - LINE_ALPHA * cov) + LINE_COL * (LINE_ALPHA * cov)


def make_image(label, rng):
    """Full synthetic securimage captcha -> IN_H x IN_W uint8 grayscale.

    Faithful layer order: sparse noise arcs (unwarped) -> upright text warped by
    the 3-pole field, opaque over the noise -> semi-transparent sine lines over
    the top (so descenders read *through* the lines). Rendered at native 215x80
    then resized to the model input, exactly like load_real."""
    base = np.full((GEN_H, GEN_W), 255.0)
    base = _blend(base, _noise_coverage(rng))              # noise arcs (alpha 0.8)

    hires = _render_text_hires(label, rng)
    warped = _distort(hires, rng)                          # text only, warped
    base = np.minimum(base, warped)                        # ink (dark) opaque over noise

    base = _blend(base, _lines_coverage(rng))              # lines (alpha 0.8) on top
    out = Image.fromarray(base.astype(np.uint8)).resize((IN_W, IN_H), Image.BILINEAR)
    return np.asarray(out, np.uint8)


def load_real(path):
    """Load a real captcha PNG -> IN_H x IN_W uint8 grayscale (model input)."""
    im = Image.open(path).convert("L").resize((IN_W, IN_H), Image.BILINEAR)
    return np.asarray(im, np.uint8)


def encode_label(label):
    return np.array([CH2I[c] for c in label], np.int64)


def decode(pred_indices):
    return "".join(I2CH[int(i)] for i in pred_indices)


# --- model ---------------------------------------------------------------------
class SecurimageNet(nn.Module):
    def __init__(self, n_classes=N_CLASSES, n_pos=LENGTH):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(),
                nn.MaxPool2d(2),
            )
        self.features = nn.Sequential(
            block(1, 48),    # /2
            block(48, 96),   # /4
            block(96, 192),  # /8
            block(192, 256),  # /16
        )
        with torch.no_grad():
            flat = self.features(torch.zeros(1, 1, IN_H, IN_W)).flatten(1).shape[1]
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(flat, 512), nn.ReLU(), nn.Dropout(0.3))
        self.heads = nn.ModuleList([nn.Linear(512, n_classes) for _ in range(n_pos)])

    def forward(self, x):
        z = self.fc(self.features(x))
        return [h(z) for h in self.heads]   # list of [B, n_classes]


def predict(model, imgs_uint8, device="cpu"):
    """imgs_uint8: (N, IN_H, IN_W) -> (list_of_strings, list_of_min_confidence)."""
    X = torch.tensor(imgs_uint8[:, None].astype(np.float32) / 255.0, device=device)
    model.eval()
    with torch.no_grad():
        logits = model(X)                       # 6 x [N, C]
        probs = [torch.softmax(l, 1) for l in logits]
    N = X.shape[0]
    out, confs = [], []
    for n in range(N):
        chars, cs = [], []
        for pos in range(LENGTH):
            p = probs[pos][n]
            i = int(p.argmax())
            chars.append(I2CH[i]); cs.append(float(p[i]))
        out.append("".join(chars)); confs.append(min(cs))
    return out, confs


# --- CRNN + CTC model (translation-invariant; generalizes to unseen images) ----
BLANK = N_CLASSES                # CTC blank index (classes 0..33 are chars, 34 is blank)


class SecurimageCRNN(nn.Module):
    """CNN -> collapse height -> width sequence -> BiLSTM -> per-step class logits.

    Trained with CTC so it never assumes a character sits at a fixed position;
    the same recurrent classifier slides across the image width, which is what
    lets it generalize from synthetic to real distorted captchas (unlike the
    flatten+fixed-head SecurimageNet, which only memorizes)."""

    def __init__(self, n_classes=N_CLASSES):
        super().__init__()

        def cb(cin, cout, pool):
            return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1),
                                 nn.BatchNorm2d(cout), nn.ReLU(), nn.MaxPool2d(pool))
        # input 64 x 192; collapse height 64 -> 1 over six /2 pools, width 192 -> 24
        self.cnn = nn.Sequential(
            cb(1, 64, (2, 2)),     # 32 x 96
            cb(64, 128, (2, 2)),   # 16 x 48
            cb(128, 256, (2, 2)),  # 8 x 24
            cb(256, 256, (2, 1)),  # 4 x 24
            cb(256, 384, (2, 1)),  # 2 x 24
            cb(384, 384, (2, 1)),  # 1 x 24  (T = 24)
        )
        self.rnn = nn.LSTM(384, 256, num_layers=2, bidirectional=True, batch_first=True, dropout=0.2)
        self.head = nn.Linear(512, n_classes + 1)   # +1 for CTC blank

    def forward(self, x):
        f = self.cnn(x)                    # [B, 384, 1, T]
        f = f.squeeze(2).permute(0, 2, 1)  # [B, T, 384]
        f, _ = self.rnn(f)                 # [B, T, 512]
        return self.head(f)                # [B, T, n_classes+1]


def ctc_greedy_decode(logits_btc):
    """logits_btc: [B, T, C] -> (list_of_strings, list_of_min_confidence)."""
    probs = torch.softmax(logits_btc, dim=2)
    conf, idx = probs.max(dim=2)           # [B,T]
    idx = idx.cpu().numpy(); conf = conf.cpu().numpy()
    out, confs = [], []
    for b in range(idx.shape[0]):
        chars, cs, prev = [], [], -1
        for t in range(idx.shape[1]):
            c = int(idx[b, t])
            if c != prev and c != BLANK:
                chars.append(I2CH[c]); cs.append(float(conf[b, t]))
            prev = c
        out.append("".join(chars))
        confs.append(min(cs) if cs else 0.0)
    return out, confs


def predict_ctc(model, imgs_uint8, device="cpu"):
    X = torch.tensor(imgs_uint8[:, None].astype(np.float32) / 255.0, device=device)
    model.eval()
    with torch.no_grad():
        logits = model(X)
    return ctc_greedy_decode(logits)


def ctc_beam_decode(log_probs_tc, beam_width=25, force_len=LENGTH):
    """Prefix beam search over [T, C] log-probs; returns (string, confidence).

    Correctly separates adjacent repeats (e.g. 'uu') via the blank probability,
    and — because the captcha is always LENGTH characters — restricts the final
    answer to prefixes of exactly `force_len` when any exist (fixes the greedy
    decoder's dominant dropped-character error)."""
    from collections import defaultdict
    lp = log_probs_tc.detach().cpu().numpy() if hasattr(log_probs_tc, "detach") else np.asarray(log_probs_tc)
    p = np.exp(lp)
    T, C = p.shape
    beams = {(): [1.0, 0.0]}                      # prefix -> [p_blank, p_nonblank]
    for t in range(T):
        pt = p[t]
        nxt = defaultdict(lambda: [0.0, 0.0])
        for prefix, (pb, pnb) in beams.items():
            ptot = pb + pnb
            nxt[prefix][0] += ptot * pt[BLANK]                 # extend with blank
            if prefix:
                nxt[prefix][1] += pnb * pt[prefix[-1]]         # merge same char
            for c in range(C):
                if c == BLANK:
                    continue
                if prefix and c == prefix[-1]:
                    nxt[prefix + (c,)][1] += pb * pt[c]        # repeat needs a blank before
                else:
                    nxt[prefix + (c,)][1] += ptot * pt[c]
        beams = dict(sorted(nxt.items(), key=lambda kv: kv[1][0] + kv[1][1], reverse=True)[:beam_width])
        s = sum(v[0] + v[1] for v in beams.values())          # renormalize to avoid underflow
        if s > 0:
            for k in beams:
                beams[k][0] /= s; beams[k][1] /= s
    scored = [(pref, pb + pnb) for pref, (pb, pnb) in beams.items() if pref]
    if force_len is not None:
        cand = [(pref, sc) for pref, sc in scored if len(pref) == force_len]
        if cand:
            scored = cand
    pref, sc = max(scored, key=lambda x: x[1])
    conf = float(sc ** (1.0 / max(len(pref), 1)))             # geometric-mean per-char confidence
    return "".join(I2CH[c] for c in pref), conf


def predict_ctc_beam(model, imgs_uint8, device="cpu", beam_width=25, force_len=LENGTH):
    X = torch.tensor(imgs_uint8[:, None].astype(np.float32) / 255.0, device=device)
    model.eval()
    with torch.no_grad():
        lps = model(X).log_softmax(2)
    out, confs = [], []
    for b in range(X.shape[0]):
        s, c = ctc_beam_decode(lps[b], beam_width, force_len)
        out.append(s); confs.append(c)
    return out, confs
