"""Supreme Court of India (www.sci.gov.in) captcha: securimage-wp in math
mode, `/?_siwp_captcha&id=<scid>` with `scid` taken from the case-status page.

220x80 RGB PNG. A single expression such as `10 - 2`, drawn in grey 87 on
grey 242 with antialiased edges only, so there is no noise to remove. Each
glyph gets its own size, rotation and position. The portal wants the result
(`8`). The expression is stored server-side per `scid`: fetching the image
again re-renders the same expression. The plain page URL is cached and
hands everyone the same `scid`; a throwaway query string gets a fresh one.

Grammar, measured over 123 human-labelled captchas: `left op right` with
left in 1..10, right in 1..5 and op `+` or `-` (never `x`). Results run
from 1 to 14 and are never negative.

No model. The reader thresholds at 165 and splits the ink into 8-connected
pieces, read left to right: left, operator, right. A left `10` comes out as
either two pieces or one touching piece (its own class).

* The operator is decided by structure. A bar's per-column ink counts are
  flat; a cross peaks where its other stroke crosses.
* Each digit is matched only against the classes its position can hold, by
  the mean distance to its 3 nearest exemplars. The exemplars are cut from
  the train split's human-labelled renders, plus copies rotated by +-6,
  +-12 and +-18 degrees (`sci_glyphs.json`, built by `mkglyphs_sci.py`).
  The distance combines the bitmap scaled to 20x20, the box size, and the
  number and height of enclosed holes. The hole measure is what tells 6
  from 9 and 8 from 3.

Confidence is the smallest per-position margin between the best class and
the runner-up. Held out (`sci_eval.json`, 30 captchas harvested after the
method was fixed): 58/60 renders exact. At a 0.6 gate 52 are accepted, all
correct. Voting over the two renders of each captcha (`solve_group`) gets
30/30.
"""
import base64
import io
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

W, H = 220, 80
THRESH = 165
NORM = 20
MIN_PX = 6
OPS = {"+": lambda a, b: a + b, "-": lambda a, b: a - b, "x": lambda a, b: a * b}
_GLYPHS = Path(__file__).resolve().with_name("sci_glyphs.json")


def _open(image):
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, (bytes, bytearray)):
        return Image.open(io.BytesIO(image))
    return Image.open(image)


def ink(image):
    """-> (H, W) bool ink mask, or None for a wrong size."""
    im = _open(image)
    if im.size != (W, H):
        return None
    return np.asarray(im.convert("L")) < THRESH


def components(mask):
    """8-connected ink pieces, left to right -> [(x0, y0, bitmap)]."""
    seen = np.zeros_like(mask, bool)
    out = []
    ys, xs = np.nonzero(mask)
    for y, x in zip(ys.tolist(), xs.tolist()):
        if seen[y, x]:
            continue
        stack, pts = [(y, x)], []
        seen[y, x] = True
        while stack:
            cy, cx = stack.pop()
            pts.append((cy, cx))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1] \
                            and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
        if len(pts) < MIN_PX:
            continue
        p = np.array(pts)
        y0, x0 = p.min(0)
        y1, x1 = p.max(0)
        bm = np.zeros((y1 - y0 + 1, x1 - x0 + 1), bool)
        bm[p[:, 0] - y0, p[:, 1] - x0] = True
        out.append((int(x0), int(y0), bm))
    out.sort(key=lambda c: c[0] + c[2].shape[1] / 2)
    return out


def holes(bm):
    """-> (number of enclosed background regions of >= 4 px, mean height of
    their pixels as a fraction of the box, or -1 with no hole). Rotation
    barely changes either: 8 has two holes, 6 one low, 9 one high."""
    bg = np.pad(~bm, 1, constant_values=True)
    out = np.zeros_like(bg)
    out[0, :] = out[-1, :] = out[:, 0] = out[:, -1] = True
    out &= bg
    while True:                      # grow the outside through 4-neighbours
        grown = out.copy()
        grown[1:] |= out[:-1]
        grown[:-1] |= out[1:]
        grown[:, 1:] |= out[:, :-1]
        grown[:, :-1] |= out[:, 1:]
        grown &= bg
        if (grown == out).all():
            break
        out = grown
    inside = bg & ~out
    if not inside.any():
        return 0, -1.0
    seen, regions = np.zeros_like(inside), []
    for sy, sx in zip(*np.nonzero(inside)):
        if seen[sy, sx]:
            continue
        stack, pts = [(sy, sx)], []
        seen[sy, sx] = True
        while stack:
            y, x = stack.pop()
            pts.append(y)
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if inside[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        if len(pts) >= 4:
            regions.append(pts)
    if not regions:
        return 0, -1.0
    ys = np.concatenate([np.array(r) for r in regions]) - 1
    return len(regions), float(ys.mean() / max(1, bm.shape[0] - 1))


def features(bm):
    """Bitmap scaled into a 20x20 box (aspect kept, centred) + box size."""
    h, w = bm.shape
    s = max(h, w)
    sq = np.zeros((s, s), np.uint8)
    sq[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = bm * 255
    v = np.asarray(Image.fromarray(sq).resize((NORM, NORM), Image.BILINEAR),
                   np.float32) / 255.0
    return v.ravel(), np.array([h, w], np.float32)


ANGLES = (0, -6, 6, -12, 12, -18, 18)      # every glyph gets its own tilt


def pack(bm):
    """bool bitmap -> [h, w, base64 of its packed bits]"""
    return [int(bm.shape[0]), int(bm.shape[1]),
            base64.b64encode(np.packbits(bm.ravel()).tobytes()).decode()]


def unpack(h, w, b64):
    bits = np.unpackbits(np.frombuffer(base64.b64decode(b64), np.uint8))
    return bits[:h * w].reshape(h, w).astype(bool)


def rotated(bm, angle):
    if angle == 0:
        return bm
    im = Image.fromarray(bm.astype(np.uint8) * 255).rotate(
        angle, resample=Image.BILINEAR, expand=True)
    a = np.asarray(im) > 127
    ys, xs = np.nonzero(a)
    return a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


@lru_cache(maxsize=1)
def templates():
    """-> (classes, 20x20 vectors, box sizes, (hole count, hole height)),
    one row per stored piece per angle in ANGLES."""
    data = json.loads(_GLYPHS.read_text())
    chars, vecs, sizes, topo = [], [], [], []
    for ch, h, w, b64 in data["glyphs"]:
        base = unpack(h, w, b64)
        for ang in ANGLES:
            bm = rotated(base, ang)
            v, hw = features(bm)
            chars.append(ch)
            vecs.append(v)
            sizes.append(hw)
            topo.append(holes(bm))
    return (chars, np.array(vecs, np.float32), np.array(sizes, np.float32),
            np.array(topo, np.float32))


LEFT = set("123456789") | {"10"}          # Securimage math: left 1..10
RIGHT = set("12345")                      # right 1..5
DIGITS = set("0123456789")
PEAK_CUT = 2.4                            # + vs -, see _operator
K = 3                                     # nearest exemplars per class


def classify(bm, allowed=None):
    """-> [(distance, class)] best first, one entry per class, restricted to
    `allowed` when given."""
    chars, vecs, sizes, topo = templates()
    v, hw = features(bm)
    n, hy = holes(bm)
    d = np.abs(vecs - v).mean(1)
    # Size matters most for "-" (short) and "1" (narrow): log-ratio penalty.
    d = d + 0.15 * np.abs(np.log((hw + 1) / (sizes + 1))).sum(1)
    # Topology: a different hole count, or a hole at a different height.
    d = d + 0.25 * (topo[:, 0] != n) + 1.5 * np.where(
        (topo[:, 0] == n) & (n > 0), np.abs(topo[:, 1] - hy), 0.0)
    # A class scores the mean of its K nearest exemplars, so a class with
    # many exemplars (1) cannot win on one lucky near-miss.
    per = {}
    for c, e in zip(chars, d.tolist()):
        if allowed is None or c in allowed:
            per.setdefault(c, []).append(e)
    return sorted((float(np.mean(sorted(es)[:K])), c) for c, es in per.items())


def _pick(bm, allowed):
    ranked = classify(bm, allowed)
    if not ranked:
        return None, 0.0
    e1, c1 = ranked[0]
    e2 = ranked[1][0] if len(ranked) > 1 else e1 + 1.0
    return c1, (e2 - e1) / max(e2, 1e-6)


def read(image):
    """-> (expression like '10-2', answer int, margin), or None.

    Layout is `left op right`: three ink pieces (a touching `10` is one
    piece and its own class), or four when the `1` and `0` stand apart.
    Each position is matched only against the classes it can hold. The
    operator piece must also look more like an operator than like a digit."""
    mask = ink(image)
    if mask is None or mask.sum() < 60:
        return None
    comps = [bm for _, _, bm in components(mask)]
    r = _read_layout(comps)
    if r is None and 2 <= len(comps) <= 3:
        # Two glyphs touching make one piece: cut the widest piece at its
        # thinnest column (middle half only) and try again.
        i = max(range(len(comps)), key=lambda j: comps[j].shape[1])
        halves = _split(comps[i])
        if halves:
            r = _read_layout(comps[:i] + halves + comps[i + 1:])
    return r


def _operator(bm):
    """+ or -, by structure rather than by template. A bar's per-column (and
    per-row) ink counts are flat; a cross peaks where its other stroke
    crosses. Peak / median: every train "+" >= 3.0, every "-" <= 1.89."""
    c, r = bm.sum(0).astype(float), bm.sum(1).astype(float)
    peak = max(c.max() / max(np.median(c), 1.0), r.max() / max(np.median(r), 1.0))
    op = "+" if peak >= PEAK_CUT else "-"
    # margin: how far past the cut, relative to the gap the train data left
    edge = 3.0 if op == "+" else 1.9
    return op, float(min(1.0, abs(peak - PEAK_CUT) / abs(edge - PEAK_CUT)))


def _split(bm):
    w = bm.shape[1]
    if w < 12:
        return None
    lo, hi = w // 4, w - w // 4
    x = lo + int(np.argmin(bm[:, lo:hi].sum(0)))
    out = []
    for part in (bm[:, :x], bm[:, x:]):
        ys, xs = np.nonzero(part)
        if len(xs) < MIN_PX:
            return None
        out.append(part[ys.min():ys.max() + 1, xs.min():xs.max() + 1])
    return out


def _read_layout(comps):
    if len(comps) not in (3, 4):
        return None
    *left_bms, op_bm, right_bm = comps
    if classify(op_bm)[0][1] not in OPS:
        return None
    op, m_op = _operator(op_bm)
    right, m_r = _pick(right_bm, RIGHT)
    if len(left_bms) == 2:                       # "1" and "0" apart
        one, m1 = _pick(left_bms[0], DIGITS)
        zero, m0 = _pick(left_bms[1], DIGITS)
        if (one, zero) != ("1", "0"):
            return None
        left, m_l = "10", min(m1, m0)
    else:
        left, m_l = _pick(left_bms[0], LEFT)
        # A touching "10" can pass for a 4 or a wide digit. Cut it at its
        # thinnest column: if the halves match 1 and 0 better than the whole
        # matches any single digit, it is 10.
        halves = _split(left_bms[0])
        if halves and left != "10":
            e1 = classify(halves[0], {"1"})
            e0 = classify(halves[1], {"0"})
            whole = classify(left_bms[0], LEFT - {"10"})[0][0]
            if e1 and e0 and max(e1[0][0], e0[0][0]) < whole:
                left, m_l = "10", (whole - max(e1[0][0], e0[0][0])) / max(whole, 1e-6)
    if left is None or right is None:
        return None
    expr = left + op + right
    return expr, OPS[op](int(left), int(right)), float(min(m_op, m_r, m_l))


def solve_image(image):
    """-> (answer as text, confidence). ('', 0.0) when unreadable."""
    try:
        r = read(image)
    except (OSError, ValueError):
        return "", 0.0
    if r is None:
        return "", 0.0
    _, answer, margin = r
    return str(answer), float(min(1.0, margin / 0.25))


def solve_group(images):
    """Vote over several renders of ONE captcha (the scid is sticky, so
    fetching the image again re-draws the same expression).
    -> (answer as text, confidence): the answer with the largest summed
    confidence, and that sum's share of the total, capped by its best
    single-render confidence."""
    votes, best = {}, {}
    for im in images:
        t, c = solve_image(im)
        if t:
            votes[t] = votes.get(t, 0.0) + c + 1e-3
            best[t] = max(best.get(t, 0.0), c)
    if not votes:
        return "", 0.0
    total = sum(votes.values())
    t = max(votes, key=votes.get)
    return t, float(min(best[t], votes[t] / total) if len(votes) > 1 else best[t])
