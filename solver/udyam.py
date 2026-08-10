"""Udyam registration portal captcha support (udyamregistration.gov.in):
225x80 RGBA, 6 uppercase-alphanumeric characters.

Composition (measured over 400+ real captchas, not assumed):

* the PNG is written by .NET GDI+ (`sRGB`/`gAMA`/`pHYs` chunks, no text chunks)
  and is fully opaque — every alpha byte is 255, so unlike Kaveri there is no
  transparency trap.
* fixed furniture, identical in every image: a 1px border at row 0, row 79 and
  columns 0 and 224; a saffron bar on rows 2-3 and a green bar on rows 7-11.
  There are no bottom bars. Both the border `(100,130,180)` and the green bar
  `(0,128,0)` are dark enough to pass any ink test, which is why the reader
  works on the fixed text band (rows 32..60, columns 1..223), not the raw image.
* the ink is a solid navy `(25,60,130)` with a ClearType subpixel fringe — the
  fringe is why a bare `rgb == INK` test leaves ragged edges.
* 2-4 straight noise lines are drawn **over** the finished text. This is the
  one structural difference from Kaveri, where the lines sit underneath and
  `rgb == (0,0,0)` recovers every glyph whole.

The lines are *alpha-blended*, not painted opaque, and that is what makes this
captcha readable exactly. Where a line crosses ink the result keeps the ink's
red and green channels and lifts only blue — `(25,60,153)`, `(25,60,175)`,
`(25,100,196)` — so the occluded pixels are still dark, while the same line over
the pale background stays light. A luminance cut at `INK_LUM` therefore recovers
the crossed pixels and rejects the line itself, which restores whole glyphs:
with it, glyphs never fragment (a `7` cut clean through by a line stops
splitting into two column runs).

What actually varies between two instances of the same character is not
occlusion but **subpixel phase**: the renderer places glyphs at fractional
positions, so the antialiased core shifts by a pixel. Recovering the occluded
pixels alone moves the exact-duplicate rate only 61.8% -> 64.7%, but matching
with a +/-2px alignment collapses 147 apparent bitmap variants into 34 stable
classes covering the full 33-character charset. So the reader normalises for
phase rather than enumerating it — mostly. `5` is the one character that renders
in two phases distinct enough to survive alignment (IoU 0.828 between them), so
the library holds a **list of variants per character** instead of one template
each. That is deliberate rather than a tuning failure: merging the two `5`s by
lowering the clustering threshold would have to reach past `E` vs `F` at 0.794,
and collapsing those two would corrupt every read containing either.

That makes this captcha invertible in the same sense Kaveri is, one step weaker:
Kaveri admits an exact cover, Udyam a cover under subtraction. Reading is
template identification, not classification — there is no model and no training.

Two consequences worth stating:

* **No ambiguous pairs.** The portal's alphabet excludes `0`, `I` and `O`
  entirely (33 characters, confirmed by the class count landing exactly on 33).
  Those are the homoglyph collisions that cap the MCA reader at 75%, so
  `api.AMBIGUOUS` has no entry for this kind, exactly as for Kaveri.
* **Drift is loud.** A font change leaves glyphs that no class explains, so
  confidence collapses instead of returning confident nonsense. That is what
  `selfimprove.py check` watches.
"""
import io
import json
import os

import numpy as np
from PIL import Image

# 33 characters: the digits 1-9 and A-Z without I and O. Zero is absent too.
# Confirmed structurally rather than by eye — alignment-tolerant clustering of
# real glyphs yields exactly 33 classes with >=3 members, and the labelled sheet
# accounts for all 33 with no character left over.
CHARSET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZ"
LENGTH = 6

W, H = 225, 80

# The band that can contain text, in absolute image coordinates. Measured from
# the corpus row profile: glyph ink lives in rows 33..59 and nothing else does.
# Rows 0-3 and 7-11 are the saffron and green bars, rows 0 and 79 and columns 0
# and 224 the border — all dark enough to pass the luminance cut, so the band is
# what keeps them out. Rows 48+ of the old wider band carried only ~2px per
# image of noise-line residue; clipping here takes the worst glyph bounding box
# from 45 rows to 28 across all 2112 measured glyphs.
BAND_TOP, BAND_BOT = 32, 61
BAND_LEFT, BAND_RIGHT = 1, 224
BAND_H = BAND_BOT - BAND_TOP
BAND_W = BAND_RIGHT - BAND_LEFT

INK = (25, 60, 130)

# Luminance cut separating {ink, line-over-ink} from {background, line-over-
# background, hatch}. Chosen from the measured gap: ink and line-crossed ink sit
# at luminance 53..90, the background at 234..255, and the lines over background
# just under 200. At 200 the lines themselves are picked up and every image
# collapses to a single column run; at 160 and 180 segmentation is identical, so
# 170 sits in the middle of the working range rather than on its edge.
INK_LUM = 170

# Alignment slack when matching a glyph against a class template, in pixels.
# Absorbs subpixel phase, which is the dominant source of bitmap variation.
ALIGN = 2

_HERE = os.path.dirname(os.path.abspath(__file__))
_GLYPH_FILE = os.path.join(_HERE, "udyam_glyphs.json")

_CLASSES = None


def _unpack(rows):
    return np.array([[c == "1" for c in r] for r in rows], dtype=bool)


def _pack(arr):
    return ["".join("1" if v else "0" for v in row) for row in arr]


def classes():
    """-> {char: [bitmap, ...]} of bool arrays. Loaded once, cached.

    A *list* per character because a character may render in more than one
    subpixel phase (only `5` does, today).

    Each bitmap is the least-occluded instance observed for that class — the
    member carrying the most ink.

    The obvious alternative is the *envelope*, the union over aligned members:
    occlusion only removes ink, so a union reconstructs the un-occluded glyph
    and no single instance can be guaranteed whole. It was built and measured,
    and it is not better. Over all 420 real captchas the two tie on accuracy
    (36/36 labelled) and on how often they fall below the 0.90 gate (3 images
    each); the envelope is better in the bulk (1st percentile 0.940 vs 0.914)
    but has a lower floor (0.842 vs 0.869) and a narrower margin to degraded
    input (0.090 vs 0.125). The floor is what the gate actually rides on, so the
    plainer choice wins on the evidence.
    """
    global _CLASSES
    if _CLASSES is None:
        with open(_GLYPH_FILE) as f:
            raw = json.load(f)
        _CLASSES = {c: [_unpack(v) for v in variants] for c, variants in raw.items()}
    return _CLASSES


def save_classes(path, data):
    """Write a class library built by `extract_classes` (see mkglyphs_udyam.py).

    `data` maps char -> list of bool-array bitmaps.
    """
    out = {c: [_pack(v) for v in variants] for c, variants in data.items()}
    with open(path, "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)


# --------------------------------------------------------------------------
# preprocessing
# --------------------------------------------------------------------------
def _open(image):
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, (bytes, bytearray)):
        return Image.open(io.BytesIO(image))
    return Image.open(image)


def ink_mask(image):
    """bytes | path | PIL.Image -> bool array (BAND_H, BAND_W), True where ink.

    Works on the fixed text band, so the border and the tricolour bars — both
    dark enough to pass the luminance cut — cannot contribute. A served image of
    the wrong size is not an error: it yields a mask no class library explains,
    which surfaces as low confidence rather than an exception mid-scrape.
    """
    im = _open(image).convert("RGB")
    a = np.asarray(im).astype(np.int16)
    lum = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    return _despeckle(_fit(lum < INK_LUM))


# Smallest connected component kept. The darkest points of a noise line crossing
# the pale background scrape under the luminance cut and survive as specks: 173
# of 176 sub-glyph components measured over 80 images are under 5px, against 469
# glyph components at 100px or more, so the gap this sits in is wide.
#
# Filtering them is not cosmetic. A stray 2px speck 20 rows below the text costs
# nothing in segmentation — it is too narrow to open a column run — but it
# stretches the glyph's bounding box from 25 rows to 46, and every template
# built from that crop is then junk. That single effect was enough to make `U`,
# `H` and `F` unreadable.
MIN_COMPONENT = 12


def _despeckle(m, min_px=MIN_COMPONENT):
    """Drop connected components (8-connected) smaller than `min_px`."""
    ys, xs = np.nonzero(m)
    if not len(ys):
        return m
    live = set(zip(ys.tolist(), xs.tolist()))
    out = np.zeros_like(m)
    while live:
        seed = live.pop()
        stack, comp = [seed], [seed]
        while stack:
            cy, cx = stack.pop()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    nb = (cy + dy, cx + dx)
                    if nb in live:
                        live.discard(nb)
                        stack.append(nb)
                        comp.append(nb)
        if len(comp) >= min_px:
            for cy, cx in comp:
                out[cy, cx] = True
    return out


def _fit(m):
    out = np.zeros((BAND_H, BAND_W), dtype=bool)
    src = m[BAND_TOP:BAND_BOT, BAND_LEFT:BAND_RIGHT]
    h = min(src.shape[0], BAND_H)
    w = min(src.shape[1], BAND_W)
    out[:h, :w] = src[:h, :w]
    return out


def load_real(path_or_bytes):
    return ink_mask(path_or_bytes)


# --------------------------------------------------------------------------
# segmentation
# --------------------------------------------------------------------------
def ink_runs(mask, min_w=3):
    """Maximal runs of columns containing ink.

    Gives exactly six runs in 82% of real images. The rest have glyphs that
    touch (measured: 13 of 79 produce five runs, one produces four), which is
    why `segments` repairs the count instead of trusting it.
    """
    cols = mask.any(0)
    runs, start = [], None
    for x in range(mask.shape[1]):
        if cols[x] and start is None:
            start = x
        elif not cols[x] and start is not None:
            if x - start >= min_w:
                runs.append((start, x - 1))
            start = None
    if start is not None and mask.shape[1] - start >= min_w:
        runs.append((start, mask.shape[1] - 1))
    return runs


def _crop(mask, x0, x1):
    """Column slice cropped to its ink bounding box -> (glyph, y_top) or None."""
    sub = mask[:, x0:x1 + 1]
    ys = np.nonzero(sub.any(1))[0]
    if not len(ys):
        return None, 0
    return sub[ys.min():ys.max() + 1], int(ys.min())


def _iou_at(g, t):
    """Best IoU of two bool bitmaps over +/-ALIGN alignment -> (iou, dy, dx).

    `(dy, dx)` places `t`'s origin at `g[dy, dx]`, which the cover scorer needs
    in order to put the chosen template back into mask coordinates.

    Computed by slicing the overlapping rectangle rather than blitting both onto
    a padded canvas: `union = |g| + |t| - |intersection|` needs only the
    intersection, so no per-shift allocation is required. That is the difference
    between this reader running at ~10ms and ~300ms per image, which matters
    because it sits inside a scrape loop.
    """
    gs = int(g.sum())
    ts = int(t.sum())
    if not gs or not ts:
        return 0.0, 0, 0
    gh, gw = g.shape
    th, tw = t.shape
    best, bdy, bdx = 0.0, 0, 0
    for dy in range(-ALIGN, ALIGN + 1):
        gy, ty = max(0, dy), max(0, -dy)
        h = min(gh - gy, th - ty)
        if h <= 0:
            continue
        for dx in range(-ALIGN, ALIGN + 1):
            gx, tx = max(0, dx), max(0, -dx)
            w = min(gw - gx, tw - tx)
            if w <= 0:
                continue
            inter = np.count_nonzero(g[gy:gy + h, gx:gx + w] & t[ty:ty + h, tx:tx + w])
            u = gs + ts - inter
            if u:
                s = inter / u
                if s > best:
                    best, bdy, bdx = s, dy, dx
    return best, bdy, bdx


def _iou(g, t):
    return _iou_at(g, t)[0]


def _paint(canvas, tpl, y, x, oy, ox):
    """OR `tpl` into `canvas`, whose (0,0) is mask coordinate (oy, ox)."""
    h, w = tpl.shape
    y -= oy
    x -= ox
    ty0, tx0 = max(0, -y), max(0, -x)
    cy0, cx0 = max(0, y), max(0, x)
    hh = min(h - ty0, canvas.shape[0] - cy0)
    ww = min(w - tx0, canvas.shape[1] - cx0)
    if hh > 0 and ww > 0:
        canvas[cy0:cy0 + hh, cx0:cx0 + ww] |= tpl[ty0:ty0 + hh, tx0:tx0 + ww]


# Cover scoring. Leftover ink is weighted above hallucinated ink because the two
# mean different things: the noise lines are subtractive, so a template may
# legitimately explain *more* than is observed, but ink the chosen templates
# cannot account for at all means the reading is simply wrong.
#
# This is what separates `WX` from `WK`. Cutting a merged `WX` too far right
# leaves an X-minus-its-left-edge that is a pixel-perfect `K`, so per-piece IoU
# scores it as high as the truth; only the X's orphaned lower-left stroke, left
# unexplained by `W` and `K` together, distinguishes them.
LEFTOVER_W = 2.0
EXTRA_W = 1.0


def _cover_score(mask, x0, x1, placed):
    """How well the chosen templates explain the ink in columns [x0, x1]."""
    sub = mask[:, x0:x1 + 1]
    canvas = np.zeros_like(sub)
    for _, _, y, x, tpl in placed:
        _paint(canvas, tpl, y, x, 0, x0)
    inter = np.count_nonzero(sub & canvas)
    leftover = np.count_nonzero(sub & ~canvas)
    extra = np.count_nonzero(canvas & ~sub)
    return inter - LEFTOVER_W * leftover - EXTRA_W * extra


# A glyph is never narrower than this once cropped; used to bound split points.
MIN_GLYPH_W = 6
# A single run can in principle hold the whole string: 4-run images already
# occur, so a cap of 3 was one merge away from being unreadable.
MAX_PER_SPAN = LENGTH


# Alternative characters carried forward per position. The truth is not always
# the local favourite — that is the entire point of scoring by cover — so the
# runners-up have to stay in play.
CAND_K = 4
# Horizontal slack when anchoring a glyph on the leftmost unexplained ink column.
ANCHOR_SLACK = 2


def _anchored_all(mask, xleft, ytop):
    """Every character's best placement with its left edge at about `xleft`.

    -> [(gain, char, y, x, tpl)] sorted best first. Glyphs are anchored on the
    leftmost ink they must explain rather than carved out by a vertical cut,
    which is what lets two of them **overlap in x**. They demonstrably do: a
    merged `WX` spans 55 columns while `W` (35) and `X` (27) need 62, so they
    share 7 columns and no cut can yield both — cutting to complete the `W`
    leaves an X-minus-left-edge that is a pixel-perfect `K`. This is the same
    reason the Kaveri reader anchors on a pixel rather than segmenting left to
    right.

    `gain` rewards ink explained and penalises ink the template invents; ink it
    fails to explain is left to the glyphs that follow.
    """
    H, Wd = mask.shape
    out = []
    for ch, variants in classes().items():
        best = None
        for t in variants:
            th, tw = t.shape
            for x in range(max(0, xleft - ANCHOR_SLACK), xleft + ANCHOR_SLACK + 1):
                if x + tw > Wd:
                    continue
                # the text sits on a near-fixed baseline, so only a couple of
                # vertical positions are worth testing
                for y in range(max(0, ytop - 1), min(ytop + 3, H - th + 1)):
                    win = mask[y:y + th, x:x + tw]
                    inter = np.count_nonzero(win & t)
                    g = inter - EXTRA_W * (int(t.sum()) - inter)
                    if best is None or g > best[0]:
                        best = (g, ch, y, x, t)
        if best is not None:
            out.append(best)
    out.sort(key=lambda r: (-r[0], r[1]))
    return out


def _anchored(mask, xleft, ctx, k=CAND_K):
    """Memoised `_anchored_all` — the same column is re-anchored many times as
    the beam explores, and recomputing it dominated the runtime."""
    lst = ctx["anchors"].get(xleft)
    if lst is None:
        lst = _anchored_all(mask, xleft, ctx["ytop"])
        ctx["anchors"][xleft] = lst
    return lst if k is None else lst[:k]


def _first_uncovered(mask, covered, x_from, x1):
    cols = (mask & ~covered)[:, x_from:x1 + 1].any(0)
    idx = np.nonzero(cols)[0]
    return int(x_from + idx[0]) if len(idx) else None


def _read_span(mask, x0, x1, n, ctx):
    """Best reading of columns [x0, x1] as exactly `n` glyphs.

    -> (cover score, [(char, y, x, tpl, a, b), ...]). A small beam: each glyph
    is anchored on the leftmost ink its predecessors left unexplained, so
    placements may overlap, and the whole span is finally scored by how well the
    union of the chosen templates explains its ink. Each part carries the
    placement that produced it, so callers can rebuild the union without having
    to re-derive it.
    """
    key = (x0, x1, n)
    memo = ctx["spans"]
    if key in memo:
        return memo[key]

    best = (-1e9, [])
    for cand in _anchored(mask, x0, ctx):
        _, ch, y, x, tpl = cand
        if n == 1:
            s = _cover_score(mask, x0, x1, [cand])
            if s > best[0]:
                best = (s, [(ch, y, x, tpl, x0, x1)])
            continue
        covered = np.zeros_like(mask)
        _paint(covered, tpl, y, x, 0, 0)
        nx = _first_uncovered(mask, covered, min(x + 1, x1), x1)
        if nx is None:
            continue
        _, sub_parts = _read_span(mask, nx, x1, n - 1, ctx)
        if not sub_parts:
            continue
        rest = [(0, p[0], p[1], p[2], p[3]) for p in sub_parts]
        s = _cover_score(mask, x0, x1, [cand] + rest)
        if s > best[0]:
            best = (s, [(ch, y, x, tpl, x0, max(x0, nx - 1))] + sub_parts)
    memo[key] = best
    return best


def segments(mask, length=LENGTH):
    """-> exactly `length` (x0, x1) column spans, one per glyph."""
    return [(a, b) for _, _, _, _, a, b in _read(mask, length)]


def _read(mask, length=LENGTH):
    """-> [(char, y, x, tpl, x0, x1)] of exactly `length` glyphs, or [].

    A DP over the column runs that may merge consecutive runs (a glyph split by
    a line) or split one run into several glyphs (glyphs that touch), choosing
    whichever reading the class library explains best. Polynomial in the number
    of runs, so unlike an exhaustive cover search it cannot blow up — the trap
    the Kaveri reader had to bound with a node budget.
    """
    runs = ink_runs(mask)
    if not runs:
        return []

    rows = np.nonzero(mask.any(1))[0]
    ctx = {"anchors": {}, "spans": {}, "ytop": int(rows[0]) if len(rows) else 0}
    n = len(runs)
    NEG = -1e9
    best = {}

    def dp(i, k):
        """Best score covering runs[i:] with exactly k glyphs."""
        if i == n:
            return (0.0, []) if k == 0 else (NEG, [])
        if k <= 0:
            return (NEG, [])
        if (i, k) in best:
            return best[(i, k)]
        out = (NEG, [])
        for j in range(i + 1, min(n, i + MAX_PER_SPAN) + 1):     # merge runs i..j-1
            x0, x1 = runs[i][0], runs[j - 1][1]
            for g in range(1, min(MAX_PER_SPAN, k) + 1):         # as g glyphs
                if x1 - x0 + 1 < MIN_GLYPH_W * g:
                    continue
                s, parts = _read_span(mask, x0, x1, g, ctx)
                if not parts:
                    continue
                rest, rparts = dp(j, k - g)
                if rest <= NEG / 2:
                    continue
                if s + rest > out[0]:
                    out = (s + rest, parts + rparts)
        best[(i, k)] = out
        return out

    _, parts = dp(0, length)
    return parts


# --------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------
def solve(mask, length=LENGTH):
    """bool mask -> (text, confidence in [0, 1]).

    Confidence is `IoU(ink, union of the chosen templates)` — how much of the
    image the reading actually explains. It is a real measure of fit rather than
    a softmax that stays high out of distribution, so a generator change shows
    up as falling confidence instead of confident nonsense.

    Unlike Kaveri this never reaches exactly 1.0, and that is honest rather than
    a defect: the text is antialiased, so no set of templates reproduces the ink
    bit for bit and no *exact* cover exists to certify. Measured separation over
    420 real captchas against deliberately degraded input:

        real captchas      min 0.869   p1 0.913   median 0.971
        text scaled 15%        0.744
        text rotated 4deg      0.684
        solid black            0.120

    so the repo-wide default gate of 0.90 sits inside a real gap, at the cost of
    re-fetching roughly 0.7% of otherwise-good captchas.

    Scoring against `core`/`env` instead was measured and rejected: a solid black
    image trivially contains every core and scores 1.000 on that measure.
    """
    parts = _read(mask, length=length)
    if not parts:
        return "", 0.0

    canvas = np.zeros_like(mask)
    for _, y, x, tpl, _, _ in parts:
        _paint(canvas, tpl, y, x, 0, 0)
    union = np.count_nonzero(mask | canvas)
    conf = np.count_nonzero(mask & canvas) / union if union else 0.0

    text = "".join(ch for ch, _, _, _, _, _ in parts)
    if len(text) != length:
        conf = min(conf, 0.5)
    return text, float(conf)


def solve_image(image, length=LENGTH):
    """bytes | path | PIL.Image -> (text, confidence)."""
    return solve(ink_mask(image), length=length)


def substitution_margin(mask, length=LENGTH):
    """-> (text, [(index, runner_up_char, margin), ...]).

    For each glyph, how much worse the cover gets when that character alone is
    replaced by the best alternative, as a fraction of the ink under it. This is
    the structural statistic worth quoting: a large margin everywhere means no
    other character could have produced that ink, which is a property of the
    image rather than an accuracy estimate on a sample.

    It is the graded counterpart of Kaveri's exact-cover uniqueness. Antialiasing
    rules out an exact cover here, so uniqueness becomes "the runner-up explains
    the pixels measurably worse" instead of "no other cover exists".
    """
    parts = _read(mask, length=length)
    if not parts:
        return "", []
    rows = np.nonzero(mask.any(1))[0]
    ctx = {"anchors": {}, "spans": {}, "ytop": int(rows[0]) if len(rows) else 0}

    def cover(placed):
        canvas = np.zeros_like(mask)
        for y, x, tpl in placed:
            _paint(canvas, tpl, y, x, 0, 0)
        return (np.count_nonzero(mask & canvas)
                - LEFTOVER_W * np.count_nonzero(mask & ~canvas)
                - EXTRA_W * np.count_nonzero(canvas & ~mask))

    base_placed = [(y, x, tpl) for _, y, x, tpl, _, _ in parts]
    base = cover(base_placed)
    out = []
    for i, (ch, y, x, tpl, _, _) in enumerate(parts):
        best = None
        for gain, ach, ay, ax, atpl in _anchored(mask, x, ctx, k=None):
            if ach == ch:
                continue
            alt = list(base_placed)
            alt[i] = (ay, ax, atpl)
            drop = base - cover(alt)
            if best is None or drop < best[1]:
                best = (ach, drop)
        ink = max(1, int(tpl.sum()))
        out.append((i, best[0] if best else "", (best[1] / ink) if best else 0.0))
    return "".join(ch for ch, _, _, _, _, _ in parts), out


def predict(masks, length=LENGTH):
    """Batch helper mirroring the CRNN readers' `predict`."""
    out = [solve(m, length=length) for m in masks]
    return [t for t, _ in out], [c for _, c in out]


# --------------------------------------------------------------------------
# rebuilding the class library
# --------------------------------------------------------------------------
def extract_classes(paths, length=LENGTH, merge_iou=0.85, min_members=3):
    """Cluster real glyphs into unlabelled classes, for re-deriving the library.

    -> [(count, envelope)] sorted by frequency. Only images that segment into
    `length` clean runs contribute, so touching glyphs cannot corrupt a
    template.

    Clusters come out **unlabelled**: mapping each bitmap to a character is a
    human step, and it is the one step no automated check catches — a
    mislabelled class silently corrupts every read containing it while leaving
    every match unique. `eval_udyam.py --sprites` dumps the labelled sheet so
    the mapping can be eyeballed directly.
    """
    reps, members = [], []
    for p in paths:
        m = ink_mask(p)
        runs = ink_runs(m)
        if len(runs) != length:
            continue
        for x0, x1 in runs:
            g, _ = _crop(m, x0, x1)
            if g is None:
                continue
            for i, r in enumerate(reps):
                if _iou(g, r) >= merge_iou:
                    members[i].append(g)
                    break
            else:
                reps.append(g)
                members.append([g])

    out = []
    for r, ms in zip(reps, members):
        if len(ms) < min_members:
            continue
        out.append((len(ms), _template(ms)))
    out.sort(key=lambda c: -c[0])
    return out


def _template(ms):
    """Pick the class template: the member carrying the most ink.

    That is the least-occluded instance seen. `envelope(ms)` below builds the
    union-based alternative; `classes()` records why this one is used instead.
    """
    return max(ms, key=lambda g: int(g.sum()))


# Fraction of aligned members a pixel must appear in to enter the envelope.
# Not "any member": a strict union inflates the template with one-off strays
# from a single mis-aligned or speckled instance.
ENV_FRAC = 0.02


def envelope(ms):
    """Frequency-thresholded union of aligned members -> the un-occluded glyph.

    Kept because it is the principled reconstruction and the natural thing to
    reach for if the generator changes; measured against `_template` in
    `classes()`.
    """
    hh = max(g.shape[0] for g in ms) + 2 * ALIGN
    ww = max(g.shape[1] for g in ms) + 2 * ALIGN
    ref = max(ms, key=lambda g: int(g.sum()))
    R = np.zeros((hh, ww), bool)
    R[ALIGN:ALIGN + ref.shape[0], ALIGN:ALIGN + ref.shape[1]] = ref
    acc = np.zeros((hh, ww), np.int32)
    for g in ms:
        bestB, bestS = None, -1.0
        for dy in range(-ALIGN, ALIGN + 1):
            for dx in range(-ALIGN, ALIGN + 1):
                y0, x0 = ALIGN + dy, ALIGN + dx
                if y0 < 0 or x0 < 0 or y0 + g.shape[0] > hh or x0 + g.shape[1] > ww:
                    continue
                B = np.zeros((hh, ww), bool)
                B[y0:y0 + g.shape[0], x0:x0 + g.shape[1]] = g
                u = np.count_nonzero(R | B)
                s = np.count_nonzero(R & B) / u if u else 0.0
                if s > bestS:
                    bestS, bestB = s, B
        acc += bestB
    env = acc >= max(2, int(np.ceil(ENV_FRAC * len(ms))))
    return _trim(env)


def _trim(env):
    """Crop to the bounding box so templates carry no padding."""
    ys = np.nonzero(env.any(1))[0]
    xs = np.nonzero(env.any(0))[0]
    if not len(ys):
        return env
    return env[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
