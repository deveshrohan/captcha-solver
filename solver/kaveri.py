"""Kaveri portal captcha support (kaveri.karnataka.gov.in): 200x60 RGBA,
6 uppercase-alphanumeric characters over coloured straight noise lines.

Composition (measured over 198 real captchas, not assumed):

* the served PNG is 200x60 **RGBA**, but the bottom 10 rows are fully
  transparent in every sample — the generator only paints a 200x50 area.
* the background is a flat `#f5fffa` (CSS "mintcream"), byte-identical in every
  captcha.
* the ink is **pure black `(0, 0, 0)`** and the noise lines are drawn
  *underneath* the text, so `rgb == (0,0,0)` recovers every glyph whole and
  unbroken. No noise line was ever black: across 198 images the black mask
  contains nothing but glyphs. The lines therefore cannot affect recognition
  and are not modelled at all.
* the text sits on a fixed baseline: the topmost ink row is `y = 14` in all 198
  images, with zero variation. Flat-topped glyphs start at 14, round ones
  (`0 6 8 9 C G O Q S`) at 15.
* every instance of a character is a **byte-identical bitmap**. Segmenting 1134
  glyph instances yields exactly 36 distinct pixel-exact clusters — the full
  `0-9A-Z` charset, roughly uniformly distributed. The renderer blits fixed
  sprites; there is no rotation, no per-character scaling and no warp.

Which makes this captcha *invertible* rather than merely learnable: the image is
a deterministic composition of 36 known bitmaps, so reading it is an exact cover
problem, not a classification problem. `solve()` searches for a set of six
sprites whose union is exactly the ink mask. There is no model and no training.

Two consequences worth stating:

* **No ambiguous pairs.** `O` is 23px wide against `0` at 17, and `1` is 15px
  with a diagonal flag against `I` at a bare 3px bar. These are exactly the
  homoglyph collisions that cap the MCA reader at 75%; here they are trivially
  separable, so `api.AMBIGUOUS` has no entry for this kind.
* **Drift is loud.** If the generator changes font, adds antialiasing or moves
  off six characters, no exact cover exists and confidence collapses toward 0
  instead of returning confident nonsense. That is what `selfimprove.py check`
  watches.
"""
import io
import json
import os

import numpy as np
from PIL import Image

CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
LENGTH = 6

W, H = 200, 60
# The generator paints only the top 50 rows; rows 50..59 are fully transparent.
# Cropping to PAINTED_H is also what makes the ink test safe against a caller
# that has already done `img.convert("RGB")` — PIL maps transparent pixels to
# (0,0,0), which is indistinguishable from ink. Text never reaches below y=45
# (the `J` descender), so the crop cannot remove signal.
PAINTED_H = 50

BACKGROUND = (245, 255, 250)

_HERE = os.path.dirname(os.path.abspath(__file__))
_GLYPH_FILE = os.path.join(_HERE, "kaveri_glyphs.json")

_GLYPHS = None


def glyphs():
    """-> {char: (bool array (h, w), top row)}. Loaded once, cached."""
    global _GLYPHS
    if _GLYPHS is None:
        with open(_GLYPH_FILE) as f:
            raw = json.load(f)
        _GLYPHS = {
            c: (np.array([[ch == "1" for ch in row] for row in v["rows"]], dtype=bool),
                int(v["top"]))
            for c, v in raw.items()
        }
    return _GLYPHS


# --------------------------------------------------------------------------
# preprocessing: exact, because the ink is a known constant and lines sit under it
# --------------------------------------------------------------------------
def _open(image):
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, (bytes, bytearray)):
        return Image.open(io.BytesIO(image))
    return Image.open(image)


def ink_mask(image):
    """bytes | path | PIL.Image -> bool array (PAINTED_H, W), True where ink.

    Ink is pure black and opaque. A fully transparent pixel is *not* ink even
    though its RGB reads as (0,0,0), which is why alpha is checked before the
    crop rather than relying on the crop alone.
    """
    im = _open(image)
    if im.mode != "RGBA":
        im = im.convert("RGBA")
    a = np.asarray(im)
    rgb = a[..., :3]
    m = (rgb[..., 0] == 0) & (rgb[..., 1] == 0) & (rgb[..., 2] == 0) & (a[..., 3] == 255)
    return _fit(m)


def _fit(m):
    """Crop or pad an arbitrary mask to exactly (PAINTED_H, W).

    A served image of the wrong size is not an error here: it produces a mask
    that no cover explains, so it surfaces as zero confidence — the signal
    `selfimprove check` is built to catch — rather than as an exception in the
    middle of a scrape.
    """
    out = np.zeros((PAINTED_H, W), dtype=bool)
    h = min(m.shape[0], PAINTED_H)
    w = min(m.shape[1], W)
    out[:h, :w] = m[:h, :w]
    return out


def load_real(path_or_bytes):
    return ink_mask(path_or_bytes)


# --------------------------------------------------------------------------
# reading: exact cover of the ink by six sprites
# --------------------------------------------------------------------------
def _placements(mask, y0, x0):
    """Every (char, x) placing a sprite so it covers pixel (y0, x0) and its ink
    lies entirely inside `mask`.

    Anchoring on the leftmost uncovered *pixel* rather than assuming the sprite
    starts at that column is what makes the search complete: glyphs can overlap
    in x. `T` followed by `J` does exactly this — the J tucks under the T's
    crossbar — which is 1 image in 198 and the only case a disjoint
    left-to-right segmentation gets wrong.
    """
    out = []
    for ch, (g, top) in glyphs().items():
        gh, gw = g.shape
        gy = y0 - top
        if gy < 0 or gy >= gh:
            continue
        if top + gh > mask.shape[0]:
            continue
        for dx in range(gw):
            if not g[gy, dx]:
                continue
            x = x0 - dx
            if x < 0 or x + gw > mask.shape[1]:
                continue
            win = mask[top:top + gh, x:x + gw]
            if not (g & ~win).any():          # sprite ink must lie inside the mask
                out.append((ch, x))
    return out


# Upper bound on the ink `length` sprites can possibly account for. The search
# is exhaustive, so it needs a way to reject hopeless masks before descending
# into them: a solid block of ink admits ~70 placements per position and would
# otherwise explore ~70^5 branches. Any real captcha is far under this bound
# (measured: 589..1081 ink pixels against a bound of 1196).
_MAX_INK = {}


def _max_ink(length):
    if length not in _MAX_INK:
        sizes = sorted((int(g.sum()) for g, _ in glyphs().values()), reverse=True)
        _MAX_INK[length] = sum(sizes[:length])
    return _MAX_INK[length]


# Backstop for shapes the ink bound lets through but that still branch widely.
# Reaching it means the image is nothing like a Kaveri captcha, which the caller
# sees as zero confidence.
MAX_NODES = 200_000


def solutions(mask, length=LENGTH, limit=2):
    """Enumerate readings whose sprite union is EXACTLY `mask`.

    Returns up to `limit` strings. `limit=2` is the useful default for callers
    that want to know whether the reading is *unique*: where exactly one cover
    exists, no other string could have produced that ink, which is a structural
    guarantee rather than a sample statistic.

    Bounded work: a mask carrying more ink than `length` sprites could account
    for is rejected outright, and the search abandons after `MAX_NODES`
    expansions. Both return "no cover found" rather than raising, because the
    caller is usually a scrape loop where a wrong-looking image must cost a
    retry, not a hang.
    """
    if int(mask.sum()) > _max_ink(length):
        return []

    found = []
    glyph_items = glyphs()
    nodes = [0]

    def rec(covered, chars, min_x):
        if len(found) >= limit or nodes[0] > MAX_NODES:
            return
        nodes[0] += 1
        rem = mask & ~covered
        if not rem.any():
            if len(chars) == length:
                found.append("".join(c for c, _ in chars))
            return
        if len(chars) == length:
            return                                  # ink left over, no budget
        cols = rem.any(0)
        x0 = int(np.argmax(cols))
        y0 = int(np.argmax(rem[:, x0]))
        for ch, x in _placements(mask, y0, x0):
            if x < min_x:
                continue                            # origins advance left to right
            g, top = glyph_items[ch]
            gh, gw = g.shape
            nxt = covered.copy()
            nxt[top:top + gh, x:x + gw] |= g
            rec(nxt, chars + [(ch, x)], x)
            if len(found) >= limit or nodes[0] > MAX_NODES:
                return

    rec(np.zeros_like(mask), [], 0)
    return found


def _best_effort(mask, length=LENGTH, beam=12):
    """Degraded read for images no exact cover explains.

    Beam search minimising pixel disagreement. This path exists so that a
    generator change degrades into a low-confidence read rather than an
    exception or a confident wrong answer; it is not expected to run at all
    against the current generator (0 of 198 images needed it).
    """
    glyph_items = glyphs()
    if not mask.any():
        return "", int(mask.sum())
    states = [(np.zeros_like(mask), [], 0)]
    for _ in range(length):
        nxt = []
        for covered, chars, min_x in states:
            rem = mask & ~covered
            if not rem.any():
                nxt.append((covered, chars, min_x))
                continue
            x0 = int(np.argmax(rem.any(0)))
            y0 = int(np.argmax(rem[:, x0]))
            cand = _placements(mask, y0, x0)
            if not cand:                            # nothing fits: try every sprite here
                cand = [(ch, x0) for ch in glyph_items]
            for ch, x in cand:
                if x < min_x:
                    continue
                g, top = glyph_items[ch]
                gh, gw = g.shape
                if x + gw > mask.shape[1] or top + gh > mask.shape[0]:
                    continue
                c2 = covered.copy()
                c2[top:top + gh, x:x + gw] |= g
                nxt.append((c2, chars + [ch], x))
        if not nxt:
            break
        nxt.sort(key=lambda s: int((mask ^ s[0]).sum()))
        states = nxt[:beam]
    covered, chars, _ = min(states, key=lambda s: int((mask ^ s[0]).sum()))
    return "".join(chars), int((mask ^ covered).sum())


def solve(mask, length=LENGTH):
    """bool mask -> (text, confidence).

    Confidence is 1.0 for an exact cover and `1 - mismatch/ink` otherwise, so it
    is a genuine measure of how well the sprite library explains the image
    rather than a softmax that stays high when the input goes out of
    distribution.
    """
    hit = solutions(mask, length=length, limit=1)
    if hit:
        return hit[0], 1.0
    text, mism = _best_effort(mask, length=length)
    ink = int(mask.sum())
    return text, max(0.0, 1.0 - mism / ink) if ink else 0.0


def solve_image(image, length=LENGTH):
    """bytes | path | PIL.Image -> (text, confidence)."""
    return solve(ink_mask(image), length=length)


def predict(masks, length=LENGTH):
    """Batch helper mirroring the CRNN readers' `predict`."""
    out = [solve(m, length=length) for m in masks]
    return [t for t, _ in out], [c for _, c in out]


# --------------------------------------------------------------------------
# rebuilding the sprite library
# --------------------------------------------------------------------------
def extract_sprites(paths, length=LENGTH):
    """-> [(count, bool array, top)] sorted by frequency, for re-deriving
    `kaveri_glyphs.json` if the generator ever changes its font.

    Only images that segment cleanly into `length` runs contribute, so touching
    glyphs cannot corrupt a template. Clusters come out **unlabelled**: mapping
    each bitmap to a character is a human step, and it is the one step no
    automated check catches — a mislabelled sprite would silently corrupt every
    read containing it. `eval_kaveri.py --sprites` dumps the labelled sheet so
    that mapping can be eyeballed directly.
    """
    clusters = {}
    for p in paths:
        m = ink_mask(p)
        runs = ink_runs(m)
        if len(runs) != length:
            continue
        for x0, x1 in runs:
            sub = m[:, x0:x1 + 1]
            ys = np.where(sub.any(1))[0]
            g = sub[ys.min():ys.max() + 1]
            key = (g.shape, g.tobytes(), int(ys.min()))
            if key not in clusters:
                clusters[key] = [0, g, int(ys.min())]
            clusters[key][0] += 1
    return sorted(((n, g, t) for n, g, t in clusters.values()),
                  key=lambda c: -c[0])


def ink_runs(mask):
    """Maximal runs of columns containing ink. Separates glyphs in 189 of 198
    real images; the rest have touching or overlapping glyphs, which is why the
    reader covers by union instead of relying on this."""
    cols = mask.any(0)
    runs, start = [], None
    for x in range(mask.shape[1]):
        if cols[x] and start is None:
            start = x
        elif not cols[x] and start is not None:
            runs.append((start, x - 1))
            start = None
    if start is not None:
        runs.append((start, mask.shape[1] - 1))
    return runs
