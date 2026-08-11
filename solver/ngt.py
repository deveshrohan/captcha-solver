"""NGT captcha support (greentribunal.gov.in): 120x40 palette PNG, six
lowercase-alphanumeric characters in pure black over pastel single-pixel noise.

This is the National Green Tribunal's case-status captcha.

Composition (measured over 60 real captchas, not assumed):

* the served PNG is a **palette** image (`imagecreate`, not truecolor) carrying
  libgd's `pHYs` = 3780 ppm (96 DPI) — the PHP-GD fingerprint, the same lineage
  as the numeric captcha and ITAT, and not Java's ImageIO (GST) or .NET (Udyam).
* the ink is **pure black `(0, 0, 0)`**; every other non-white pixel has all
  three channels in `[150, 255]` (minimum luminance measured: 154). So
  `rgb == (0,0,0)` is an *exact* ink mask with a 154-level gulf beneath it —
  there is no threshold to tune and no train/test skew to worry about.
* the noise is 50 isolated single pixels at random positions, each an
  independent `rand(150,255)` per channel. It never forms lines and never
  touches a stroke.
* the text sits on a **fixed grid**: origin `x = 20`, pitch `9`, rows `13..24`.
  Across 60 images there was not one black pixel outside that grid, and every
  image inked all six cells.
* every instance of a character is a **byte-identical bitmap**. Segmenting 360
  glyph cells yields exactly 36 pixel-exact clusters with zero singletons — the
  full `0-9a-z` charset. There is no rotation, no antialiasing, no warp.

### The palette proves the layer order

The decisive measurement is not in the pixels but in the `PLTE` chunk, which is
**52 entries in every single image** — white, black, and exactly 50 noise
colours — while only 41..49 noise pixels are ever *visible*.

A palette entry survives its pixel being overdrawn. So the missing noise
colours are the ones painted underneath the glyphs, which fixes the layer order:
noise first, text last. Hence **the ink can never be damaged**, and that is not
a lucky observation extrapolated from a sample — it is forced by the order of
the drawing calls, and the palette is the evidence for the order.

Contrast ITAT, where opaque lines drawn *over* the text erase strokes and the
renderer has to reproduce the damage so the model can learn to read through it.
Here there is nothing to reproduce and nothing to learn.

Reconstructed generator:

    $im = imagecreate(120, 40);
    imagecolorallocate($im, 255,255,255);          // PLTE[0] background
    imagecolorallocate($im, 0,0,0);                // PLTE[1] text
    for ($i = 0; $i < 50; $i++)                    // PLTE[2..51]
        imagesetpixel($im, rand(0,119), rand(0,39),
            imagecolorallocate($im, rand(150,255), rand(150,255), rand(150,255)));
    imagestring($im, 5, 20, 10, $code, $black);    // font 5 = gdFontGiant, 9x15
    imagepng($im);

Which makes this captcha **invertible rather than merely learnable**, and more
cheaply so than Kaveri: the glyphs sit on a fixed grid, so there is not even a
cover search to run. Reading is six array slices and six dictionary lookups.
There is no model and no training.

Three consequences worth stating:

* **No ambiguous pairs.** All 36 bitmaps are distinct, and the homoglyphs that
  cap the MCA reader at 75% are far apart here (`0`/`o` 32 px, `5`/`s` 35,
  `9`/`g` 53). So `api.AMBIGUOUS` has no entry for this kind.
* **But `1`, `l` and `i` are close** — mutually 6 px apart, the tightest cluster
  in the font. Exactness is what makes them safe, so matching is *exact or
  bust*: a cell that does not match byte for byte reports its nearest label but
  can never reach confidence 1.0. A nearest-neighbour fallback that silently
  snapped a damaged cell to the closest bar-glyph would flip `1` to `l`
  invisibly, and three wrong pixels would be enough to do it.
* **Confidence 1.0 is a proof, not a saturated softmax.** It is returned only
  when all six cells match exactly *and* no ink lies outside the grid, which for
  this generator means the read is correct. So the retry gate to use here is
  `min_conf=1.0` — and if the generator ever changes, confidence collapses
  immediately instead of returning confident nonsense. That is what
  `selfimprove.py check` watches.
"""
import io
import json
import os

import numpy as np
from PIL import Image

# The full lowercase alphanumeric set: all 36 classes were observed, and the
# census is roughly uniform. Unlike EPFO/Udyam/ITAT nothing is excluded — the
# homoglyph-prone characters are all present and all separable (see above).
CHARSET = "0123456789abcdefghijklmnopqrstuvwxyz"
LENGTH = 6

# Native size. NOTE: this collides with the gstat numeric captcha, the only
# size collision in the repo — see `api._route_120x40` for the content test that
# separates them.
W, H = 120, 40

# The measured text grid. `imagestring` advances by a fixed 9px cell (GD's
# built-in font 5 = gdFontGiant), and the origin never moved across 60 images.
#
# The 15-row font cell sits at image rows 10..24; rows 13..24 are used because
# the 36 alphanumeric glyphs put ink in exactly font rows 3..14 and nowhere
# else. So this crop is lossless by construction, not a fitted bounding box --
# tests/test_ngt_font.py checks that against libgd's own bitmaps.
X0, PITCH = 20, 9
R0, R1 = 13, 25
CELL_W, CELL_H = PITCH, R1 - R0

# The noise, as reconstructed from the palette: exactly 50 pixels, each channel
# uniform in [150, 255].
NOISE_N = 50
NOISE_LO, NOISE_HI = 150, 255

_HERE = os.path.dirname(os.path.abspath(__file__))
_GLYPH_FILE = os.path.join(_HERE, "ngt_glyphs.json")

_GLYPHS = None
_BY_BITS = None


def _unpack(rows):
    return np.array([[c == "1" for c in r] for r in rows], dtype=bool)


def _pack(arr):
    return ["".join("1" if v else "0" for v in row) for row in arr]


def glyphs():
    """-> {char: bitmap} of (CELL_H, CELL_W) bool arrays. Loaded once, cached.

    One bitmap per character, not a list: unlike Udyam there is no subpixel
    phase to absorb, because the glyphs are blitted at integer positions on a
    fixed grid and every instance is byte-identical.

    Keys beginning with `_` are metadata, not characters (see `glyph_meta`)."""
    global _GLYPHS, _BY_BITS
    if _GLYPHS is None:
        with open(_GLYPH_FILE) as f:
            raw = json.load(f)
        _GLYPHS = {c: _unpack(v) for c, v in raw.items()
                   if not c.startswith("_")}
        _BY_BITS = {b.tobytes(): c for c, b in _GLYPHS.items()}
    return _GLYPHS


def glyph_meta():
    """-> the library's provenance dict, or {} for a library built before it was
    recorded.

    Carries `built_from`: "train-split" or "all-images". `eval_ngt.py` refuses to
    report a held-out accuracy against an "all-images" library, because
    templates derived from the images being scored would inflate the number.
    Recording it in the artifact makes that guard mechanical rather than a
    promise in a docstring."""
    with open(_GLYPH_FILE) as f:
        return json.load(f).get("_meta", {})


def save_glyphs(path, data, meta=None):
    """Write a glyph library built by mkglyphs_ngt.py. `data` maps char -> bool
    array; `meta` is a provenance dict stored under `_meta`."""
    out = {c: _pack(v) for c, v in data.items()}
    if meta:
        out["_meta"] = meta
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
    if isinstance(image, str):
        return Image.open(image)
    raise TypeError("image must be bytes, a path str, or a PIL.Image")


def ink_mask(image):
    """-> bool array, True exactly where the pixel is pure black.

    Equality, not a threshold. The noise floor is 150 so nothing else can enter
    the mask, and because the text is painted over the noise nothing is missing
    from it either."""
    a = np.asarray(_open(image).convert("RGB"), np.uint8)
    return (a == 0).all(2)


def cell_spans():
    """-> [(x0, x1)] for the six fixed glyph cells."""
    return [(X0 + PITCH * k, X0 + PITCH * (k + 1)) for k in range(LENGTH)]


def cells(mask):
    """-> [bitmap] of six (CELL_H, CELL_W) bool arrays.

    Pads rather than raising when the image is smaller than the grid, so an
    image of the wrong size degrades to a bad read instead of an exception."""
    out = []
    for x0, x1 in cell_spans():
        c = np.zeros((CELL_H, CELL_W), bool)
        src = mask[R0:R1, x0:x1]
        if src.size:
            c[:src.shape[0], :src.shape[1]] = src
        out.append(c)
    return out


# --------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------
def _match(cell):
    """-> (char, hamming). Exact hit by hash; otherwise the nearest template.

    The nearest label is reported so a caller can see *what* was read, but the
    distance is returned with it so `solve` can refuse to call it certain."""
    glyphs()
    hit = _BY_BITS.get(cell.tobytes())
    if hit is not None:
        return hit, 0
    best, bd = None, None
    for c, b in _GLYPHS.items():
        d = int((cell ^ b).sum())
        if bd is None or d < bd:
            best, bd = c, d
    return best, bd


def solve_mask(mask):
    """-> (text, confidence) from an ink mask."""
    total = int(mask.sum())
    if total == 0:
        return "", 0.0

    chars, exact, covered = [], True, 0
    for cell in cells(mask):
        if not cell.any():
            exact = False                 # a blank cell is not a character
            continue
        c, d = _match(cell)
        chars.append(c)
        covered += int((cell & _GLYPHS[c]).sum())
        if d:
            exact = False

    text = "".join(chars)
    if not text:
        return "", 0.0

    # ink outside the six cells means the image is not what this reader models
    off = mask.copy()
    off[R0:R1, X0:X0 + PITCH * LENGTH] = False
    stray = bool(off.any())

    if exact and not stray and len(text) == LENGTH:
        return text, 1.0                  # a proof, for this generator

    # Cap strictly below 1.0. A single wrong glyph is only ~6 px against ~190
    # of ink, so the bare ratio would read 0.97 and sail through a 0.90 gate.
    return text, min(0.99, covered / total)


def solve_image(image):
    """-> (text, confidence). Accepts bytes, a path, or a PIL.Image."""
    return solve_mask(ink_mask(image))


def predict(images):
    """-> [(text, confidence)] for a batch. No model, so this is just a loop."""
    return [solve_image(im) for im in images]


# --------------------------------------------------------------------------
# generator
# --------------------------------------------------------------------------
def random_label(rng):
    return "".join(rng.choice(list(CHARSET), LENGTH))


def make_image(label, rng):
    """Compose a captcha the way the server does: noise first, then text.

    The layer order is load-bearing, not cosmetic — painting the noise last
    would let it hole the strokes, and every exactness claim in this module
    would become false. `tests/test_ngt.py` asserts the round trip that this
    ordering is what makes possible."""
    g = glyphs()
    a = np.full((H, W, 3), 255, np.uint8)

    ys = rng.integers(0, H, NOISE_N)
    xs = rng.integers(0, W, NOISE_N)
    cols = rng.integers(NOISE_LO, NOISE_HI + 1, (NOISE_N, 3))
    a[ys, xs] = cols

    for k, ch in enumerate(label[:LENGTH]):
        x0 = X0 + PITCH * k
        a[R0:R1, x0:x0 + CELL_W][g[ch]] = 0

    # no explicit mode: a (H, W, 3) uint8 array is RGB already, and passing one
    # is deprecated from Pillow 13
    return Image.fromarray(a)
