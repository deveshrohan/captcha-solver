"""Programmatic captcha-solving API (in-process, CPU).

Solves the two captcha types from a Python process with no network calls.
Only needs torch/numpy/pillow (scipy is training-only).

Typical use (in-process):

    from solver.api import solve_bytes, solve_with_retry

    # one-shot: you already have the image bytes from the HTTP response
    text, conf, kind = solve_bytes(resp.content)
    if conf < 0.90:
        ...            # low-confidence: re-fetch a fresh captcha and try again

    # or let the helper do the retry loop for you:
    def fetch():       # must return fresh captcha image bytes each call
        return session.get(CAPTCHA_URL, params={"_": time.time()}).content
    text, conf, kind = solve_with_retry(fetch, min_conf=0.90, max_tries=4)

Models are loaded once (lazily) and cached; calls are thread-safe for inference.
"""
import io
import threading

import numpy as np
import torch
from PIL import Image

from solver import bharatkosh as BK
from solver import epfo as E
from solver import gst as G
from solver import itat as T
from solver import kaveri as K
from solver import mca as M
from solver import ngt as N
from solver import securimage as S
from solver import udyam as Y
from solver.model import DigitCNN
from solver.pipeline import solve_image as _solve_gstat_path

import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_SECURIMAGE_MODEL = _os.path.join(_HERE, "securimage_model.pt")
_GSTAT_MODEL = _os.path.join(_HERE, "model.pt")
_GST_MODEL = _os.path.join(_HERE, "gst_model.pt")
_MCA_MODEL = _os.path.join(_HERE, "mca_model.pt")
_EPFO_MODEL = _os.path.join(_HERE, "epfo_model.pt")
_ITAT_MODEL = _os.path.join(_HERE, "itat_model.pt")
_BHARATKOSH_MODEL = _os.path.join(_HERE, "bharatkosh_model.pt")

_lock = threading.Lock()
_securi = None
_gstat = None
_gst = None
_mca = None
_epfo = None
_itat = None
_bharatkosh = None

def _route_120x40(pil):
    """Split the one size collision in the repo: gstat and NGT are both 120x40.

    gstat's palette is `#333` ink, `#ccc` lines and white — it cannot emit a
    pure-black pixel — while NGT's text *is* pure black. Measured: 0 black
    pixels across 1024 gstat images (1000 synthetic, 24 real), against 159..228
    across 60 NGT ones.

    The margin is structural rather than merely empirical, which is why a
    content test is safe to route on: NGT's thinnest possible word is six `r`
    at 21 ink px each, so it can never fall below 126.
    """
    return "ngt" if (np.asarray(pil.convert("RGB")) == 0).all(2).sum() >= 20 \
        else "gstat"


# (width, height) -> kind, or a callable(pil) -> kind where one size serves more
# than one captcha.
_SIZES = {
    (215, 80): "securimage",
    (182, 50): "gst",
    (200, 80): "mca",
    (120, 40): _route_120x40,          # gstat or ngt, decided by content
    (150, 50): "epfo",
    (200, 60): "kaveri",
    (225, 80): "udyam",
    (150, 42): "itat",
    (150, 40): "bharatkosh",           # ITAT is 150x42 -- no collision
}

# Characters that carry no distinguishing information in a given captcha style,
# so a read containing one is a coin flip no matter how good the model is.
#
# MCA: uppercase `I` and lowercase `l` are both plain vertical bars at the same
# height, and the portal validates case-sensitively. Measured on held-out reals,
# exact accuracy is 28.6% when the read contains one of these versus 83.8% when
# it does not — so the right move is to throw that captcha away and fetch a new
# one rather than submit a guess. Doing that costs ~1.4 fetches per solve and
# takes the success rate to ~97% within 3 fetches (vs 75% single-shot).
#
# Kaveri deliberately has no entry: `O` is 23px wide against `0` at 17, and `1`
# carries a diagonal flag against `I`'s bare 3px bar, so the pairs that cost MCA
# its accuracy are separable there by construction.
#
# Udyam has no entry either, for a stronger reason: the portal never issues `0`,
# `I` or `O` at all. Clustering 2100+ real glyphs yields exactly 33 classes and
# those three are not among them, so the homoglyph collision cannot arise.
#
# ITAT has no entry for two compounding reasons: `0 1 I O` never occur in its
# census, and the reader folds case (see solver/itat.py). Folding removes the
# case-homoglyphs (c/C, s/S, ...) and the excluded digits remove the digit/letter
# ones, so no ambiguous pair survives.
#
# NGT has no entry despite issuing the full `0-9a-z` set — including every
# homoglyph the others exclude. Its 36 glyph bitmaps are all distinct and the
# match is pixel-exact, so `0`/`o` (32 px apart) and `5`/`s` (35) are not coin
# flips but different dictionary keys. `1`, `l` and `i` sit closer at 6 px, and
# the reader answers that with exactness rather than a retry: a cell that does
# not match byte for byte cannot reach confidence 1.0 (see solver/ngt.py), so a
# `min_conf=1.0` gate already rejects exactly the reads a retry would be for.
AMBIGUOUS = {
    "mca": "lI",
}


def _load_securimage():
    global _securi
    if _securi is None:
        with _lock:
            if _securi is None:
                m = S.SecurimageCRNN()
                m.load_state_dict(torch.load(_SECURIMAGE_MODEL, map_location="cpu"))
                m.eval()
                _securi = m
    return _securi


def _load_gstat():
    global _gstat
    if _gstat is None:
        with _lock:
            if _gstat is None:
                m = DigitCNN()
                m.load_state_dict(torch.load(_GSTAT_MODEL, map_location="cpu"))
                m.eval()
                _gstat = m
    return _gstat


def _load_gst():
    global _gst
    if _gst is None:
        with _lock:
            if _gst is None:
                m = G.GstCRNN()
                m.load_state_dict(torch.load(_GST_MODEL, map_location="cpu"))
                m.eval()
                _gst = m
    return _gst


def _load_mca():
    global _mca
    if _mca is None:
        with _lock:
            if _mca is None:
                m = M.McaCRNN()
                m.load_state_dict(torch.load(_MCA_MODEL, map_location="cpu"))
                m.eval()
                _mca = m
    return _mca


def _load_epfo():
    global _epfo
    if _epfo is None:
        with _lock:
            if _epfo is None:
                m = E.EpfoCRNN()
                m.load_state_dict(torch.load(_EPFO_MODEL, map_location="cpu"))
                m.eval()
                _epfo = m
    return _epfo


def _load_itat():
    global _itat
    if _itat is None:
        with _lock:
            if _itat is None:
                m = T.ItatCRNN()
                m.load_state_dict(torch.load(_ITAT_MODEL, map_location="cpu"))
                m.eval()
                _itat = m
    return _itat


def _load_bharatkosh():
    global _bharatkosh
    if _bharatkosh is None:
        with _lock:
            if _bharatkosh is None:
                m = BK.BharatkoshCRNN()
                m.load_state_dict(torch.load(_BHARATKOSH_MODEL, map_location="cpu"))
                m.eval()
                _bharatkosh = m
    return _bharatkosh


def warmup(securimage=True, gstat=False, gst=False, mca=False, epfo=False,
           kaveri=False, udyam=False, itat=False, ngt=False, bharatkosh=False):
    """Pre-load models at scraper startup so the first live solve isn't slow.
    Also runs one dummy forward to trigger lazy CUDA/oneDNN init. Call once.

    `kaveri`, `udyam` and `ngt` have no model — they only parse their template
    libraries, which is cheap; the flags exist so callers can enable every kind
    uniformly."""
    if securimage:
        m = _load_securimage()
        with torch.no_grad():
            m(torch.zeros(1, 1, S.IN_H, S.IN_W))
    if gstat:
        _load_gstat()
    if gst:
        m = _load_gst()
        with torch.no_grad():
            m(torch.zeros(1, 3, G.IN_H, G.IN_W))
    if mca:
        m = _load_mca()
        with torch.no_grad():
            m(torch.zeros(1, 1, M.IN_H, M.IN_W))
    if epfo:
        m = _load_epfo()
        with torch.no_grad():
            m(torch.zeros(1, 1, E.IN_H, E.IN_W))
    if itat:
        m = _load_itat()
        with torch.no_grad():
            m(torch.zeros(1, 1, T.IN_H, T.IN_W))
    if bharatkosh:
        m = _load_bharatkosh()
        with torch.no_grad():
            m(torch.zeros(1, 1, BK.IN_H, BK.IN_W))
    if kaveri:
        K.glyphs()
    if udyam:
        Y.classes()
    if ngt:
        N.glyphs()


def _to_pil(image):
    """Accept raw bytes, a filesystem path, or a PIL.Image -> grayscale PIL."""
    return _to_pil_rgb(image).convert("L")


def _to_pil_rgb(image):
    """Accept raw bytes, a filesystem path, or a PIL.Image -> RGB PIL.

    RGB rather than grayscale because the GST captcha's noise line is pure red
    and the MCA captcha's ink is pure black: both routers need colour."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (bytes, bytearray)):
        return Image.open(io.BytesIO(image)).convert("RGB")
    if isinstance(image, str):
        return Image.open(image).convert("RGB")
    raise TypeError("image must be bytes, a path str, or a PIL.Image")


def solve_securimage(image):
    """Solve an eCourts securimage captcha. Returns (text, confidence).

    `image` may be bytes / path / PIL.Image. confidence is the beam decoder's
    geometric-mean per-character probability in [0, 1]."""
    pil = _to_pil(image).resize((S.IN_W, S.IN_H), Image.BILINEAR)
    arr = np.asarray(pil, np.uint8)[None]
    text, conf = S.predict_ctc_beam(_load_securimage(), arr, "cpu")
    return text[0], float(conf[0])


def solve_gstat(image):
    """Solve a gstat numeric captcha. Returns (text, confidence).

    confidence is the minimum per-digit softmax probability (a reliability gate)."""
    pil = _to_pil(image)
    # the gstat pipeline works from a file path; write bytes/PIL to a temp buffer
    if isinstance(image, str):
        text, confs = _solve_gstat_path(_load_gstat(), image)
    else:
        import tempfile, os
        fd, path = tempfile.mkstemp(suffix=".png")
        try:
            os.close(fd)
            pil.save(path)
            text, confs = _solve_gstat_path(_load_gstat(), path)
        finally:
            os.remove(path)
    return text, float(min(confs)) if confs else 0.0


def solve_gst(image):
    """Solve a GST portal captcha (182x50, 6 digits). Returns (text, confidence).

    confidence is the beam decoder's geometric-mean per-character probability."""
    pil = _to_pil_rgb(image)
    arr = G.load_real(pil)[None]
    text, conf = G.predict(_load_gst(), arr, "cpu")
    return text[0], float(conf[0])


def solve_mca(image):
    """Solve an MCA portal captcha (200x80, 6 mixed-case alnum chars).
    Returns (text, confidence)."""
    pil = _to_pil_rgb(image)
    arr = M.load_real(pil)[None]
    text, conf = M.predict(_load_mca(), arr, "cpu")
    return text[0], float(conf[0])


def solve_epfo(image):
    """Solve an EPFO portal captcha (150x50, 5 uppercase-alnum chars).
    Returns (text, confidence)."""
    pil = _to_pil_rgb(image)
    arr = E.load_real(pil)[None]
    text, conf = E.predict(_load_epfo(), arr, "cpu")
    return text[0], float(conf[0])


def solve_itat(image):
    """Solve an ITAT portal captcha (150x42, 6 mixed-case alnum chars).
    Returns (text, confidence).

    The read is CASE-INSENSITIVE: the text is uppercase and matches the display
    up to case. confidence is the beam decoder's geometric-mean per-character
    probability."""
    pil = _to_pil_rgb(image)
    arr = T.load_real(pil)[None]
    text, conf = T.predict(_load_itat(), arr, "cpu")
    return text[0], float(conf[0])


def solve_kaveri(image):
    """Solve a Kaveri portal captcha (200x60, 6 uppercase-alnum chars).
    Returns (text, confidence).

    No model is involved: the image is a deterministic composition of 36 known
    sprites, so this is an exact cover of the ink mask. Confidence is 1.0 when a
    cover exists and `1 - mismatch/ink` otherwise — a real measure of how well
    the sprite library explains the image, not a softmax that stays high when
    the generator changes."""
    return K.solve_image(image)


def solve_udyam(image):
    """Solve a Udyam portal captcha (225x80, 6 uppercase-alnum chars).
    Returns (text, confidence).

    No model is involved: the glyphs are a fixed set of 34 bitmaps, so this is a
    cover of the ink by templates. Unlike Kaveri the noise lines are drawn *over*
    the text, but they are alpha-blended, so the crossed pixels stay dark and are
    recovered by a luminance cut rather than lost. Confidence is the fraction of
    the ink the chosen templates explain — never exactly 1.0, because the text is
    antialiased and no cover is pixel-exact."""
    return Y.solve_image(image)


def solve_ngt(image):
    """Solve an NGT captcha (120x40, 6 lowercase-alnum chars).
    Returns (text, confidence).

    No model is involved: the glyphs are a fixed bitmap font blitted on a fixed
    grid, and the noise is painted *underneath* the text, so the ink mask is
    exact and the read is six dictionary lookups.

    Confidence is 1.0 only when all six cells match a template byte for byte and
    no ink lies outside the grid — for this generator that is a proof the read
    is correct, not a softmax that happens to be saturated. Anything less means
    the generator moved, so the gate to use here is `min_conf=1.0`."""
    return N.solve_image(image)


def solve_bharatkosh(image):
    """Solve ONE render of a Bharatkosh captcha (150x40, 6 mixed-case alnum
    chars, case-sensitive read). Returns (text, confidence).

    Prefer `solve_bharatkosh_group`: a single render loses six rows to the two
    opaque bars, and what they hide is sometimes the whole difference between two
    characters (a tilted `A` with one foot in the bar is a `4`)."""
    return solve_bharatkosh_group([image])


def solve_bharatkosh_group(images):
    """Solve a Bharatkosh captcha from several renders of the SAME answer.
    Returns (text, confidence).

    `GenerateCaptcha?New=0` re-renders the session's existing text with fresh
    fonts, rotations and colours, so fetch once with `New=1`, then `New=0` a few
    more times on the same session and pass every image here. Each candidate
    word is scored by its summed CTC log-likelihood over all renders; confidence
    is the winner's posterior over the candidates, so it rises as renders agree.

    That a re-render leaves the submittable answer unchanged is inferred from
    the text staying constant, not confirmed by posting a form."""
    arr = np.stack([BK.load_real(_to_pil_rgb(im)) for im in images])
    return BK.predict_group(_load_bharatkosh(), arr, "cpu")


def solve_bharatkosh_with_rerender(fetch, min_conf=0.90, max_renders=5,
                                   max_texts=3):
    """Fetch-and-solve for Bharatkosh, spending re-renders before new texts.

    `fetch(new)` returns captcha image bytes through the session the form will
    be submitted on: `new=True` must request `?New=1` (a new text), `new=False`
    `?New=0` (the same text, re-rendered). Returns (text, confidence, renders,
    texts).

    Re-rendering never changes the session's answer, so unlike
    `solve_with_retry` a low-confidence read costs another look at the SAME
    word rather than a gamble on a new one. Only when `max_renders` looks still
    do not clear `min_conf` is a new text drawn. The returned read always
    belongs to the text the session currently holds."""
    text, conf, n = "", 0.0, 0
    for t in range(1, max_texts + 1):
        images = [fetch(True)]
        while True:
            text, conf = solve_bharatkosh_group(images)
            n = len(images)
            if conf >= min_conf or n >= max_renders:
                break
            images.append(fetch(False))
        if conf >= min_conf:
            return text, conf, n, t
    return text, conf, n, max_texts


def solve_bytes(image, kind=None):
    """Auto-route and solve. Returns (text, confidence, kind).

    kind is inferred from the image's native size; pass kind explicitly to force
    it. Sizes are distinct except 120x40, which serves both gstat and NGT and is
    resolved by a content test (`_route_120x40`)."""
    pil = _to_pil_rgb(image)
    if kind is None:
        kind = _SIZES.get((pil.width, pil.height))
        if callable(kind):                    # shared size: decide by content
            kind = kind(pil)
        if kind is None:                      # unknown size: fall back to width
            kind = "securimage" if pil.width > 180 else "gstat"
    if kind == "securimage":
        text, conf = solve_securimage(pil)
    elif kind == "gst":
        text, conf = solve_gst(pil)
    elif kind == "mca":
        text, conf = solve_mca(pil)
    elif kind == "epfo":
        text, conf = solve_epfo(pil)
    elif kind == "itat":
        text, conf = solve_itat(pil)
    elif kind == "bharatkosh":
        text, conf = solve_bharatkosh(pil)
    elif kind == "kaveri":
        # Pass the ORIGINAL object, not `pil`: this captcha is RGBA and PIL maps
        # transparent pixels to (0,0,0) on convert("RGB"), which is exactly the
        # value the ink test looks for. `kaveri.ink_mask` also crops the
        # transparent band away, so either input is read correctly — but keeping
        # the alpha channel means the crop is a second line of defence, not the
        # only one.
        text, conf = solve_kaveri(image)
    elif kind == "udyam":
        text, conf = solve_udyam(pil)
    elif kind == "ngt":
        text, conf = solve_ngt(pil)
    elif kind == "gstat":
        text, conf = solve_gstat(image if isinstance(image, str) else pil.convert("L"))
    else:
        raise ValueError(f"unknown kind {kind!r}")
    return text, conf, kind


def solve_with_retry(fetch, min_conf=0.90, max_tries=4, kind=None,
                     avoid_ambiguous=True):
    """Fetch-and-solve with confidence-gated retry.

    `fetch` is a zero-arg callable returning FRESH captcha image bytes each call.
    Returns (text, confidence, kind, tries).

    IMPORTANT — session-bound captchas (Securimage/eCourts): each captcha GET
    overwrites the code stored in the server session, so only the MOST RECENTLY
    fetched captcha is valid to submit. Therefore:
      * `fetch` MUST re-request through the SAME session/cookies your form
        submission will use, and
      * this helper returns the LAST fetched read when it never reaches min_conf
        (not the historical best), because that is the code the session holds.
      * submit the returned code with that same session BEFORE fetching again.
    Returns as soon as a read is confident (that read == current session code)."""
    text, conf, k = "", 0.0, kind
    for i in range(1, max_tries + 1):
        text, conf, k = solve_bytes(fetch(), kind=kind)   # last read == session code
        if conf < min_conf:
            continue
        if avoid_ambiguous and set(text) & set(AMBIGUOUS.get(k, "")):
            continue          # unresolvable glyph in the read -> take a fresh captcha
        return text, conf, k, i
    return text, conf, k, max_tries
