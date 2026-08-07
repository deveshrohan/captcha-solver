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

from solver import gst as G
from solver import mca as M
from solver import securimage as S
from solver.model import DigitCNN
from solver.pipeline import solve_image as _solve_gstat_path

import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_SECURIMAGE_MODEL = _os.path.join(_HERE, "securimage_model.pt")
_GSTAT_MODEL = _os.path.join(_HERE, "model.pt")
_GST_MODEL = _os.path.join(_HERE, "gst_model.pt")
_MCA_MODEL = _os.path.join(_HERE, "mca_model.pt")

_lock = threading.Lock()
_securi = None
_gstat = None
_gst = None
_mca = None

# (width, height) -> kind. Every supported captcha has a distinct native size.
_SIZES = {
    (215, 80): "securimage",
    (182, 50): "gst",
    (200, 80): "mca",
    (120, 40): "gstat",
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


def warmup(securimage=True, gstat=False, gst=False, mca=False):
    """Pre-load models at scraper startup so the first live solve isn't slow.
    Also runs one dummy forward to trigger lazy CUDA/oneDNN init. Call once."""
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


def solve_bytes(image, kind=None):
    """Auto-route and solve. Returns (text, confidence, kind).

    kind is inferred from the image's native size (every supported captcha has a
    distinct one); pass kind explicitly to force it."""
    pil = _to_pil_rgb(image)
    if kind is None:
        kind = _SIZES.get((pil.width, pil.height))
        if kind is None:                      # unknown size: fall back to width
            kind = "securimage" if pil.width > 180 else "gstat"
    if kind == "securimage":
        text, conf = solve_securimage(pil)
    elif kind == "gst":
        text, conf = solve_gst(pil)
    elif kind == "mca":
        text, conf = solve_mca(pil)
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
