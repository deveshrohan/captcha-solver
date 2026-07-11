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

from solver import securimage as S
from solver.model import DigitCNN
from solver.pipeline import solve_image as _solve_gstat_path

import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_SECURIMAGE_MODEL = _os.path.join(_HERE, "securimage_model.pt")
_GSTAT_MODEL = _os.path.join(_HERE, "model.pt")

_lock = threading.Lock()
_securi = None
_gstat = None


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


def warmup(securimage=True, gstat=False):
    """Pre-load models at scraper startup so the first live solve isn't slow.
    Also runs one dummy forward to trigger lazy CUDA/oneDNN init. Call once."""
    if securimage:
        m = _load_securimage()
        with torch.no_grad():
            m(torch.zeros(1, 1, S.IN_H, S.IN_W))
    if gstat:
        _load_gstat()


def _to_pil(image):
    """Accept raw bytes, a filesystem path, or a PIL.Image -> grayscale PIL."""
    if isinstance(image, Image.Image):
        return image.convert("L")
    if isinstance(image, (bytes, bytearray)):
        return Image.open(io.BytesIO(image)).convert("L")
    if isinstance(image, str):
        return Image.open(image).convert("L")
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


def solve_bytes(image, kind=None):
    """Auto-route and solve. Returns (text, confidence, kind).

    kind is inferred from image width by default ('securimage' if width > 180,
    else 'gstat'); pass kind='securimage'|'gstat' to force it."""
    pil = _to_pil(image)
    if kind is None:
        kind = "securimage" if pil.width > 180 else "gstat"
    if kind == "securimage":
        text, conf = solve_securimage(pil)
    elif kind == "gstat":
        text, conf = solve_gstat(image if isinstance(image, str) else pil)
    else:
        raise ValueError(f"unknown kind {kind!r}")
    return text, conf, kind


def solve_with_retry(fetch, min_conf=0.90, max_tries=4, kind=None):
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
        if conf >= min_conf:
            return text, conf, k, i
    return text, conf, k, max_tries
