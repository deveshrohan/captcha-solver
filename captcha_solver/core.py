"""Public API: identify a captcha, read it, and retry until it is solved."""
import inspect
import io
from dataclasses import dataclass, field, replace
from typing import Any, Callable, List, Optional

import importlib.util

# Installed wheel: the engine ships inside as captcha_solver.engine. Source
# checkout: it is ./solver. Decide by whether the engine package EXISTS, so a
# genuine import error inside it (e.g. numpy missing) is not masked.
if importlib.util.find_spec(__package__ + ".engine") is not None:
    from .engine import api as _api
else:
    from solver import api as _api

from PIL import Image, UnidentifiedImageError

__all__ = ["KINDS", "Result", "CaptchaError", "UnsupportedCaptcha",
           "CaptchaUnsolved", "identify", "solve", "solve_with_retry",
           "solve_url", "warmup"]


# kind -> what it is. Size is the identifying fingerprint; length is how many
# characters a valid read has (None: not fixed).
KINDS = {
    "securimage": dict(size=(215, 80), length=6, what="eCourts Securimage"),
    "gstat":      dict(size=(120, 40), length=None, what="gstat numeric"),
    "ngt":        dict(size=(120, 40), length=6, what="National Green Tribunal"),
    "gst":        dict(size=(182, 50), length=6, what="GST portal"),
    "mca":        dict(size=(200, 80), length=6, what="MCA portal"),
    "epfo":       dict(size=(150, 50), length=5, what="EPFO portal"),
    "kaveri":     dict(size=(200, 60), length=6, what="Kaveri (Karnataka registration)"),
    "udyam":      dict(size=(225, 80), length=6, what="Udyam registration"),
    "itat":       dict(size=(150, 42), length=6, what="ITAT"),
    "bharatkosh": dict(size=(150, 40), length=6, what="Bharatkosh (NTRP)"),
    "hrmut":      dict(size=(60, 20), length=6, what="Haryana Jamabandi mutation"),
    "mhigr":      dict(size=(100, 30), length=6, what="Maharashtra IGR"),
    "gujhc":      dict(size=(200, 50), length=5, what="Gujarat High Court"),
    "madhc":      dict(size=(120, 60), length=6, what="Madras High Court"),
    "calhc":      dict(size=(120, 36), length=6, what="Calcutta High Court"),
    "karhc":      dict(size=(100, 20), length=6, what="Karnataka High Court"),
}
for _k, _v in KINDS.items():
    _v["gate"] = _api.GATES[_k]


class CaptchaError(Exception):
    """Base class for everything this package raises."""


class UnsupportedCaptcha(CaptchaError):
    """The image does not match any captcha this package can read."""


class CaptchaUnsolved(CaptchaError):
    """Every attempt was rejected. `.attempts` lists each Result, in order."""

    def __init__(self, attempts):
        self.attempts = attempts
        why = ", ".join(a.reason or "accepted" for a in attempts)
        super().__init__("captcha not solved after %d attempt(s): %s"
                         % (len(attempts), why))


@dataclass(frozen=True)
class Result:
    text: str
    confidence: float
    kind: str
    accepted: bool                 # safe to submit: passed every check
    reason: Optional[str] = None   # why not accepted / why it failed
    attempts: int = 1              # fetches used (solve_with_retry)
    response: Any = field(default=None, repr=False)   # what submit() returned

    def __str__(self):
        return self.text


def _open(image):
    if isinstance(image, Image.Image):
        return image
    try:
        if isinstance(image, (bytes, bytearray)):
            return Image.open(io.BytesIO(image))
        if isinstance(image, str):
            return Image.open(image)
        if hasattr(image, "read"):
            return Image.open(io.BytesIO(image.read()))
    except UnidentifiedImageError as e:
        raise UnsupportedCaptcha("not an image: %s" % e) from None
    raise TypeError("image must be bytes, a file path, a file object or a PIL.Image")


def identify(image) -> str:
    """Which captcha is this? Returns a key of KINDS, or raises
    UnsupportedCaptcha. Strict: an unknown size is an error, never a guess."""
    pil = _open(image)
    kind = _api._SIZES.get(pil.size)
    if callable(kind):                       # two kinds share a size
        kind = kind(pil.convert("RGB"))
    if kind is None:
        raise UnsupportedCaptcha(
            "no reader for %dx%d images (supported sizes: %s)" % (
                pil.size + (", ".join(sorted({"%dx%d" % v["size"]
                                              for v in KINDS.values()})),)))
    return kind


def _judge(text, conf, kind, min_conf):
    gate = KINDS[kind]["gate"] if min_conf is None else min_conf
    if kind == "hrmut":                      # its retry floor is enforced
        gate = max(gate, KINDS["hrmut"]["gate"])
    want = KINDS[kind]["length"]
    if not text:
        return False, "no-text"
    if want is not None and len(text) != want:
        return False, "wrong-length"
    if set(text) & set(_api.AMBIGUOUS.get(kind, "")):
        return False, "ambiguous-glyph"      # e.g. MCA I/l: a coin flip
    if conf < gate:
        return False, "low-confidence"
    return True, None


def solve(image, kind: Optional[str] = None, min_conf: Optional[float] = None) -> Result:
    """Identify and read one captcha image (bytes, path, file object or PIL).

    `accepted` says whether the read passed every check: confidence at or
    above the kind's gate (or `min_conf`), the right length, and no glyph
    the generator makes indistinguishable. A read that is not accepted is
    still returned -- `solve_with_retry` is what acts on it."""
    if hasattr(image, "read"):
        image = image.read()
    if kind is None:
        kind = identify(image)
    elif kind not in KINDS:
        raise ValueError("unknown kind %r; expected one of %s" % (kind, sorted(KINDS)))
    # hand the engine the original object: Kaveri needs its alpha channel
    text, conf, kind = _api.solve_bytes(image, kind=kind)
    ok, reason = _judge(text, float(conf), kind, min_conf)
    return Result(text, float(conf), kind, ok, reason)


def _takes_new(fetch):
    try:
        return "new" in inspect.signature(fetch).parameters
    except (TypeError, ValueError):
        return False


def solve_with_retry(fetch: Callable, submit: Optional[Callable] = None,
                     kind: Optional[str] = None, max_attempts: int = 5,
                     min_conf: Optional[float] = None, max_renders: int = 5,
                     raise_on_failure: bool = True) -> Result:
    """Fetch, read and (optionally) submit until a captcha is solved.

    fetch()        -> image bytes of a FRESH captcha. It must go through the
                      same session/cookies the form is submitted with: most
                      of these portals keep only the last captcha served.
    submit(text)   -> optional. Submit the form with `text`; return something
                      truthy if the site accepted it (that value comes back as
                      `Result.response`), or False/None if it said the captcha
                      was wrong. Without `submit` only reads that look wrong
                      (low confidence, ambiguous glyph, wrong length) are
                      retried; with it, confidently wrong reads are too.

    A read that fails a check is never submitted: a fresh captcha is fetched
    instead. The returned Result always belongs to the captcha fetched last,
    which is the one the server session holds.

    Bharatkosh: if `fetch` takes a `new` argument, `fetch(new=True)` must draw
    a new answer and `fetch(new=False)` re-render the current one
    (`GenerateCaptcha?New=1` / `New=0`); up to `max_renders` renders of one
    answer are voted over before a new answer is drawn.

    Raises CaptchaUnsolved after `max_attempts` fetches of a new answer
    (unless raise_on_failure=False, which returns the last Result)."""
    rerender = _takes_new(fetch)
    history: List[Result] = []
    for attempt in range(1, max_attempts + 1):
        image = fetch(new=True) if rerender else fetch()
        r = solve(image, kind=kind, min_conf=min_conf)
        if r.kind == "bharatkosh" and rerender and not r.accepted:
            images = [image]
            while not r.accepted and len(images) < max_renders:
                images.append(fetch(new=False))
                text, conf = _api.solve_bharatkosh_group(images)
                ok, reason = _judge(text, float(conf), "bharatkosh", min_conf)
                r = Result(text, float(conf), "bharatkosh", ok, reason)
        r = replace(r, attempts=attempt)
        if r.accepted and submit is not None:
            response = submit(r.text)
            if response is False or response is None:
                r = replace(r, accepted=False, reason="rejected-by-site")
            else:
                r = replace(r, response=response)
        history.append(r)
        if r.accepted:
            return r
    if raise_on_failure:
        raise CaptchaUnsolved(history)
    return history[-1]


def solve_url(url: str, session=None, submit: Optional[Callable] = None,
              timeout: float = 30, **kwargs) -> Result:
    """solve_with_retry over a plain GET of `url` through a requests.Session.

    Pass the session your form submission uses, so the answer is bound to it
    (a throwaway session is created otherwise). Needs `requests`. Include any
    portal-specific URL details yourself, e.g. Gujarat HC needs `?ct=S`.
    Extra keyword arguments go to solve_with_retry."""
    import requests
    s = session or requests.Session()

    def fetch():
        r = s.get(url, timeout=timeout)
        r.raise_for_status()
        return r.content

    return solve_with_retry(fetch, submit=submit, **kwargs)


def warmup(kinds=None):
    """Load models/templates now rather than on first use. `kinds`: iterable
    of KINDS keys (default: every kind -- several hundred MB of models)."""
    kinds = set(KINDS) if kinds is None else set(kinds)
    unknown = kinds - set(KINDS)
    if unknown:
        raise ValueError("unknown kinds: %s" % sorted(unknown))
    _api.warmup(**{k: (k in kinds) for k in KINDS})
