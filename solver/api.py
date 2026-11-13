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

from . import bharatkosh as BK
from . import epfo as E
from . import gst as G
from . import itat as T
from . import itat2 as T2
from . import kaveri as K
from . import mca as M
from . import mca2 as M2
from . import ngt as N
from . import securimage as S
from . import udyam as Y
from . import hrmut as H
from . import mhigr as I
from . import calhc as CH
from . import gujhc as GH
from . import madhc as MH
from . import karhc as KH
from . import cersai as CE
from . import aphc as AP
from . import k2 as K2
from . import mcacdm as CDM
from . import cestat as CT
from . import dcourts as DC
from . import tg_rera as TGR
from . import ga_rera as GAR
from . import up_rera as UPR
from . import sci as SCI
# The RERA readers (hrrera, pbrera, hprera) are imported on first use:
# they shell out to Tesseract and hprera needs scipy, neither of which is a
# package dependency, so `import solver.api` must not pull them in.
from .model import DigitCNN
from .pipeline import solve_image as _solve_gstat_path

import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_SECURIMAGE_MODEL = _os.path.join(_HERE, "securimage_model.pt")
_GSTAT_MODEL = _os.path.join(_HERE, "model.pt")
_GST_MODEL = _os.path.join(_HERE, "gst_model.pt")
_MCA_MODEL = _os.path.join(_HERE, "mca_model.pt")
_MCA2_MODEL = _os.path.join(_HERE, "mca2_model.pt")
_EPFO_MODEL = _os.path.join(_HERE, "epfo_model.pt")
_ITAT_MODEL = _os.path.join(_HERE, "itat_model.pt")
_ITAT2_MODEL = _os.path.join(_HERE, "itat2_model.pt")
_BHARATKOSH_MODEL = _os.path.join(_HERE, "bharatkosh_model.pt")
_CALHC_MODEL = _os.path.join(_HERE, "calhc_model.pt")

_lock = threading.Lock()
_securi = None
_gstat = None
_gst = None
_mca = None
_mca2 = None
_epfo = None
_itat = None
_itat2 = None
_bharatkosh = None
_calhc = None

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


def _route_200x50(pil):
    """Split the second size collision: Gujarat HC and Karnataka K2 are both
    200x50 SimpleCaptcha renders.

    K2 is an RGBA PNG with its ink in the alpha channel and the background
    fully transparent. Gujarat HC is opaque RGB on a light background, with a
    mean of at least 176 across all 300 harvested images. A K2 image whose
    alpha was stripped by `convert("RGB")` comes out entirely black, so it
    still routes to k2 (where it reads as unreadable) rather than to gujhc.
    """
    if pil.mode in ("RGBA", "LA", "PA") or "transparency" in pil.info:
        if np.asarray(pil.convert("RGBA"))[:, :, 3].min() < 255:
            return "k2"
    return "k2" if np.asarray(pil.convert("L")).mean() < 100 else "gujhc"


def _route_200x80(pil):
    """MCA changed generators in Nov 2026 without changing size. The old image
    has exactly three flat colours (230 / 150 / 0); the new one blurs coloured
    ink, lines and speckle into ~1700. See solver/mca2.py."""
    return "mca2" if M2.looks_like_mca2(pil) else "mca"


# (width, height) -> kind, or a callable(pil) -> kind where one size serves more
# than one captcha. Callables get the image as opened, alpha intact.
_SIZES = {
    (60, 20): "hrmut",
    (100, 30): "mhigr",
    (215, 80): "securimage",
    (182, 50): "gst",
    (200, 80): _route_200x80,          # MCA: old 3-colour format or Nov 2026 (mca2)
    (120, 40): _route_120x40,          # gstat or ngt, decided by content
    (150, 50): "epfo",
    (200, 60): "kaveri",
    (225, 80): "udyam",
    (150, 42): "itat",                 # ITAT before Oct 2026
    (150, 38): "itat2",                # ITAT since Oct 2026
    (150, 40): "bharatkosh",           # ITAT is 150x42 -- no collision
    (200, 50): _route_200x50,          # Gujarat HC or Karnataka K2, by alpha
    (120, 60): "madhc",                # Madras HC
    (120, 36): "calhc",                # Calcutta HC
    (100, 20): "karhc",                # Karnataka HC (MH IGR is 100x30)
    (200, 45): GAR.route_200x45,       # CERSAI (green ink) or Goa RERA
    (110, 33): "aphc",                 # Andhra Pradesh HC (sum captcha)
    (130, 40): "hrrera",               # HR / J&K / Assam RERA (NIC PHP GD)
    (100, 38): "pbrera",               # Punjab RERA, also serves Chandigarh
    (130, 35): "hprera",               # HP RERA document downloads
    (120, 38): "mcacdm",               # MCA CDM export (calhc is 120x36)
    (100, 35): "cestat",               # CESTAT (shows 4 of its 6 characters)
    (225, 85): "dcourts",              # district courts, securimage-wp (Udyam is 225x80)
    (158, 48): "tgrera",               # Telangana RERA
    (150, 35): "uprera",               # Uttar Pradesh RERA (MSCaptcha)
    (220, 80): "sci",                  # Supreme Court arithmetic (securimage is 215x80)
}

# Kinds whose reader needs something outside the package's dependencies.
# hrrera only falls back to Tesseract on a low-confidence glyph read, and
# that read can never pass its gate, so it works without Tesseract.
REQUIRES = {
    "pbrera": ("tesseract",),
    "hprera": ("tesseract", "scipy"),
}

# Recommended `min_conf` per kind. For the newer readers it was chosen on the
# validation split (never the test split); for the older ones it is the value
# each README section recommends. Cover readers report fidelity, so their
# gates sit near 1.0. For the model readers (securimage, gst, mca, epfo, itat,
# bharatkosh, calhc) confidence is NOT accuracy -- see the README -- so pair
# the gate with submit feedback (captcha_solver.solve_with_retry(submit=...)).
GATES = {
    "securimage": 0.90,
    "gstat": 0.90,
    "gst": 0.90,
    "mca": 0.90,
    "mca2": 0.80,            # val split: 25/25 right, lowest 0.845; test 59/60, miss at 0.45
    "epfo": 0.90,
    "itat": 0.90,
    "itat2": 0.90,
    "bharatkosh": 0.90,
    "kaveri": 0.99,          # 1.0 whenever an exact sprite cover exists
    "udyam": 0.85,           # real 0.87-0.99; changed fonts score <= 0.75
    "ngt": 1.0,              # only a byte-exact match reaches 1.0
    "hrmut": 0.97,
    "mhigr": 0.90,
    "madhc": 0.99,
    "gujhc": 0.90,
    "calhc": 0.90,
    "karhc": 0.70,
    "cersai": 0.95,          # source's gate: 39/40 held-out accepted, all right
    "aphc": 0.50,            # worst digit mismatch <= 30 px; live reads 0.90-0.98
    "k2": 0.90,              # exact cover reads score 1.0
    "hrrera": 0.95,          # glyph path only; a Tesseract fallback scores 0.5
    "pbrera": 0.50,          # 2 of 4 Tesseract variants agree
    "hprera": 0.30,          # uncalibrated: 27/30 live, no confidences recorded
    "mcacdm": 0.99,          # exact GD font-5 cells: 75/75 held-out at 1.0
    "cestat": 0.35,          # margin; real reads >= 0.40, shifted text 0
    "dcourts": 0.50,         # val split; 39/40 held-out voted, 40/40 within 2 guesses
    "tgrera": 0.85,          # lowest correct 0.906; one blanked letter <= 0.817
    "garera": 0.70,          # 51/53 held-out accepted, all correct
    "uprera": 0.70,          # 34/40 held-out exact; 33 of 34 accepted correct
    "sci": 0.60,             # 52/60 held-out renders accepted, all correct
}


def solve_dcourts_group(images):
    """District-courts captcha from several renders of the SAME code (re-fetch
    `/?_siwp_captcha&id=<scid>` with the same scid). -> (text, confidence)."""
    return DC.solve_group(images)


# Kinds whose server re-renders the SAME answer on request: several renders
# are voted over. -> fn(images) -> (text, confidence)
GROUPS = {"bharatkosh": None, "dcourts": solve_dcourts_group,      # filled below
          "sci": lambda images: SCI.solve_group(images)}

# Kinds whose code survives a wrong guess, so the runner-up readings of the
# same renders can be submitted before fetching again.
STICKY = {"dcourts": lambda images: DC.rank_group(images)}

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


def _load_mca2():
    global _mca2
    if _mca2 is None:
        with _lock:
            if _mca2 is None:
                m = M2.Mca2SlotCNN()
                m.load_state_dict(torch.load(_MCA2_MODEL, map_location="cpu"))
                m.eval()
                _mca2 = m
    return _mca2


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


def _load_itat2():
    global _itat2
    if _itat2 is None:
        with _lock:
            if _itat2 is None:
                m = T2.Itat2CRNN()
                m.load_state_dict(torch.load(_ITAT2_MODEL, map_location="cpu"))
                m.eval()
                _itat2 = m
    return _itat2


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


def _load_calhc():
    global _calhc
    if _calhc is None:
        with _lock:
            if _calhc is None:
                m = CH.CalhcCRNN()
                m.load_state_dict(torch.load(_CALHC_MODEL, map_location="cpu"))
                m.eval()
                _calhc = m
    return _calhc


def warmup(securimage=True, gstat=False, gst=False, mca=False, epfo=False,
           kaveri=False, udyam=False, itat=False, ngt=False, bharatkosh=False, hrmut=False, mhigr=False,
           gujhc=False, madhc=False, calhc=False, karhc=False, itat2=False,
           cersai=False, aphc=False, k2=False, hrrera=False, pbrera=False,
           hprera=False, mcacdm=False, cestat=False, dcourts=False,
           tgrera=False, garera=False, uprera=False, sci=False, mca2=False):
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
    if mca2:
        m = _load_mca2()
        with torch.no_grad():
            m(torch.zeros(6, 1, M2.IN_H, 2 * M2.HALF))
    if epfo:
        m = _load_epfo()
        with torch.no_grad():
            m(torch.zeros(1, 1, E.IN_H, E.IN_W))
    if itat:
        m = _load_itat()
        with torch.no_grad():
            m(torch.zeros(1, 1, T.IN_H, T.IN_W))
    if itat2:
        m = _load_itat2()
        with torch.no_grad():
            m(torch.zeros(1, 1, T2.IN_H, T2.IN_W))
    if bharatkosh:
        m = _load_bharatkosh()
        with torch.no_grad():
            m(torch.zeros(1, 1, BK.IN_H, BK.IN_W))
    if hrmut or mhigr:
        from .fixedgreen import library
        if hrmut: library("hrmut")
        if mhigr: library("mhigr")
    if gujhc:
        GH.templates()
    if madhc:
        MH.templates()
    if karhc:
        KH.glyphs()
    if calhc:
        m = _load_calhc()
        with torch.no_grad():
            m(torch.zeros(1, 1, CH.IN_H, CH.IN_W))
    if kaveri:
        K.glyphs()
    if udyam:
        Y.classes()
    if ngt:
        N.glyphs()
    if cersai:
        CE.templates()
    if aphc:
        AP.templates()
    if k2:
        K2.templates()
    if mcacdm:
        CDM.glyphs()
    if cestat:
        CT._cells()
    if dcourts:
        DC.model()
    if tgrera:
        TGR.templates()
    if garera:
        GAR.model()
    if uprera:
        UPR.model()
    if sci:
        SCI.templates()
    if hrrera:
        from . import hrrera as HR
        HR.glyphs()
    if hprera and _available("hprera"):
        from . import hprera as HP
        if HP.FONT.exists():
            HP.templates()
    # pbrera is Tesseract only: nothing to load


def _available(kind):
    """True when everything REQUIRES lists for `kind` is installed."""
    import importlib.util
    import shutil
    for need in REQUIRES.get(kind, ()):
        if need == "tesseract":
            if shutil.which("tesseract") is None:
                return False
        elif importlib.util.find_spec(need) is None:
            return False
    return True


def _solve_rera(kind, image):
    """Run one of the RERA readers. A missing Tesseract binary is an
    unreadable captcha, not a crash."""
    try:
        if kind == "hrrera":
            from . import hrrera as HR
            return HR.solve_image(image)
        if kind == "pbrera":
            from . import pbrera as PB
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            return PB.solve(buf.getvalue())
        from . import hprera as HP
        return HP.solve_image(image)
    except FileNotFoundError:          # tesseract not on PATH
        return "", 0.0


def _to_pil(image):
    """Accept raw bytes, a filesystem path, or a PIL.Image -> grayscale PIL."""
    return _to_pil_rgb(image).convert("L")


def _to_pil_any(image):
    """Accept raw bytes, a filesystem path, or a PIL.Image -> PIL, mode as
    stored (K2's ink and Kaveri's background live in the alpha channel)."""
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, (bytes, bytearray)):
        return Image.open(io.BytesIO(image))
    if isinstance(image, str):
        return Image.open(image)
    raise TypeError("image must be bytes, a path str, or a PIL.Image")


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


def solve_mca2(image):
    """Solve an MCA portal captcha in the Nov 2026 format (200x80, 6 chars
    from `234679ABCDEFGHJKLMNPQRSTUVWXYZ`). Returns (text, confidence)."""
    arr = M2.load_real(_to_pil_rgb(image))[None]
    text, conf = M2.predict(_load_mca2(), arr, "cpu")
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


def solve_itat2(image):
    """Solve an ITAT captcha in the format served since October 2026 (150x38,
    6 chars from 1-9 and A-Z without O). Returns (text, confidence).

    confidence is the CTC beam's geometric-mean per-character probability --
    not accuracy, so pair it with submit feedback (see README)."""
    pil = _to_pil_rgb(image)
    arr = T2.load_real(pil)[None]
    text, conf = T2.predict(_load_itat2(), arr, "cpu")
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


def solve_gujhc(image):
    """Solve a Gujarat High Court captcha (200x50, 5 lowercase-alnum chars).
    Returns (text, confidence).

    No model: SimpleCaptcha draws DejaVu Sans Mono glyphs at pen positions
    that follow from the characters themselves, so the read is a cover of the
    green channel (where all three ink colours are pure transmittance) with
    the noise line masked. Confidence is the fraction of visible ink the five
    glyphs explain, capped by the margin to the next-best word."""
    return GH.solve_image(image)


def solve_madhc(image):
    """Solve a Madras High Court captcha (120x60, 6 digits).
    Returns (text, confidence).

    No model: securimage with angles, warp and noise off draws AHGBold digits
    at 80% opacity, so overlapping digits ADD in -log space and the read is an
    exact cover. Confidence is cover fidelity (>= 0.997 on every real image
    harvested), so gate it at `GATES["madhc"]`."""
    return MH.solve_image(image)


def solve_karhc(image):
    """Solve a Karnataka High Court captcha (100x20, 6 digits).
    Returns (text, confidence).

    No model and no learned templates: six cells of GD's built-in font 5
    (gdFontGiant) at a fixed origin, matched against that font's bitmaps from
    libgd's source. JPEG blur caps confidence near 0.9 even on a clean read,
    so gate at `GATES["karhc"]`."""
    return KH.solve_image(image)


def solve_calhc(image):
    """Solve a Calcutta High Court captcha (120x36, 6 chars from
    `2346789abcdefghjmnpqrtuxyz`). Returns (text, confidence).

    The portal lowercases both the drawn text and the submission (mews/captcha
    with `sensitive` off), so the read is lowercase and case never matters.
    confidence is the CTC beam's geometric-mean per-character probability."""
    arr = CH.load_real(_to_pil_rgb(image))[None]
    text, conf = CH.predict(_load_calhc(), arr, "cpu")
    return text[0], float(conf[0])


def solve_bytes(image, kind=None):
    """Auto-route and solve. Returns (text, confidence, kind).

    kind is inferred from the image's native size; pass kind explicitly to force
    it. Where two kinds share a size (120x40 gstat/NGT, 200x50 Gujarat HC/K2,
    200x45 CERSAI/Goa RERA, 200x80 old/new MCA) a content test in _SIZES
    decides."""
    original = _to_pil_any(image)
    pil = original.convert("RGB")
    if kind is None:
        kind = _SIZES.get((pil.width, pil.height))
        if callable(kind):                    # shared size: decide by content
            kind = kind(original)
        if kind is None:                      # unknown size: fall back to width
            kind = "securimage" if pil.width > 180 else "gstat"
    if kind == "k2":
        text, conf = K2.solve_image(original)     # its ink is the alpha channel
    elif kind == "cersai":
        text, conf = CE.solve_image(pil)
    elif kind == "aphc":
        text, conf = AP.solve_image(pil)
    elif kind == "mcacdm":
        text, conf = CDM.solve_image(pil)
    elif kind == "cestat":
        text, conf = CT.solve_image(pil)
    elif kind == "dcourts":
        text, conf = DC.solve_image(pil)
    elif kind == "tgrera":
        text, conf = TGR.solve_image(original)    # RGBA PNG
    elif kind == "garera":
        text, conf = GAR.solve_image(pil)
    elif kind == "uprera":
        text, conf = UPR.solve_image(pil)
    elif kind == "sci":
        text, conf = SCI.solve_image(pil)
    elif kind in ("hrrera", "pbrera", "hprera"):
        text, conf = _solve_rera(kind, pil)
    elif kind == "gujhc":
        text, conf = solve_gujhc(pil)
    elif kind == "madhc":
        text, conf = solve_madhc(pil)
    elif kind == "calhc":
        text, conf = solve_calhc(pil)
    elif kind == "karhc":
        text, conf = solve_karhc(pil)
    elif kind == "hrmut":
        text, conf = H.solve_image(pil)
    elif kind == "mhigr":
        text, conf = I.solve_image(pil)
    elif kind == "securimage":
        text, conf = solve_securimage(pil)
    elif kind == "gst":
        text, conf = solve_gst(pil)
    elif kind == "mca":
        text, conf = solve_mca(pil)
    elif kind == "mca2":
        text, conf = solve_mca2(pil)
    elif kind == "epfo":
        text, conf = solve_epfo(pil)
    elif kind == "itat":
        text, conf = solve_itat(pil)
    elif kind == "itat2":
        text, conf = solve_itat2(pil)
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
    Returns as soon as a read is confident (that read == current session code).
    HR mutation uses a validation-selected confidence floor of 0.97, even when
    min_conf is lower; after exhaustion the returned confidence still needs
    checking by the caller."""
    text, conf, k = "", 0.0, kind
    for i in range(1, max_tries + 1):
        text, conf, k = solve_bytes(fetch(), kind=kind)   # last read == session code
        threshold = max(min_conf, 0.97) if k == "hrmut" else min_conf
        if conf < threshold:
            continue
        if avoid_ambiguous and set(text) & set(AMBIGUOUS.get(k, "")):
            continue          # unresolvable glyph in the read -> take a fresh captcha
        return text, conf, k, i
    return text, conf, k, max_tries


# looked up at call time, so the function stays patchable
GROUPS["bharatkosh"] = lambda images: solve_bharatkosh_group(images)
