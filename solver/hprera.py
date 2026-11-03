"""HP RERA ViewOpenFile 130x35 CAPTCHA: local Arial coverage and Tesseract.

Images, answers and session tokens stay in memory. No network, training writes,
or third-party inference. The caller must measure correctness using the portal.
"""
import io
import re
import subprocess
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from scipy.ndimage import label

CHARSET = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
FONT = Path('/System/Library/Fonts/Supplemental/Arial.ttf')


@lru_cache(maxsize=1)
def templates():
    """Local Arial rasters; whole-word coverage keeps variable glyph advances."""
    font = ImageFont.truetype(str(FONT), 20)
    library = {}
    for char in CHARSET:
        advance = font.getlength(char)
        for width in {int(advance), round(advance), int(advance) + 1}:
            if width < 3:
                continue
            im = Image.new('L', (width, 35), 255)
            ImageDraw.Draw(im).text((0, 26), char, font=font, fill=0, anchor='ls')
            library.setdefault(width, []).append((char, np.asarray(im) < 128))
    return {width: (np.stack([g for _, g in variants]), [c for c, _ in variants])
            for width, variants in library.items()}


def cover(image, threshold):
    target = np.asarray(image.convert('L')) < threshold
    components, _ = label(target, structure=np.ones((3, 3)))
    sizes = np.bincount(components.ravel())
    keep = sizes >= 3
    keep[0] = False
    target = keep[components]
    transitions = {}
    for x in range(5, 125):
        options = []
        for width, (stack, chars) in templates().items():
            if x + width > 130:
                continue
            errors = np.logical_xor(stack, target[:, x:x + width]).sum((1, 2))
            best = int(errors.argmin())
            options.append((x + width, chars[best], float(errors[best])))
        transitions[x] = options
    states = {x: (float(target[:, :x].sum()), '') for x in range(6, 11)}
    for _ in range(6):
        following = {}
        for x, (score, word) in states.items():
            for end, char, error in transitions.get(x, []):
                candidate = score + error, word + char
                if end not in following or candidate < following[end]:
                    following[end] = candidate
        states = following
    if not states:
        return '', 1.0
    score, word = min((score + target[:, end:].sum(), word)
                      for end, (score, word) in states.items())
    return word, score / max(1, target.sum())


def cleaned(image, threshold=80):
    im = image if isinstance(image, Image.Image) else Image.open(
        io.BytesIO(image) if isinstance(image, (bytes, bytearray)) else image)
    rgb = np.asarray(im.convert('RGB'))
    # Characters are dark neutral pixels; colored speckles are not characters.
    dark = (rgb.max(axis=2) < threshold)
    components, count = label(dark, structure=np.ones((3, 3)))
    sizes = np.bincount(components.ravel())
    keep = sizes >= 6
    keep[0] = False
    ink = keep[components]
    return Image.fromarray(np.uint8(~ink) * 255)


def solve_image(image):
    votes = []
    original = image if isinstance(image, Image.Image) else Image.open(
        io.BytesIO(image) if isinstance(image, (bytes, bytearray)) else image)
    if original.size != (130, 35):
        raise ValueError('HP RERA captcha shape changed')
    if FONT.exists():
        candidates = [cover(original, threshold) for threshold in (120, 140, 100)]
        candidates = [(word, error) for word, error in candidates if word and error < .72]
        if candidates:
            # Threshold consensus per character tolerates a noisy or faint stem.
            word = ''.join(Counter(w[i] for w, _ in candidates).most_common(1)[0][0]
                           for i in range(6))
            return word, max(0., 1. - min(e for _, e in candidates))
    # Text starts near x=8; the right-hand background is mostly colored noise.
    # A conservative crop keeps the full six-character word even with wide caps.
    for threshold in (50, 80, 110):
        im = original.convert('L').point(lambda p: 255 if p > threshold else 0)
        im = ImageOps.expand(im.crop((3, 5, 110, 33)), border=12, fill=255)
        im = im.resize((im.width * 4, im.height * 4), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format='PNG')
        for psm in (8, 7):
            result = subprocess.run(
                ['tesseract', 'stdin', 'stdout', '--psm', str(psm), '-c',
                 'tessedit_char_whitelist=' + CHARSET], input=buf.getvalue(),
                capture_output=True, timeout=15, check=True)
            word = result.stdout.decode().strip()
            if re.fullmatch('[A-Za-z0-9]{6}', word):
                votes.append(word)
    if not votes:
        return '', 0.0
    word, count = Counter(votes).most_common(1)[0]
    return word, count / max(len(votes), 1)
