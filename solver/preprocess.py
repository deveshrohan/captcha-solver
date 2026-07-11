"""Preprocessing: isolate the #333 digits, drop the #ccc noise lines.

The captcha renders digits in ~#333333 and the crossing noise lines in the
lighter ~#cccccc. A single brightness threshold therefore separates them
cleanly: anything darker than the threshold is digit ink, everything else
(lines + white background) becomes background.
"""
import numpy as np
from PIL import Image

# Digits sit around gray 51 (#333); lines around 204 (#ccc); bg 255.
# 128 sits safely between the digit ink and the line color.
DEFAULT_THRESHOLD = 128


def load_ink(path_or_img, threshold=DEFAULT_THRESHOLD):
    """Return an H x W uint8 array where 1 = digit ink, 0 = background."""
    if isinstance(path_or_img, (str, bytes)):
        im = Image.open(path_or_img)
    else:
        im = path_or_img
    a = np.asarray(im.convert("L"))
    return (a < threshold).astype(np.uint8)
