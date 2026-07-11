"""Segment a cleaned (ink=1) captcha into exactly N digit cells.

Strategy: vertical projection (ink pixels per column) gives runs of ink
separated by whitespace gaps. Because the captcha always has a fixed digit
count, we reconcile the run list to exactly N by:
  - dropping 1px specks,
  - merging the closest-together runs when we have too many,
  - splitting the widest run at its lowest-ink interior column when too few.
Each cell is then cropped to its ink bounding box, padded square, and resized.
"""
import numpy as np
from PIL import Image


def _runs(col):
    runs, x, W = [], 0, len(col)
    while x < W:
        if col[x] > 0:
            x0 = x
            while x < W and col[x] > 0:
                x += 1
            runs.append([x0, x])
        else:
            x += 1
    return runs


def _split_widest(runs, col):
    # widest run -> cut at its interior column of minimum ink
    i = max(range(len(runs)), key=lambda k: runs[k][1] - runs[k][0])
    x0, x1 = runs[i]
    if x1 - x0 < 4:
        return False
    m = 3  # keep away from the edges of the run
    seg = col[x0 + m:x1 - m]
    if len(seg) == 0:
        return False
    cut = x0 + m + int(np.argmin(seg))
    runs[i:i + 1] = [[x0, cut], [cut, x1]]
    return True


def _merge_closest(runs):
    # merge the adjacent pair with the smallest inter-run gap
    gaps = [(runs[k + 1][0] - runs[k][1], k) for k in range(len(runs) - 1)]
    _, k = min(gaps)
    runs[k:k + 2] = [[runs[k][0], runs[k + 1][1]]]


def find_cells(ink, n=6):
    """Return N [x0, x1) column spans covering the digits, left to right."""
    col = ink.sum(axis=0)
    runs = _runs(col)
    # drop 1px specks first if that still leaves us enough runs
    specks = [r for r in runs if (r[1] - r[0]) <= 1]
    if specks and len(runs) - len(specks) >= n:
        runs = [r for r in runs if (r[1] - r[0]) > 1]
    while len(runs) > n:
        _merge_closest(runs)
    while len(runs) < n:
        if not _split_widest(runs, col):
            break
    return runs


def crop_digit(ink, x0, x1, out=28, pad=4):
    """Crop [x0,x1) columns to the ink bbox, pad square, resize to out x out."""
    cell = ink[:, x0:x1]
    rows = np.where(cell.any(axis=1))[0]
    cols = np.where(cell.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return np.zeros((out, out), np.uint8)
    cell = cell[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    h, w = cell.shape
    side = max(h, w) + 2 * pad
    canvas = np.zeros((side, side), np.uint8)
    y = (side - h) // 2
    x = (side - w) // 2
    canvas[y:y + h, x:x + w] = cell
    img = Image.fromarray(canvas * 255).resize((out, out), Image.LANCZOS)
    return (np.asarray(img) > 127).astype(np.uint8)


def segment(ink, n=6, out=28):
    """Full segmentation: cleaned ink array -> list of N (out x out) digit arrays."""
    cells = find_cells(ink, n)
    return [crop_digit(ink, x0, x1, out) for x0, x1 in cells]
