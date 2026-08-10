#!/usr/bin/env python3
"""Rebuild the Udyam class library from a corpus of real captchas.

    python3 mkglyphs_udyam.py udyam_raw            # cluster + write the sheet
    python3 mkglyphs_udyam.py udyam_raw --labels "4NUZL9..."   # write the JSON

Two-step on purpose. Clustering is automatic and produces *unlabelled* classes;
mapping each bitmap to a character is the one human step in this reader, and a
mislabelled class would silently corrupt every read containing it while leaving
every match structurally unique. So the first invocation dumps
`udyam_classes.png` for a human to read, and the second takes that reading back
as a string — one character per class, in sheet order.

The script also reports **saturation**: how the class count grows with corpus
size. The library is only complete if that curve has flattened, i.e. the last
few hundred images introduced no class that was not already known. A count that
is still climbing means rare characters are undersampled — building from too
few images is exactly how a first attempt ended up with `W` missing and `5`
split across two phase variants.
"""
import glob
import os
import sys

import numpy as np

from solver import udyam as U

SHEET = "udyam_classes.png"


def saturation(paths, step=50):
    """Class count as a function of corpus size — the completeness evidence."""
    print("saturation (corpus size -> classes with >=%d members):" % 3)
    prev = None
    for n in range(step, len(paths) + step, step):
        n = min(n, len(paths))
        k = len(U.extract_classes(paths[:n]))
        flag = "" if prev is None else ("  +%d" % (k - prev))
        print("  %4d images -> %2d classes%s" % (n, k, flag))
        prev = k
        if n == len(paths):
            break
    return prev


def dump_sheet(clusters, path):
    from PIL import Image, ImageDraw
    cols = 9
    cw, ch = 40, 52
    rows = (len(clusters) + cols - 1) // cols
    sheet = Image.new("L", (cw * cols, ch * rows), 255)
    d = ImageDraw.Draw(sheet)
    for i, (n, tpl) in enumerate(clusters):
        cx, cy = (i % cols) * cw, (i // cols) * ch
        sub = Image.fromarray(np.where(tpl, 0, 255).astype("uint8"))
        sheet.paste(sub, (cx + 4, cy + 2))
        d.text((cx + 4, cy + 38), "#%d n=%d" % (i, n), fill=0)
    sheet = sheet.resize((sheet.width * 3, sheet.height * 3), Image.NEAREST)
    sheet.save(path)
    print("wrote %s — %d classes, in index order" % (path, len(clusters)))


def main():
    corpus = sys.argv[1] if len(sys.argv) > 1 else "udyam_raw"
    paths = sorted(glob.glob(os.path.join(corpus, "*.png")))
    if not paths:
        print("no images in %s — run download_udyam.py first" % corpus)
        return 1
    print("corpus: %d images" % len(paths))

    labels = None
    if "--labels" in sys.argv:
        labels = sys.argv[sys.argv.index("--labels") + 1].strip().upper()

    if "--saturation" in sys.argv:
        saturation(paths)

    clusters = U.extract_classes(paths)
    print("clustered -> %d classes (expected %d)" % (len(clusters), len(U.CHARSET)))
    counts = [n for n, _ in clusters]
    print("members per class: min %d  median %d  max %d"
          % (min(counts), int(np.median(counts)), max(counts)))

    if labels is None:
        dump_sheet(clusters, SHEET)
        print("\nnow read %s and re-run with:" % SHEET)
        print('  python3 mkglyphs_udyam.py %s --labels "<one char per class, in order>"'
              % corpus)
        return 0

    if len(labels) != len(clusters):
        print("!! %d labels for %d classes — refusing to write a misaligned library"
              % (len(labels), len(clusters)))
        return 1
    unknown = sorted(set(labels) - set(U.CHARSET))
    if unknown:
        print("!! labels contain characters outside CHARSET: %s" % unknown)
        return 1
    # Duplicate labels are expected, not an error: a character that renders in
    # two subpixel phases yields two clusters. They are kept as separate
    # variants rather than merged, because the clustering threshold that would
    # merge them (IoU 0.828 for the two `5`s) sits below the closest genuinely
    # distinct pair (`E` vs `F` at 0.794).
    data = {}
    for ch, (_, tpl) in zip(labels, clusters):
        data.setdefault(ch, []).append(tpl)
    multi = {c: len(v) for c, v in data.items() if len(v) > 1}
    if multi:
        print("characters with multiple phase variants: %s" % multi)
    missing = sorted(set(U.CHARSET) - set(data))
    if missing:
        print("!! no class for %s — corpus too small for those characters" % missing)
        return 1

    U.save_classes(os.path.join("solver", "udyam_glyphs.json"), data)
    size = os.path.getsize(os.path.join("solver", "udyam_glyphs.json"))
    print("wrote solver/udyam_glyphs.json — %d classes, %.1f KB"
          % (len(data), size / 1024.0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
