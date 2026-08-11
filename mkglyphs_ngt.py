#!/usr/bin/env python3
"""Rebuild the NGT glyph library from a corpus of real captchas.

    python3 mkglyphs_ngt.py ngt_raw --split ngt_split.json
    python3 mkglyphs_ngt.py ngt_raw            # every image; NOT eval-safe

Writes `solver/ngt_glyphs.json`.

The NGT font is a fixed bitmap face blitted at integer positions on a fixed
grid, so every instance of a character is byte-identical and "extracting the
library" is just clustering the cells. That makes the build almost trivially
strict, and this script leans on that: it asserts the structural invariants and
refuses to write a library if any of them breaks.

    1. every cell in the corpus falls into a cluster with a duplicate
       (zero singletons)
    2. exactly 36 clusters
    3. every cluster's bitmap is one this script has a *label* for
    4. min pairwise Hamming across the 36 is >= 6

Invariant 1 is the sharp one: a single singleton means some cell was not
byte-identical to any other, which would mean the font, the grid or the
noise-under-text layer order had changed — and the exact-cover reader would no
longer be valid. Better to fail here than to ship a library that quietly stopped
describing the generator.

### Why the labels live here as hashes

The cluster -> character map is 36 hand decisions, read once off a montage of
the clusters. Keying them by bitmap hash rather than by position means a rebuild
on any corpus produces the same labels, and a bitmap that ever changes fails to
match instead of silently taking its neighbour's label.

Those 36 hand decisions are the whole correctness surface of the reader, and the
riskiest of them are `1`, `l` and `i` — mutually 6 px apart. A human labelling
the eval set would likely repeat any transposition made here, so the held-out
score could not catch it. `tests/test_ngt_font.py` therefore checks these labels
against libgd's canonical `gdFontGiant` (PHP's font 5), which is external
ground truth indexed by ASCII code rather than by eye.

### The split

The library is built from the TRAIN split only. Deriving templates from every
labelled image and then scoring those same images is the inflated-number trap
the README documents for MCA (96.5% over all labels vs 75.0% held out). Passing
no --split builds from everything, which is fine for a coverage check but must
not be used to produce a reported accuracy.
"""
import argparse
import collections
import glob
import hashlib
import itertools
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from solver import ngt as N

# bitmap sha1[:12] -> character. See the module docstring.
LABELS = {
    "4bdff18fb75c": "q", "d2267b3ef569": "o", "cda6610bb569": "e",
    "fa9fa14bfb74": "a", "4e9702659830": "c", "83034532ecf9": "g",
    "ed2004433fa0": "z", "be43a3e07611": "s", "404573464beb": "m",
    "b2d503ef20de": "x", "41dfe7f232c2": "v", "28f3ac2f0543": "u",
    "bc6d2530c664": "y", "86c3d3140436": "w", "fec643505d04": "n",
    "a9e33e8501ee": "p", "f5470c3840fb": "r", "c10b65ae1036": "t",
    "c26398931eb4": "d", "3e907e431171": "j", "b4255147cbf2": "4",
    "2026a01cc4df": "i", "427eab5049a1": "1", "0e4d35ee6ce7": "0",
    "e970418cffdf": "f", "ecf64a28f64f": "l", "5e984b8badde": "6",
    "03d2211f1627": "2", "d83ddd2a7de4": "8", "a821f7d95dfc": "9",
    "2e96c1739089": "k", "8aa62eef5532": "3", "4569535b8950": "h",
    "b1d93022f2d1": "b", "51a6815b6a6e": "5", "a86a07ebe726": "7",
}

MIN_HAMMING = 6


def digest(bits):
    return hashlib.sha1(bits).hexdigest()[:12]


def collect(paths):
    """-> Counter of packed cell bitmaps over every image."""
    seen = collections.Counter()
    for p in paths:
        mask = N.ink_mask(p)
        for cell in N.cells(mask):
            seen[cell.tobytes()] += 1
    return seen


def build(paths):
    seen = collect(paths)

    singles = [k for k, v in seen.items() if v == 1]
    if singles:
        raise SystemExit(
            "%d singleton cluster(s): some cell is not byte-identical to any "
            "other, so the font, the grid or the layer order has changed and "
            "the exact-cover reader is no longer valid." % len(singles))

    if len(seen) != 36:
        raise SystemExit("expected 36 clusters, got %d" % len(seen))

    unknown = [k for k in seen if digest(k) not in LABELS]
    if unknown:
        raise SystemExit(
            "%d cluster(s) with no label — the font changed, or this corpus "
            "contains a glyph the library was never built for." % len(unknown))

    out = {}
    for k in seen:
        arr = np.frombuffer(k, bool).reshape(N.CELL_H, N.CELL_W)
        out[LABELS[digest(k)]] = arr

    if sorted(out) != sorted(N.CHARSET):
        missing = sorted(set(N.CHARSET) - set(out))
        raise SystemExit("charset incomplete, missing: %s" % "".join(missing))

    worst = min((int((out[a] ^ out[b]).sum()), a, b)
                for a, b in itertools.combinations(sorted(out), 2))
    if worst[0] < MIN_HAMMING:
        raise SystemExit("min pairwise Hamming %d (%s/%s) < %d — two labels "
                         "may have collapsed onto one shape"
                         % (worst[0], worst[1], worst[2], MIN_HAMMING))

    return out, seen, worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", nargs="?", default="ngt_raw")
    ap.add_argument("--split", help="ngt_split.json; build from its train side")
    ap.add_argument("-o", "--out",
                    default=os.path.join("solver", "ngt_glyphs.json"))
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.corpus, "*.png")))
    if not paths:
        raise SystemExit("no images in %s" % args.corpus)

    if args.split:
        with open(args.split) as f:
            train = set(json.load(f)["train"])
        # split files store repo-relative paths; accept a bare basename too
        keep = train | {os.path.basename(t) for t in train}
        paths = [p for p in paths
                 if p in keep or os.path.basename(p) in keep]
        if not paths:
            raise SystemExit("split's train side matched no image")
        built_from = "train-split"
        print("building from the TRAIN split only: %d images" % len(paths))
    else:
        built_from = "all-images"
        print("building from ALL %d images — coverage check only, the result "
              "is NOT safe to report an accuracy against" % len(paths))

    out, seen, worst = build(paths)

    # Provenance travels with the artifact so eval_ngt.py can *enforce* the
    # train-only rule rather than trust that whoever built it remembered.
    N.save_glyphs(args.out, out, meta={
        "built_from": built_from,
        "n_images": len(paths),
        "n_cells": sum(seen.values()),
        "split": os.path.basename(args.split) if args.split else None,
        "min_hamming": worst[0],
    })
    print("wrote %s: %d glyphs from %d cells" % (args.out, len(out),
                                                 sum(seen.values())))
    print("min pairwise Hamming %d (%s/%s); ink %d..%d px"
          % (worst[0], worst[1], worst[2],
             min(int(v.sum()) for v in out.values()),
             max(int(v.sum()) for v in out.values())))


if __name__ == "__main__":
    main()
