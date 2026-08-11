#!/usr/bin/env python3
"""Score the NGT reader.

    python3 eval_ngt.py                # held-out split + the label-free check
    python3 eval_ngt.py --all          # score every label (inflated; says so)
    python3 eval_ngt.py --coverage     # label-free check only, no labels needed
    python3 eval_ngt.py --sheet g.png  # dump the labelled glyph sheet

Two numbers, answering different questions. Reporting only one of them would be
misleading in a way that is easy to miss.

**Exact-cover rate** is the structural one and needs *no labels at all*. Every
cell of every image either matches a template byte for byte or it does not; the
reader claims all six always do. That is a falsifiable claim about the generator
which no hand-reading error can flatter, and it is the number the README leads
on.

**Labelled accuracy** checks the one thing the cover rate cannot. The 36 class
bitmaps were mapped to characters by a human reading a montage, and a
mislabelled class corrupts every read containing it while leaving the cover
perfectly exact. Exactness proves the *shape* was found; only labels prove the
*character* is right.

Note that `tests/test_ngt_font.py` already checks those 36 labels against
libgd's canonical `gdFontGiant`, indexed by ASCII code — which is stronger
evidence than this eval, because it owes nothing to anyone's eye. The labelled
score here is the end-to-end confirmation, not the primary guard.

### Honest by default

The glyph library must be built from the TRAIN split only. Templates derived
from the very images being scored would inflate the number — the trap the README
documents for MCA, which reads 96.5% over all its labels against 75.0% held out.

That is not left to memory: `mkglyphs_ngt.py` records `built_from` inside the
library, and this script refuses to report a held-out accuracy unless it says
`train-split`.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

from solver import ngt as N

LABELS = "ngt_labels.json"
SPLIT = "ngt_split.json"
CORPUS = "ngt_raw"


def dump_sheet(path):
    """Render the labelled glyph library so the char mapping can be eyeballed.

    This is the artifact the 36 hand decisions were made from; dumping it lets
    them be re-checked directly rather than taken on trust."""
    from PIL import Image, ImageDraw
    g = N.glyphs()
    items = sorted(g)
    cols, cw, chh = 6, 20, 30
    rows = (len(items) + cols - 1) // cols
    sheet = Image.new("L", (cw * cols, chh * rows), 255)
    d = ImageDraw.Draw(sheet)
    for i, c in enumerate(items):
        cx, cy = (i % cols) * cw, (i // cols) * chh
        bmp = np.where(g[c], 0, 255).astype("uint8")
        sheet.paste(Image.fromarray(bmp), (cx + 5, cy + 2))
        d.text((cx + 5, cy + 16), c, fill=0)
    sheet = sheet.resize((sheet.width * 4, sheet.height * 4), Image.NEAREST)
    sheet.save(path)
    print("wrote %s" % path)


def coverage(paths):
    """The label-free claim: every cell of every image is an exact template
    match, so confidence is exactly 1.0."""
    bad, confs, texts = [], [], []
    for p in paths:
        text, conf = N.solve_image(p)
        confs.append(conf)
        texts.append(text)
        if conf != 1.0 or len(text) != N.LENGTH:
            bad.append((os.path.basename(p), text, conf))
    confs = np.array(confs)
    print("\nexact-cover check (no labels involved)")
    print("  images                    : %d" % len(paths))
    print("  six-cell exact cover      : %d/%d" % (len(paths) - len(bad), len(paths)))
    print("  confidence mean / min     : %.4f / %.4f"
          % (confs.mean(), confs.min()) if len(confs) else "  (none)")
    print("  distinct reads            : %d/%d" % (len(set(texts)), len(texts)))
    seen = {c for t in texts for c in t}
    print("  charset classes exercised : %d/36" % len(seen))
    for b in bad[:10]:
        print("    NOT EXACT: %s -> %r (conf %.3f)" % b)
    return bad


def score(pairs, title):
    exact = sum(1 for _, want, got in pairs if want == got)
    chars = sum(sum(a == b for a, b in zip(want, got.ljust(len(want))))
                for _, want, got in pairs)
    total = sum(len(want) for _, want, _ in pairs)
    print("\n%s" % title)
    print("  exact : %d/%d (%.1f%%)"
          % (exact, len(pairs), 100.0 * exact / max(1, len(pairs))))
    print("  chars : %d/%d (%.2f%%)"
          % (chars, total, 100.0 * chars / max(1, total)))
    for name, want, got in pairs:
        if want != got:
            print("    %s  want %r  got %r" % (name, want, got))
    return exact, len(pairs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true",
                    help="score every label, not just the held-out split")
    ap.add_argument("--coverage", action="store_true",
                    help="label-free check only")
    ap.add_argument("--sheet", help="dump the labelled glyph sheet to PNG")
    args = ap.parse_args()

    if args.sheet:
        dump_sheet(args.sheet)
        return 0

    paths = sorted(glob.glob(os.path.join(CORPUS, "*.png")))
    if not paths:
        print("no corpus at %s/ — run download_ngt.py first" % CORPUS)
        return 1

    bad = coverage(paths)

    if args.coverage:
        return 1 if bad else 0

    if not os.path.exists(LABELS):
        print("\nno %s — skipping the labelled score. The cover check above "
              "still holds, but it cannot catch a mislabelled class." % LABELS)
        return 1 if bad else 0

    labels = json.load(open(LABELS))
    meta = N.glyph_meta()
    built = meta.get("built_from")

    if args.all:
        keys = sorted(labels)
        title = ("labelled accuracy, ALL %d labels — INFLATED if the library "
                 "saw them (built_from=%s)" % (len(keys), built))
    else:
        if not os.path.exists(SPLIT):
            print("\nno %s — refusing to report an accuracy without a split, "
                  "since the library may have been built from these images."
                  % SPLIT)
            return 1
        test = json.load(open(SPLIT))["test"]
        keep = set(test) | {os.path.basename(t) for t in test}
        keys = sorted(k for k in labels
                      if k in keep or os.path.basename(k) in keep)
        if built != "train-split":
            print("\nREFUSING to report a held-out accuracy: the glyph library "
                  "records built_from=%r, so its templates may derive from the "
                  "images being scored. Rebuild with:\n"
                  "    python3 mkglyphs_ngt.py %s --split %s"
                  % (built, CORPUS, SPLIT))
            return 1
        title = ("labelled accuracy, HELD-OUT split (%d images; library built "
                 "from %d train images)" % (len(keys), meta.get("n_images", 0)))

    pairs = []
    for k in keys:
        p = k if os.path.exists(k) else os.path.join(CORPUS, os.path.basename(k))
        if not os.path.exists(p):
            print("  missing image for label %s" % k)
            continue
        pairs.append((os.path.basename(p), labels[k], N.solve_image(p)[0]))

    if not pairs:
        print("\nno labelled images found on disk")
        return 1

    exact, n = score(pairs, title)
    return 0 if (exact == n and not bad) else 1


if __name__ == "__main__":
    sys.exit(main())
