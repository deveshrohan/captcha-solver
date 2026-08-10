#!/usr/bin/env python3
"""Score the Udyam reader.

    python3 eval_udyam.py                      # labels + structural check
    python3 eval_udyam.py --sprites out.png    # dump the labelled class sheet

As with Kaveri there is nothing to hold out: no model is trained, so every
labelled image is held-out by construction. That removes the failure mode
`eval_mca.py` has to guard against, where scoring memorised training images
reads 96.5% against a true 75%.

Two numbers, and they answer different questions.

**Substitution margin** is the structural one, and it does not depend on the
labels at all. For each glyph it asks how much worse the cover gets if that
character alone is swapped for the best alternative. A positive margin
everywhere means every character explains its pixels strictly better than any
rival, so the reading is forced by the image rather than merely likely. This is
the graded counterpart of Kaveri's exact-cover uniqueness — antialiasing rules
out an exact cover here, so "no other cover exists" becomes "every rival
explains the pixels measurably worse".

**Labelled accuracy** checks the one thing uniqueness cannot: the mapping from
each of the 34 class bitmaps to a character was made by a human reading a
contact sheet, and a mislabelled class corrupts every read containing it while
leaving every margin healthy. The labels in `udyam_labels.json` were read by eye
from zoomed images before being compared with the reader, and they are weighted
towards the hard cases — 26 of the 36 are images whose glyphs touch, which is
where a cut-based reader fails. `--sprites` dumps the labelled sheet so the
mapping can be re-checked directly.
"""
import glob
import json
import os
import sys
import time

import numpy as np

from solver import udyam as U

LABELS = "udyam_labels.json"
CORPUS = "udyam_raw"


def dump_sprites(path):
    """Render the labelled class library so the char mapping can be eyeballed."""
    from PIL import Image, ImageDraw
    C = U.classes()
    items = [(ch, i, v) for ch in sorted(C) for i, v in enumerate(C[ch])]
    cols, cw, ch_ = 9, 42, 56
    rows = (len(items) + cols - 1) // cols
    sheet = Image.new("L", (cw * cols, ch_ * rows), 255)
    d = ImageDraw.Draw(sheet)
    for i, (c, vi, g) in enumerate(items):
        cx, cy = (i % cols) * cw, (i // cols) * ch_
        sheet.paste(Image.fromarray(np.where(g, 0, 255).astype("uint8")), (cx + 4, cy + 2))
        tag = c if len(C[c]) == 1 else "%s (phase %d)" % (c, vi + 1)
        d.text((cx + 4, cy + 42), tag, fill=0)
    sheet = sheet.resize((sheet.width * 3, sheet.height * 3), Image.NEAREST)
    sheet.save(path)
    print("wrote %s — each bitmap is printed above the character it is mapped to"
          % path)


def main():
    if "--sprites" in sys.argv:
        i = sys.argv.index("--sprites")
        dump_sprites(sys.argv[i + 1] if len(sys.argv) > i + 1 else "udyam_sprites.png")
        return 0

    paths = sorted(glob.glob(os.path.join(CORPUS, "*.png")))
    if not paths:
        print("no images in %s — run download_udyam.py first" % CORPUS)
        return 1

    print("class library: %d characters, %d bitmaps (%s)"
          % (len(U.classes()), sum(len(v) for v in U.classes().values()),
             ", ".join("%s x%d" % (c, len(v))
                       for c, v in sorted(U.classes().items()) if len(v) > 1) or "no phase variants"))

    # ---- structural: substitution margins over the whole corpus -------------
    t0 = time.time()
    margins, confs, nbad = [], [], 0
    for p in paths:
        m = U.ink_mask(p)
        text, conf = U.solve(m)
        confs.append(conf)
        if len(text) != U.LENGTH:
            nbad += 1
        _, mar = U.substitution_margin(m)
        margins += [x[2] for x in mar]
    dt = time.time() - t0
    margins = np.array(margins)
    confs = np.array(confs)
    print("\ncorpus: %d images, %.0f ms/image" % (len(paths), 1000 * dt / len(paths)))
    print("  read %d characters: %d images not of length %d" % (len(margins), nbad, U.LENGTH))
    print("  substitution margin  min %.2f  p1 %.2f  median %.2f   (<=0: %d)"
          % (margins.min(), np.percentile(margins, 1), np.median(margins),
             int((margins <= 0).sum())))
    print("  confidence           min %.3f  p1 %.3f  median %.3f  (<0.90: %d)"
          % (confs.min(), np.percentile(confs, 1), np.median(confs),
             int((confs < 0.90).sum())))

    # ---- labelled accuracy ---------------------------------------------------
    if not os.path.exists(LABELS):
        print("\nno %s — skipping the labelled check" % LABELS)
        return 0
    with open(LABELS) as f:
        labels = json.load(f)
    labels = {k: v for k, v in labels.items() if os.path.exists(k)}
    if not labels:
        print("\nlabelled images are not present locally — skipping")
        return 0

    ok = 0
    hard = hard_ok = 0
    for p, want in sorted(labels.items()):
        m = U.ink_mask(p)
        got, conf = U.solve(m)
        touching = len(U.ink_runs(m)) != U.LENGTH
        hard += touching
        if got == want:
            ok += 1
            hard_ok += touching
        else:
            print("  MISS %s want %s got %s (conf %.3f%s)"
                  % (p, want, got, conf, ", touching glyphs" if touching else ""))
    print("\nlabelled: %d/%d exact (%.1f%%)" % (ok, len(labels), 100.0 * ok / len(labels)))
    print("  of which glyphs touch: %d/%d exact" % (hard_ok, hard))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
