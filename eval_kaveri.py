#!/usr/bin/env python3
"""Score the Kaveri reader.

    python3 eval_kaveri.py                 # labels + structural check over the corpus
    python3 eval_kaveri.py --sprites out.png   # dump the labelled sprite sheet

Unlike the other readers there is nothing to hold out: no model is trained, so
every labelled image is held-out by construction and `--all` would be
meaningless. That removes the failure mode `eval_mca.py` has to guard against,
where scoring memorised training images reads 96.5% against a true 75%.

The headline number here is not the labelled accuracy but the **uniqueness
rate**: for each image, enumerate every set of six sprites whose union is
exactly the ink. Where exactly one such cover exists, no other string could have
produced that image, so the read is forced rather than merely likely. That is a
structural guarantee, and it does not get better or worse with sample size.

Uniqueness cannot catch one thing, and it is worth being explicit about it: the
mapping from each of the 36 bitmaps to a character was made by a human reading a
contact sheet. A mislabelled sprite would corrupt every read containing it while
leaving every cover unique. That is what the labelled set and `--sprites` are
for — the labels below were read by eye from zoomed images *before* the matcher
was written, so agreement between them is an end-to-end check of the mapping and
not a restatement of it.
"""
import glob
import json
import os
import sys
import time
from collections import Counter

import numpy as np

from solver import kaveri as K

LABELS = "kaveri_labels.json"
CORPUS = "kaveri_raw"


def dump_sprites(path):
    """Render the labelled sprite library so the char mapping can be eyeballed."""
    from PIL import Image, ImageDraw
    G = K.glyphs()
    chars = sorted(G)
    cols, cw, ch = 12, 34, 46
    rows = (len(chars) + cols - 1) // cols
    sheet = Image.new("L", (cw * cols, ch * rows), 255)
    d = ImageDraw.Draw(sheet)
    for i, c in enumerate(chars):
        g, top = G[c]
        cx, cy = (i % cols) * cw, (i // cols) * ch
        sub = Image.fromarray(np.where(g, 0, 255).astype("uint8"))
        sheet.paste(sub, (cx + 4, cy + top - 12))
        d.text((cx + 4, cy + 34), c, fill=0)
    sheet = sheet.resize((sheet.width * 3, sheet.height * 3), Image.NEAREST)
    sheet.save(path)
    print("wrote %s — each bitmap is printed above the character it is mapped to"
          % path)


def main():
    if "--sprites" in sys.argv:
        i = sys.argv.index("--sprites")
        out = sys.argv[i + 1] if len(sys.argv) > i + 1 else "kaveri_sprites.png"
        return dump_sprites(out)

    paths = sorted(glob.glob(os.path.join(CORPUS, "*.png")))
    if not paths:
        print("no corpus at %s/ — run `python3 download_kaveri.py 200 %s`"
              % (CORPUS, CORPUS))
        return 1

    # ---- structural: does the sprite library explain the image exactly, and
    # ---- is the explanation unique?
    t0 = time.time()
    exact = unique = 0
    multi, none = [], []
    confs = []
    for p in paths:
        m = K.ink_mask(p)
        sols = K.solutions(m, limit=2)
        if len(sols) >= 1:
            exact += 1
            confs.append(1.0)
        else:
            none.append(p)
            confs.append(K.solve(m)[1])
        if len(sols) == 1:
            unique += 1
        elif len(sols) > 1:
            multi.append((p, sols))
    dt = time.time() - t0

    n = len(paths)
    print("corpus : %d real captchas (%s/)" % (n, CORPUS))
    print("exact cover exists     : %d/%d (%.1f%%)" % (exact, n, exact / n * 100))
    print("cover is UNIQUE        : %d/%d (%.1f%%)" % (unique, n, unique / n * 100))
    print("mean confidence        : %.4f" % float(np.mean(confs)))
    print("time                   : %.1f ms/image" % (dt / n * 1000))
    if none:
        print("\nno exact cover (would fall back to a scored read):")
        for p in none[:10]:
            t, c = K.solve_image(p)
            print("   %s -> %r conf=%.3f" % (p, t, c))
    if multi:
        print("\nAMBIGUOUS — more than one exact cover, so the read is a guess:")
        for p, s in multi[:10]:
            print("   %s -> %s" % (p, s))

    # ---- labelled: end-to-end check of the bitmap -> character mapping
    if not os.path.exists(LABELS):
        print("\nno %s — skipping the labelled check" % LABELS)
        return 0
    labels = json.load(open(LABELS))
    items = [(p, l) for p, l in sorted(labels.items()) if os.path.exists(p)]
    if not items:
        print("\n%s references no existing images — skipping" % LABELS)
        return 0

    preds = [K.solve_image(p)[0] for p, _ in items]
    ex = sum(p == l for p, (_, l) in zip(preds, items))
    chars = sum(sum(a == b for a, b in zip(p, l)) for p, (_, l) in zip(preds, items))
    total = sum(len(l) for _, l in items)
    print("\nlabelled images        : %d (read by eye before the matcher existed)" % len(items))
    print("exact                  : %d/%d (%.1f%%)" % (ex, len(items), ex / len(items) * 100))
    print("chars                  : %d/%d (%.2f%%)" % (chars, total, chars / total * 100))

    conf = Counter()
    for p, (_, l) in zip(preds, items):
        if len(p) != len(l):
            print("   LENGTH MISMATCH %s: %r vs %r" % (_, p, l))
            continue
        for a, b in zip(l, p):
            if a != b:
                conf[(a, b)] += 1
    if conf:
        print("\ncharacter confusions (true -> pred):")
        for (a, b), k in conf.most_common(15):
            print("   %s -> %s   x%d" % (a, b, k))
    return 0


if __name__ == "__main__":
    sys.exit(main())
