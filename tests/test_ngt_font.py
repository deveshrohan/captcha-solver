#!/usr/bin/env python3
"""Corroborate the NGT glyph labels against libgd's canonical font.

    python3 -m unittest discover -s tests -v

This file exists to close one specific hole, and it is worth being explicit
about which.

The shipped glyph library (`solver/ngt_glyphs.json`) is built by clustering real
captcha cells and attaching a character to each cluster — 36 hand decisions,
read once off a montage (see mkglyphs_ngt.py). Clustering is objective; the
*labels* are not. And the three glyphs most likely to be misread are `1`, `l`
and `i`, which are mutually 6 pixels apart in this font.

A hand-labelled evaluation cannot catch a mistake there. Whoever transposed `1`
and `l` while labelling the clusters would transpose them again while labelling
the eval set, and the held-out score would come back a confident 100% for a
reader that swaps them on every read forever.

So the labels are checked against external ground truth instead. PHP's
`imagestring($im, 5, ...)` draws with GD's built-in font 5, whose bitmaps are
public in libgd's `src/gdfontg.c`, indexed **by ASCII code**. That indexing is
the point: the character identity on the reference side comes from the C array
position, not from anyone looking at a picture.

(Font 5 is `gdFontGiant`, 9x15 — *not* `gdFontLarge`, which is font 4 at 8x16.
The measured 9px pitch is what distinguishes them, and an earlier draft of the
design named the wrong one.)
"""
import json
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solver import ngt as N

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, "gdfontgiant_alnum.json")

# The font cell is 15 rows; the reader's cell is 12. The alphanumeric glyphs
# occupy exactly rows 3..14, which is what makes the 12-row cell lossless.
ROW_OFFSET = 3


def _load_reference():
    with open(REF) as f:
        raw = json.load(f)
    cell = tuple(raw["cell"])
    glyphs = {c: np.array([[ch == "1" for ch in row] for row in rows], bool)
              for c, rows in raw["glyphs"].items()}
    return cell, glyphs


class TestReferenceFont(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cell, cls.ref = _load_reference()

    def test_reference_is_the_giant_font_cell(self):
        self.assertEqual(self.cell, (15, 9))
        self.assertEqual(sorted(self.ref), sorted(N.CHARSET))
        for c, b in self.ref.items():
            self.assertEqual(b.shape, (15, 9), c)

    def test_alnum_glyphs_occupy_exactly_rows_3_to_14(self):
        """The reader crops the 15-row font cell to 12 rows. That is only safe
        if no alphanumeric glyph puts ink in the first three rows — otherwise
        the reader would be silently clipping the tops off its own templates."""
        for c, b in self.ref.items():
            self.assertFalse(b[:ROW_OFFSET].any(),
                             "%r has ink above the reader's cell" % c)
        used = [r for r in range(15)
                if any(b[r].any() for b in self.ref.values())]
        self.assertEqual((min(used), max(used)), (ROW_OFFSET, 14))

    def test_descenders_reach_the_last_row(self):
        """g j p q y are why the cell runs to image row 24 rather than 22."""
        deep = "".join(sorted(c for c, b in self.ref.items() if b[14].any()))
        self.assertEqual(deep, "gjpqy")


class TestLabelsMatchTheFont(unittest.TestCase):
    """The check this file exists for."""

    @classmethod
    def setUpClass(cls):
        cls.cell, cls.ref = _load_reference()
        cls.g = N.glyphs()

    def test_every_shipped_glyph_matches_the_canonical_font(self):
        wrong = []
        for c in sorted(N.CHARSET):
            want = self.ref[c][ROW_OFFSET:ROW_OFFSET + N.CELL_H]
            if not np.array_equal(self.g[c], want):
                wrong.append((c, int((self.g[c] ^ want).sum())))
        self.assertEqual(wrong, [],
                         "labels disagree with gdFontGiant: %r" % (wrong,))

    def test_the_bar_glyphs_specifically(self):
        """Named on their own because they are the ones a human gets wrong, and
        the ones a hand-labelled eval could never catch."""
        for c in ("1", "l", "i"):
            want = self.ref[c][ROW_OFFSET:ROW_OFFSET + N.CELL_H]
            self.assertTrue(np.array_equal(self.g[c], want),
                            "%r does not match the canonical font" % c)

    def test_no_label_is_merely_the_nearest_match(self):
        """Each shipped bitmap must match its OWN character exactly, and no
        other character's bitmap. A transposition would still satisfy a
        'matches some glyph' check, so assert identity, not membership."""
        for c in sorted(N.CHARSET):
            want = self.ref[c][ROW_OFFSET:ROW_OFFSET + N.CELL_H]
            others = [d for d in N.CHARSET
                      if d != c and np.array_equal(self.g[c],
                                                   self.ref[d][ROW_OFFSET:ROW_OFFSET + N.CELL_H])]
            self.assertEqual(others, [], "%r also matches %r" % (c, others))
            self.assertTrue(np.array_equal(self.g[c], want), c)


if __name__ == "__main__":
    unittest.main()
