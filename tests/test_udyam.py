#!/usr/bin/env python3
"""Tests for the Udyam reader.

    python3 -m unittest discover -s tests -v

`eval_udyam.py` verifies the reader against real captchas (36/36 hand-labelled
exact, positive substitution margin on every glyph of 420 images). These cover
what real samples exercise rarely, never, or only by luck — and, in several
cases, the specific bugs that made earlier versions of this reader wrong:

* the tricolour bars and the border are dark enough to pass the ink test, so the
  band that excludes them is load-bearing rather than tidiness;
* a 2px speck of noise-line residue below the text used to stretch a glyph's
  bounding box from 25 rows to 46 and corrupted the templates built from it;
* `W` overlaps its neighbour, so no vertical cut yields both glyphs whole;
* confidence has to fall on out-of-distribution input, which by construction the
  real corpus never contains.
"""
import json
import os
import sys
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solver import udyam as U

CORPUS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "udyam_raw")
LABELS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "udyam_labels.json")


def furniture(bg=(240, 247, 255)):
    """A blank captcha with the real fixed furniture: border and both bars."""
    im = Image.new("RGB", (U.W, U.H), bg)
    a = np.asarray(im).copy()
    a[0, :] = a[U.H - 1, :] = (100, 130, 180)      # top/bottom border
    a[:, 0] = a[:, U.W - 1] = (100, 130, 180)      # side border
    a[2:4, :] = (255, 153, 51)                     # saffron bar
    a[7:12, :] = (0, 128, 0)                       # green bar
    return Image.fromarray(a)


def paste_glyphs(im, chars, x0=8, gap=2, y=1, overlap_at=None, overlap=0):
    """Blit real class bitmaps in ink colour, the way the generator does.

    `overlap_at` shifts one glyph left by `overlap` so it collides with its
    predecessor, which is how the real generator packs a wide `W` against its
    neighbour. Applying an overlap to *every* pair would not be faithful — real
    captchas only collide where a wide glyph meets one.
    """
    a = np.asarray(im).copy()
    C = U.classes()
    x = x0
    for i, ch in enumerate(chars):
        g = C[ch][0]
        h, w = g.shape
        if overlap_at is not None and i == overlap_at:
            x -= overlap
        yy = U.BAND_TOP + y
        if x < 0 or x + w > U.W or yy + h > U.H:
            raise AssertionError("test layout does not fit: %r at x=%d" % (ch, x))
        region = a[yy:yy + h, x:x + w]
        region[g] = U.INK
        a[yy:yy + h, x:x + w] = region
        x += w + gap
    return Image.fromarray(a)


class TestLibrary(unittest.TestCase):
    def test_covers_charset_exactly(self):
        C = U.classes()
        self.assertEqual(sorted(C), sorted(U.CHARSET))
        self.assertEqual(len(U.CHARSET), 33)

    def test_excludes_homoglyphs(self):
        """The portal issues no 0, I or O — the pairs that cap the MCA reader."""
        for ch in "0IO":
            self.assertNotIn(ch, U.CHARSET)
            self.assertNotIn(ch, U.classes())

    def test_templates_are_plausible_glyphs(self):
        """Guards the bug that produced 45-row `U`, `H` and `F` templates from
        crops stretched by a speck of noise-line residue."""
        for ch, variants in U.classes().items():
            for t in variants:
                h, w = t.shape
                self.assertTrue(20 <= h <= 30, "%s height %d" % (ch, h))
                self.assertTrue(10 <= w <= 40, "%s width %d" % (ch, w))
                self.assertGreater(t.sum(), 0.15 * h * w, "%s too sparse" % ch)

    def test_pack_roundtrip(self):
        a = np.array([[True, False], [False, True]])
        self.assertTrue((U._unpack(U._pack(a)) == a).all())


class TestInkMask(unittest.TestCase):
    def test_furniture_alone_yields_no_ink(self):
        """Border and green bar are both dark enough to pass a luminance cut, so
        this fails loudly if the band ever drifts."""
        m = U.ink_mask(furniture())
        self.assertEqual(int(m.sum()), 0)

    def test_shape_is_the_band(self):
        self.assertEqual(U.ink_mask(furniture()).shape, (U.BAND_H, U.BAND_W))

    def test_recovers_line_over_ink(self):
        """Lines are alpha-blended, so a crossed pixel keeps R/G and only lifts
        blue. It must still read as ink, or glyphs fragment. Painted as a block
        rather than a lone pixel because the despeckler drops isolated dots."""
        for rgb in [(25, 60, 130), (25, 60, 153), (25, 60, 175), (25, 100, 196)]:
            a = np.asarray(Image.new("RGB", (U.W, U.H), (240, 247, 255))).copy()
            a[U.BAND_TOP + 4:U.BAND_TOP + 9, 40:45] = rgb
            self.assertTrue(U.ink_mask(Image.fromarray(a))[6, 41],
                            "%s should read as ink" % (rgb,))

    def test_rejects_line_over_background(self):
        a = np.asarray(Image.new("RGB", (U.W, U.H), (240, 247, 255))).copy()
        a[U.BAND_TOP + 4:U.BAND_TOP + 9, 40:45] = (200, 214, 235)
        self.assertFalse(U.ink_mask(Image.fromarray(a))[6, 41])

    def test_despeckle_drops_residue(self):
        """The regression that corrupted the U, H and F templates: a 2px speck
        far below the text stretched the glyph bbox from 25 rows to 46."""
        im = paste_glyphs(furniture(), "AB")
        a = np.asarray(im).copy()
        a[U.BAND_BOT - 2, 30] = U.INK               # stray speck, 2px
        a[U.BAND_BOT - 2, 31] = U.INK
        m = U.ink_mask(Image.fromarray(a))
        heights = [U._crop(m, x0, x1)[0].shape[0] for x0, x1 in U.ink_runs(m)]
        self.assertTrue(all(h <= 30 for h in heights),
                        "speck stretched a glyph bbox: %s" % heights)

    def test_wrong_size_does_not_raise(self):
        for size in [(200, 60), (10, 10), (400, 200)]:
            text, conf = U.solve_image(Image.new("RGB", size, (255, 255, 255)))
            self.assertEqual(conf, 0.0)
            self.assertEqual(text, "")


class TestReading(unittest.TestCase):
    def test_reads_synthetic_composition(self):
        for word in ["ABC123", "1J5TQZ", "EFLPRS"]:
            im = paste_glyphs(furniture(), word)
            got, conf = U.solve_image(im)
            self.assertEqual(got, word)
            self.assertGreater(conf, 0.90)

    def test_reads_touching_glyphs(self):
        for word in ["ABC123", "1J5TQZ"]:
            im = paste_glyphs(furniture(), word, gap=0)
            self.assertEqual(U.solve_image(im)[0], word)

    def test_reads_overlapping_glyphs(self):
        """`W` is 35px against 27 for the next widest, so it collides with its
        neighbour: a merged WX spans 55 columns where W+X need 62. No vertical
        cut yields both — cutting to complete the W leaves an X-minus-left-edge
        that is a pixel-perfect `K`, which is exactly what a cut-based reader
        returned here."""
        for word, at, ov in [("WX3YTK", 1, 7), ("WHPTKJ", 1, 6), ("MWHW3H", 2, 5)]:
            im = paste_glyphs(furniture(), word, gap=1, overlap_at=at, overlap=ov)
            self.assertEqual(U.solve_image(im)[0], word,
                             "%s overlapping by %d at %d" % (word, ov, at))

    def test_confidence_falls_out_of_distribution(self):
        good = U.solve_image(paste_glyphs(furniture(), "ABC123"))[1]
        black = U.solve_image(Image.new("RGB", (U.W, U.H), (0, 0, 0)))[1]
        self.assertGreater(good, 0.90)
        self.assertLess(black, 0.50)

    def test_blank_reads_nothing(self):
        text, conf = U.solve_image(furniture())
        self.assertEqual(text, "")
        self.assertEqual(conf, 0.0)

    def test_substitution_margin_is_positive(self):
        _, margins = U.substitution_margin(U.ink_mask(paste_glyphs(furniture(), "ABC123")))
        self.assertEqual(len(margins), 6)
        for i, alt, margin in margins:
            self.assertGreater(margin, 0.0,
                               "position %d is a tie against %r" % (i, alt))


@unittest.skipUnless(os.path.isdir(CORPUS) and os.path.exists(LABELS),
                     "real corpus not present (it is gitignored)")
class TestRealCaptchas(unittest.TestCase):
    def test_hand_labelled_exact(self):
        with open(LABELS) as f:
            labels = json.load(f)
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        checked = 0
        for rel, want in labels.items():
            p = os.path.join(root, rel)
            if not os.path.exists(p):
                continue
            checked += 1
            self.assertEqual(U.solve_image(p)[0], want, rel)
        self.assertGreater(checked, 0)


class TestApiRouting(unittest.TestCase):
    def test_size_routes_to_udyam(self):
        from solver import api
        self.assertEqual(api._SIZES[(U.W, U.H)], "udyam")

    def test_no_ambiguous_entry(self):
        """0/I/O are absent from the alphabet, so there is nothing to avoid."""
        from solver import api
        self.assertNotIn("udyam", api.AMBIGUOUS)

    def test_solve_bytes_routes(self):
        from solver import api
        im = paste_glyphs(furniture(), "ABC123")
        text, conf, kind = api.solve_bytes(im)
        self.assertEqual(kind, "udyam")
        self.assertEqual(text, "ABC123")


if __name__ == "__main__":
    unittest.main(verbosity=2)
