#!/usr/bin/env python3
"""Tests for the NGT reader.

    python3 -m unittest discover -s tests -v

`eval_ngt.py` scores the reader against hand-labelled real captchas; these cover
the structural facts the reader *depends* on, each of which is a decision that
would silently break the reader if the generator moved:

* the ink is **exactly** black and the noise floor is 150, so the mask needs no
  threshold and cannot be fooled by a noise pixel;
* the noise is painted **underneath** the text, so ink is never damaged — this
  is what makes a pixel-exact cover legitimate at all, and `make_image` has to
  reproduce that layer order or the round-trip tests would be testing fiction;
* the six cells sit on a **fixed** grid, so segmentation is arithmetic;
* `1`, `l` and `i` are only 6 pixels apart, which is why matching is exact-or-
  bust: a nearest-neighbour fallback would flip them silently;
* `120x40` collides with the gstat numeric captcha, so routing is a content test
  and both directions have to hold.
"""
import os
import sys
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solver import ngt as N

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = os.path.join(ROOT, "ngt_raw")
LABELS = os.path.join(ROOT, "ngt_labels.json")
GSTAT_CORPUS = os.path.join(ROOT, "corpus")


class TestCharset(unittest.TestCase):
    def test_size_and_length(self):
        self.assertEqual(len(N.CHARSET), 36)
        self.assertEqual(N.LENGTH, 6)

    def test_is_lowercase_alnum(self):
        self.assertEqual(sorted(N.CHARSET),
                         sorted("0123456789abcdefghijklmnopqrstuvwxyz"))

    def test_random_label_in_charset(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            lab = N.random_label(rng)
            self.assertEqual(len(lab), N.LENGTH)
            self.assertTrue(set(lab) <= set(N.CHARSET))


class TestGeometry(unittest.TestCase):
    def test_grid_matches_the_measurement(self):
        self.assertEqual((N.W, N.H), (120, 40))
        self.assertEqual((N.X0, N.PITCH), (20, 9))
        self.assertEqual((N.R0, N.R1), (13, 25))
        self.assertEqual((N.CELL_W, N.CELL_H), (9, 12))

    def test_grid_fits_inside_the_image(self):
        self.assertLessEqual(N.X0 + N.PITCH * N.LENGTH, N.W)
        self.assertLessEqual(N.R1, N.H)

    def test_cells_are_contiguous_and_disjoint(self):
        mask = np.zeros((N.H, N.W), bool)
        seen = np.zeros((N.H, N.W), int)
        for x0, x1 in N.cell_spans():
            seen[N.R0:N.R1, x0:x1] += 1
        self.assertEqual(seen.max(), 1, "cells overlap")
        self.assertEqual(int(seen.sum()), N.LENGTH * N.CELL_W * N.CELL_H)
        self.assertFalse(mask.any())


class TestGlyphTable(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.g = N.glyphs()

    def test_covers_the_charset_exactly(self):
        self.assertEqual(sorted(self.g), sorted(N.CHARSET))

    def test_bitmaps_have_the_cell_shape(self):
        for c, b in self.g.items():
            self.assertEqual(b.shape, (N.CELL_H, N.CELL_W), c)
            self.assertEqual(b.dtype, np.bool_, c)

    def test_every_glyph_carries_ink(self):
        for c, b in self.g.items():
            self.assertGreater(int(b.sum()), 0, c)

    def test_all_36_are_mutually_distinct(self):
        seen = {}
        for c, b in self.g.items():
            key = b.tobytes()
            self.assertNotIn(key, seen,
                             "%r and %r share a bitmap" % (c, seen.get(key)))
            seen[key] = c

    def test_bar_glyphs_stay_separated(self):
        """`1`, `l` and `i` are the tightest cluster in the font at Hamming 6.

        Exactness is what makes them safe, so this is the regression guard: if a
        rebuild of the table ever collapsed two of them, every other test would
        still pass while the reader silently confused them."""
        for a, b in (("1", "l"), ("1", "i"), ("i", "l")):
            d = int((self.g[a] ^ self.g[b]).sum())
            self.assertGreaterEqual(d, 6, "%s/%s only %d px apart" % (a, b, d))

    def test_classic_homoglyphs_are_far_apart(self):
        """The pairs that cap the MCA reader at 75% are not close here."""
        for a, b, lo in (("0", "o", 20), ("5", "s", 20), ("9", "g", 20)):
            self.assertGreater(int((self.g[a] ^ self.g[b]).sum()), lo)


class TestInkModel(unittest.TestCase):
    def test_exact_black_is_ink(self):
        im = Image.new("RGB", (N.W, N.H), "white")
        im.putpixel((5, 5), (0, 0, 0))
        self.assertTrue(N.ink_mask(im)[5, 5])

    def test_the_noise_floor_is_not_ink(self):
        """Every non-white pixel measured had all channels in [150,255]. The
        darkest possible noise pixel must still fall outside the mask."""
        im = Image.new("RGB", (N.W, N.H), "white")
        im.putpixel((7, 7), (N.NOISE_LO, N.NOISE_LO, N.NOISE_LO))
        self.assertFalse(N.ink_mask(im)[7, 7])

    def test_near_black_is_not_ink(self):
        """The test is equality, not a threshold: #333 (gstat's ink) is not
        NGT ink, which is also what keeps the router honest."""
        im = Image.new("RGB", (N.W, N.H), "white")
        im.putpixel((9, 9), (51, 51, 51))
        self.assertFalse(N.ink_mask(im)[9, 9])

    def test_noise_beside_a_stroke_does_not_join_the_mask(self):
        im = Image.new("RGB", (N.W, N.H), "white")
        im.putpixel((20, 20), (0, 0, 0))
        im.putpixel((21, 20), (150, 200, 255))
        m = N.ink_mask(im)
        self.assertTrue(m[20, 20])
        self.assertFalse(m[20, 21])

    def test_accepts_bytes_and_path(self):
        import io
        im = N.make_image("abc123", np.random.default_rng(1))
        buf = io.BytesIO()
        im.save(buf, "PNG")
        self.assertEqual(N.ink_mask(buf.getvalue()).shape, (N.H, N.W))


class TestRenderer(unittest.TestCase):
    def test_native_size_and_mode(self):
        im = N.make_image("kqip8x", np.random.default_rng(2))
        self.assertEqual(im.size, (N.W, N.H))

    def test_noise_never_damages_ink(self):
        """The layer order is the whole basis of the exact cover: the generator
        paints 50 noise pixels and *then* the text, so no stroke can be holed.

        Rendered with the noise on top instead, this test fails — which is the
        point of asserting it rather than assuming it."""
        rng = np.random.default_rng(3)
        for _ in range(25):
            lab = N.random_label(rng)
            text, conf = N.solve_image(N.make_image(lab, rng))
            self.assertEqual(text, lab)
            self.assertEqual(conf, 1.0)

    def test_noise_pixel_count_matches_the_palette(self):
        """PLTE is 52 entries in every real image: white, black and exactly 50
        noise colours."""
        self.assertEqual(N.NOISE_N, 50)

    def test_noise_is_in_the_measured_band(self):
        rng = np.random.default_rng(4)
        a = np.asarray(N.make_image("mnpqrs", rng).convert("RGB"), np.int16)
        ink = (a == 0).all(2)
        white = (a == 255).all(2)
        noise = a[~ink & ~white]
        if len(noise):
            self.assertGreaterEqual(int(noise.min()), N.NOISE_LO)
            self.assertLessEqual(int(noise.max()), N.NOISE_HI)

    def test_ink_lands_only_on_the_grid(self):
        rng = np.random.default_rng(5)
        m = N.ink_mask(N.make_image("wxyz09", rng))
        off = m.copy()
        off[N.R0:N.R1, N.X0:N.X0 + N.PITCH * N.LENGTH] = False
        self.assertEqual(int(off.sum()), 0)


class TestReading(unittest.TestCase):
    def test_round_trip_over_the_whole_charset(self):
        """Every character must survive a render/read cycle in every column."""
        rng = np.random.default_rng(6)
        cs = N.CHARSET
        for i in range(0, len(cs), N.LENGTH):
            lab = (cs[i:i + N.LENGTH] + cs)[:N.LENGTH]
            text, conf = N.solve_image(N.make_image(lab, rng))
            self.assertEqual(text, lab)
            self.assertEqual(conf, 1.0)

    def test_every_char_reads_in_every_position(self):
        rng = np.random.default_rng(7)
        for c in N.CHARSET:
            for k in range(N.LENGTH):
                lab = list("aaaaaa")
                lab[k] = c
                lab = "".join(lab)
                text, _ = N.solve_image(N.make_image(lab, rng))
                self.assertEqual(text, lab, "%r in column %d" % (c, k))

    def test_confidence_is_one_only_for_an_exact_cover(self):
        rng = np.random.default_rng(8)
        im = N.make_image("h1ejn4", rng)
        self.assertEqual(N.solve_image(im)[1], 1.0)

    def test_a_single_flipped_pixel_drops_confidence(self):
        """A wrong glyph is only ~6 px against ~190 of ink, so an uncapped
        ink-ratio would read 0.97 and pass a 0.90 gate. It must not."""
        rng = np.random.default_rng(9)
        im = N.make_image("h1ejn4", rng).convert("RGB")
        m = N.ink_mask(im)
        ys, xs = np.nonzero(m)
        im.putpixel((int(xs[0]), int(ys[0])), (255, 255, 255))
        conf = N.solve_image(im)[1]
        self.assertLess(conf, 1.0)
        self.assertLessEqual(conf, 0.99)

    def test_stray_ink_outside_the_grid_drops_confidence(self):
        rng = np.random.default_rng(10)
        im = N.make_image("h1ejn4", rng).convert("RGB")
        im.putpixel((2, 35), (0, 0, 0))
        self.assertLess(N.solve_image(im)[1], 1.0)

    def test_blank_image_reads_nothing(self):
        im = Image.new("RGB", (N.W, N.H), "white")
        text, conf = N.solve_image(im)
        self.assertEqual(text, "")
        self.assertEqual(conf, 0.0)

    def test_wrong_size_does_not_raise(self):
        for size in ((100, 30), (215, 80), (1, 1)):
            with self.subTest(size=size):
                text, conf = N.solve_image(Image.new("RGB", size, "white"))
                self.assertIsInstance(text, str)
                self.assertGreaterEqual(conf, 0.0)


class TestRouting(unittest.TestCase):
    def test_ngt_image_routes_to_ngt(self):
        from solver import api
        im = N.make_image("kqip8x", np.random.default_rng(11))
        self.assertEqual(api.solve_bytes(im)[2], "ngt")

    def test_gstat_like_image_routes_to_gstat(self):
        """gstat's palette is #333 ink and #ccc lines: it cannot emit a pure
        black pixel, which is exactly what the router keys on."""
        from solver import api
        im = Image.new("RGB", (120, 40), "white")
        for x in range(10, 110):
            im.putpixel((x, 20), (204, 204, 204))
        for x in range(15, 100, 3):
            im.putpixel((x, 18), (51, 51, 51))
        self.assertEqual(api.solve_bytes(im)[2], "gstat")

    @unittest.skipUnless(os.path.isdir(GSTAT_CORPUS), "gstat corpus absent")
    def test_no_gstat_image_trips_the_threshold(self):
        import glob
        from solver import api
        ps = sorted(glob.glob(os.path.join(GSTAT_CORPUS, "*.png")))[:300]
        if not ps:
            self.skipTest("gstat corpus empty")
        for p in ps:
            im = Image.open(p).convert("RGB")
            if im.size != (120, 40):
                continue
            self.assertEqual(api._route_120x40(im), "gstat", p)


class TestRealCorpus(unittest.TestCase):
    @unittest.skipUnless(os.path.isdir(CORPUS), "ngt corpus absent")
    def test_every_real_cell_is_an_exact_template_match(self):
        """The strong claim, checked without needing a single label: if the
        cover is exact then every cell of every real image matches a template
        byte for byte, and confidence is 1.0 everywhere."""
        import glob
        ps = sorted(glob.glob(os.path.join(CORPUS, "*.png")))
        if not ps:
            self.skipTest("ngt corpus empty")
        bad = []
        for p in ps:
            text, conf = N.solve_image(p)
            if conf != 1.0 or len(text) != N.LENGTH:
                bad.append((os.path.basename(p), text, conf))
        self.assertEqual(bad, [], "%d/%d real images not an exact cover"
                         % (len(bad), len(ps)))


if __name__ == "__main__":
    unittest.main()
