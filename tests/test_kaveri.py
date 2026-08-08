#!/usr/bin/env python3
"""Tests for the Kaveri reader.

    python3 -m unittest discover -s tests -v

The rest of the repo verifies through `eval_*.py` against real captchas, which
this reader also does (`eval_kaveri.py`: 210/210 exact and unique). These cover
the parts real samples exercise rarely or not at all — the transparent-band
trap, overlapping glyphs, and the degraded path that only runs if the generator
changes.
"""
import os
import sys
import unittest

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solver import kaveri as K


def render(text, x0=23, gap=7, lines=0, seed=0):
    """Compose an image the way the generator does: mintcream, coloured lines
    UNDER pure-black sprites, and a fully transparent bottom band."""
    img = Image.new("RGBA", (K.W, K.H), K.BACKGROUND + (255,))
    d = ImageDraw.Draw(img)
    rng = np.random.default_rng(seed)
    for _ in range(lines):
        c = tuple(int(v) for v in rng.integers(1, 250, size=3))
        if c == (0, 0, 0):
            c = (1, 1, 1)
        d.line([int(rng.integers(0, K.W)), int(rng.integers(0, 50)),
                int(rng.integers(0, K.W)), int(rng.integers(0, 50))], fill=c + (255,))
    a = np.asarray(img).copy()
    G = K.glyphs()
    x = x0
    for ch in text:
        g, top = G[ch]
        gh, gw = g.shape
        if x < 0 or x + gw > K.W:
            raise ValueError(f"{text!r} does not fit at gap={gap}")
        a[top:top + gh, x:x + gw][g] = (0, 0, 0, 255)
        x += gw + gap
    a[K.PAINTED_H:, :, 3] = 0          # the transparent band the generator leaves
    a[K.PAINTED_H:, :, :3] = 0
    return Image.fromarray(a)


class TestGlyphLibrary(unittest.TestCase):
    def test_full_charset(self):
        G = K.glyphs()
        self.assertEqual("".join(sorted(G)), K.CHARSET)
        self.assertEqual(len(G), 36)

    def test_sprites_are_tight_and_on_the_baseline(self):
        for c, (g, top) in K.glyphs().items():
            self.assertTrue(g[:, 0].any(), f"{c}: left column empty")
            self.assertTrue(g[:, -1].any(), f"{c}: right column empty")
            self.assertTrue(g[0].any() and g[-1].any(), f"{c}: top/bottom row empty")
            self.assertIn(top, (14, 15), f"{c}: unexpected top row {top}")

    def test_zero_and_oh_are_distinguishable(self):
        """The pair that caps the MCA reader at 75% is separable here."""
        G = K.glyphs()
        self.assertNotEqual(G["0"][0].shape, G["O"][0].shape)
        self.assertNotEqual(G["1"][0].shape, G["I"][0].shape)


class TestInkMask(unittest.TestCase):
    def test_transparent_band_is_not_ink(self):
        m = K.ink_mask(render("ABC123"))
        self.assertEqual(m.shape, (K.PAINTED_H, K.W))
        self.assertTrue(m.any())

    def test_rgb_flattened_input_still_reads(self):
        """PIL maps transparent pixels to (0,0,0) on convert("RGB") — the exact
        value the ink test looks for. The crop to PAINTED_H must absorb that."""
        rgba = render("ABC123")
        flat = rgba.convert("RGB")
        self.assertEqual(np.asarray(flat)[55, 10].tolist(), [0, 0, 0])  # trap is real
        self.assertEqual(K.solve_image(flat)[0], "ABC123")

    def test_noise_lines_do_not_enter_the_mask(self):
        clean = K.ink_mask(render("XY7Z90", lines=0))
        noisy = K.ink_mask(render("XY7Z90", lines=12, seed=3))
        np.testing.assert_array_equal(clean, noisy)

    def test_wrong_size_does_not_raise(self):
        text, conf = K.solve_image(Image.new("RGB", (64, 32), (245, 255, 250)))
        self.assertIsInstance(text, str)
        self.assertEqual(conf, 0.0)


class TestSolve(unittest.TestCase):
    def test_roundtrip_over_the_whole_charset(self):
        cs = K.CHARSET
        for i in range(0, len(cs), 6):
            word = (cs[i:i + 6] + cs)[:6]
            self.assertEqual(K.solve_image(render(word))[0], word)

    def test_confidence_is_one_for_an_exact_cover(self):
        self.assertEqual(K.solve_image(render("Q1J0IW"))[1], 1.0)

    def test_reading_is_unique(self):
        sols = K.solutions(K.ink_mask(render("M8G4KZ")), limit=5)
        self.assertEqual(sols, ["M8G4KZ"])

    def test_overlapping_glyphs(self):
        """`J` tucks under `T`'s crossbar, so their x-ranges overlap and a
        disjoint left-to-right segmentation reads them wrong. 1 image in 198."""
        img = render("I4MTJY", gap=-1)
        m = K.ink_mask(img)
        self.assertLess(len(K.ink_runs(m)), 6)      # segmentation cannot separate them
        self.assertEqual(K.solve(m)[0], "I4MTJY")   # union cover still reads it

    def test_touching_glyphs_with_zero_gap(self):
        m = K.ink_mask(render("LJLJLJ", gap=0))
        self.assertEqual(K.solve(m)[0], "LJLJLJ")

    def test_corrupted_ink_degrades_instead_of_lying(self):
        """A generator change must show up as falling confidence, because that
        is the signal `selfimprove check` and `solve_with_retry` act on."""
        img = render("ABCDEF")
        a = np.asarray(img).copy()
        a[20:30, 30:120] = (245, 255, 250, 255)     # erase a band of ink
        text, conf = K.solve(K.ink_mask(Image.fromarray(a)))
        self.assertLess(conf, 1.0)
        self.assertIsInstance(text, str)

    def test_blank_image_is_zero_confidence(self):
        text, conf = K.solve(np.zeros((K.PAINTED_H, K.W), dtype=bool))
        self.assertEqual((text, conf), ("", 0.0))


class TestBoundedWork(unittest.TestCase):
    """The cover search is exhaustive, so it must refuse hopeless masks rather
    than explore them. A solid ink block admits ~70 placements per position;
    unguarded it runs for hours, which in a scrape loop is worse than a wrong
    read."""

    def _timed(self, mask, budget=5.0):
        import time
        t0 = time.time()
        sols = K.solutions(mask, limit=2)
        text, conf = K.solve(mask)
        dt = time.time() - t0
        self.assertLess(dt, budget, f"took {dt:.1f}s")
        return sols, text, conf

    def test_solid_block_returns_fast_with_low_confidence(self):
        m = np.zeros((K.PAINTED_H, K.W), dtype=bool)
        m[14:40, :] = True
        sols, _, conf = self._timed(m)
        self.assertEqual(sols, [])
        self.assertLess(conf, 0.5)

    def test_ink_bound_rejects_impossible_masks(self):
        m = np.ones((K.PAINTED_H, K.W), dtype=bool)
        self.assertGreater(int(m.sum()), K._max_ink(K.LENGTH))
        self.assertEqual(K.solutions(m, limit=1), [])

    def test_ink_bound_cannot_reject_a_real_captcha(self):
        """The guard is only safe if it sits above the heaviest real image.
        Measured over 210 real captchas: 589..1081 ink pixels."""
        self.assertGreater(K._max_ink(K.LENGTH), 1081)

    def test_widest_realistic_text_still_reads(self):
        """The bound must not trip on an all-wide-glyph captcha."""
        text, conf = K.solve_image(render("WWMMQO", gap=1))
        self.assertEqual((text, conf), ("WWMMQO", 1.0))


class TestApiRouting(unittest.TestCase):
    def test_size_routes_to_kaveri(self):
        import io
        from solver import api
        buf = io.BytesIO()
        render("7PDR5N").save(buf, format="PNG")
        text, conf, kind = api.solve_bytes(buf.getvalue())
        self.assertEqual((text, conf, kind), ("7PDR5N", 1.0, "kaveri"))

    def test_kaveri_has_no_ambiguous_characters(self):
        from solver import api
        self.assertEqual(api.AMBIGUOUS.get("kaveri", ""), "")

    def test_warmup_needs_no_model(self):
        from solver import api
        api.warmup(securimage=False, kaveri=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
