#!/usr/bin/env python3
"""Tests for the ITAT reader.

    python3 -m unittest discover -s tests -v

`eval_itat.py` verifies the reader against real captchas; these cover the pieces
that do not need a trained model — the darkness projection that separates ink
from the light-blue noise, the pinned synthetic geometry, the case-insensitive
charset — plus a model round-trip and a real-corpus check that run only when the
weights / corpus are present (both are gitignored).

The key structural facts under test, each a decision the reader depends on:
* the noise is light-blue and drawn OVER near-black text, so a darkness cut both
  removes it AND punches holes where an opaque line crosses a stroke — the reader
  must be trained on that fragmentation, so the projection must actually show it;
* the alphabet is homoglyph-free and case is FOLDED, so there is no ambiguous
  pair to retry around (unlike the case-sensitive MCA reader).
"""
import os
import sys
import unittest

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solver import itat as I

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(ROOT, "solver", "itat_model.pt")
CORPUS = os.path.join(ROOT, "itat_raw")
LABELS = os.path.join(ROOT, "itat_labels.json")


class TestCharset(unittest.TestCase):
    def test_size_and_length(self):
        self.assertEqual(I.N_CLASSES, len(I.CHARSET))
        self.assertEqual(I.LENGTH, 6)
        self.assertEqual(I.BLANK, I.N_CLASSES)

    def test_homoglyph_free(self):
        """0/1/I/O/L never occur in the census; excluding them removes the
        digit/letter homoglyphs (L because a lowercase l is a bare bar), and
        folding case removes the rest."""
        for ch in "01IOL":
            self.assertNotIn(ch, I.CHARSET)

    def test_charset_is_upper_and_digits_only(self):
        """The reader is case-insensitive: labels live in one uppercase class,
        so no lowercase letter is ever a class."""
        self.assertTrue(all(c.isdigit() or c.isupper() for c in I.CHARSET))

    def test_no_ambiguous_entry(self):
        from solver import api
        self.assertEqual(api.AMBIGUOUS.get("itat", ""), "")

    def test_random_label_in_charset(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            lab = I.random_label(rng)
            self.assertEqual(len(lab), I.LENGTH)
            self.assertTrue(all(c in I.CH2I for c in lab))

    def test_encode_roundtrip(self):
        self.assertEqual([I.I2CH[i] for i in I.encode_label("AB3K7M")], list("AB3K7M"))


class TestDarknessProjection(unittest.TestCase):
    """`d = 1 - max(r,g,b)/255`: ink -> ~1, white/light-blue -> ~0."""

    def _d(self, rgb):
        return I.darkness(Image.new("RGB", (4, 4), rgb))[0, 0]

    def test_black_ink_is_bright(self):
        self.assertGreater(self._d((0, 0, 0)), 0.99)

    def test_white_is_dark(self):
        self.assertLess(self._d((255, 255, 255)), 0.01)

    def test_light_blue_noise_is_dark(self):
        """Every measured noise-line colour must project to near-zero, or the
        model would have to learn to ignore colour it never should have seen."""
        for col in I._LINE_BLUES:
            self.assertLess(self._d(col), 0.2, "%s not suppressed" % (col,))

    def test_line_over_ink_punches_a_hole(self):
        """An opaque blue line crossing black text drops the darkness there, i.e.
        it fragments the stroke. The reader is trained on that, so the projection
        must expose it rather than hide it."""
        blue = I._LINE_BLUES[0]
        self.assertLess(self._d(blue), self._d((0, 0, 0)) - 0.5)

    def test_load_real_shape_and_dtype(self):
        im = Image.new("RGB", (I.W, I.H), (255, 255, 255))
        out = I.load_real(im)
        self.assertEqual(out.shape, (I.IN_H, I.IN_W))
        self.assertEqual(out.dtype, np.uint8)

    def test_load_real_accepts_bytes_and_path(self):
        import io
        buf = io.BytesIO()
        Image.new("RGB", (I.W, I.H), (255, 255, 255)).save(buf, format="PNG")
        b = buf.getvalue()
        self.assertEqual(I.load_real(b).shape, (I.IN_H, I.IN_W))


class TestRenderer(unittest.TestCase):
    def test_make_rgb_is_native_size(self):
        rng = np.random.default_rng(1)
        im = I.make_rgb(I.random_label(rng), rng)
        self.assertEqual(im.size, (I.W, I.H))
        self.assertEqual(im.mode, "RGB")

    def test_make_input_shape(self):
        rng = np.random.default_rng(2)
        x = I.make_input(I.random_label(rng), rng)
        self.assertEqual(x.shape, (I.IN_H, I.IN_W))

    def test_geometry_is_pinned_to_the_real_target(self):
        """Whatever the font's native proportions, the word's ink bbox is scaled
        to the measured real ~116x31, so synthetic never drifts off the real
        geometry (the same pinning the GST reader uses)."""
        rng = np.random.default_rng(3)
        Hs, Ws = [], []
        for _ in range(40):
            d = I.darkness(I.make_rgb(I.random_label(rng), rng))
            ink = d > 0.6
            ys, xs = np.where(ink)
            Hs.append(ys.max() - ys.min() + 1)
            Ws.append(xs.max() - xs.min() + 1)
        self.assertTrue(26 <= np.median(Hs) <= 34, "median H %.0f" % np.median(Hs))
        self.assertTrue(105 <= np.median(Ws) <= 128, "median W %.0f" % np.median(Ws))

    def test_case_is_rendered_both_ways(self):
        """The model is case-insensitive, so the generator must draw a letter as
        both cases (mapping both shapes to one class). Lower/upper glyphs must
        therefore differ in bitmap."""
        fnt = I._font(I.FONT_FILES[0], I.FONT_PX * I.SS)
        up, uw, _ = I._glyph_alpha("P", fnt, 0.0)
        lo, lw, _ = I._glyph_alpha("p", fnt, 0.0)
        self.assertNotEqual((up.size, uw), (lo.size, lw))

    def test_baseline_sits_low(self):
        """Ink should occupy the lower band (baseline ~y40 of 42), not float."""
        rng = np.random.default_rng(5)
        d = I.darkness(I.make_rgb("ABCDEF", rng))
        ys = np.where((d > 0.6).any(1))[0]
        self.assertGreater(ys.max(), 30)


class TestApiRouting(unittest.TestCase):
    def test_size_routes_to_itat(self):
        from solver import api
        self.assertEqual(api._SIZES[(I.W, I.H)], "itat")

    def test_size_is_distinct_from_epfo(self):
        """150x42 must not collide with EPFO's 150x50."""
        from solver import api
        self.assertNotEqual((I.W, I.H), (150, 50))
        self.assertIn((150, 50), api._SIZES)


@unittest.skipUnless(os.path.exists(MODEL), "trained model not present (gitignored until built)")
class TestModelRoundTrip(unittest.TestCase):
    def test_reads_its_own_synthetic(self):
        import torch
        model = I.ItatCRNN()
        model.load_state_dict(torch.load(MODEL, map_location="cpu"))
        rng = np.random.default_rng(7)
        labels = [I.random_label(rng) for _ in range(64)]
        X = np.stack([I.make_input(l, rng) for l in labels])
        preds, confs = I.predict(model, X, "cpu")
        chars = sum(sum(a == b for a, b in zip(p, l)) for p, l in zip(preds, labels))
        acc = chars / (len(labels) * I.LENGTH)
        self.assertGreater(acc, 0.9, "synthetic self-consistency only %.2f" % acc)
        self.assertTrue(all(len(p) == I.LENGTH for p in preds))


@unittest.skipUnless(os.path.isdir(CORPUS) and os.path.exists(LABELS),
                     "real corpus not present (gitignored)")
class TestRealCaptchas(unittest.TestCase):
    def test_held_out_accuracy(self):
        import json
        import torch
        model = I.ItatCRNN()
        model.load_state_dict(torch.load(MODEL, map_location="cpu"))
        with open(LABELS) as f:
            labels = json.load(f)
        items = [(os.path.join(ROOT, p) if not os.path.isabs(p) else p, l.upper())
                 for p, l in labels.items()]
        items = [(p, l) for p, l in items
                 if os.path.exists(p) and len(l) == I.LENGTH and all(c in I.CH2I for c in l)]
        self.assertGreater(len(items), 0)
        X = np.stack([I.load_real(p) for p, _ in items])
        preds, _ = I.predict(model, X, "cpu")
        chars = sum(sum(a == b for a, b in zip(p, l)) for p, (_, l) in zip(preds, items))
        acc = chars / (len(items) * I.LENGTH)
        self.assertGreater(acc, 0.7, "real char accuracy only %.2f" % acc)


if __name__ == "__main__":
    unittest.main(verbosity=2)
