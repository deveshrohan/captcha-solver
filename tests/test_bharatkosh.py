#!/usr/bin/env python3
"""Tests for the Bharatkosh reader.

    python3 -m unittest discover -s tests -v

`eval_bharatkosh.py` scores the reader on real captchas; these cover the pieces
that need no trained model. The structural facts under test, each one a decision
the reader depends on:

* the two opaque bars sit at FIXED rows (14-16, 24-26) and are drawn last, so the
  input zeroes those rows for real and synthetic alike -- one blind band, never
  200 different bar colours;
* isolated pastel specks are removed without eating connected ink;
* one answer is observable through several independent renders (`?New=0`), so
  the read is a VOTE: summed CTC likelihood must recover a word that no single
  render reads correctly, and must not depend on render order.
"""
import os
import sys
import unittest

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solver import bharatkosh as B

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = os.path.join(ROOT, "bharatkosh_raw")


def _lp_for(word, wrong_at=None, wrong_ch=None, T=30, p=0.9):
    """A hand-built (T, C) CTC log-prob matrix that spells `word`, optionally
    preferring `wrong_ch` at one position (true char kept as runner-up)."""
    C = B.N_CLASSES + 1
    probs = np.full((T, C), 1e-4)
    probs[:, B.BLANK] = 1.0
    for i, ch in enumerate(word):
        t = 2 + i * 4
        probs[t, B.BLANK] = 1e-4
        if i == wrong_at:
            probs[t, B.CH2I[wrong_ch]] = p
            probs[t, B.CH2I[ch]] = 1 - p
        else:
            probs[t, B.CH2I[ch]] = 1.0
    probs /= probs.sum(1, keepdims=True)
    return torch.tensor(np.log(probs), dtype=torch.float32)


class TestCharset(unittest.TestCase):
    def test_shape(self):
        self.assertEqual(B.LENGTH, 6)
        self.assertEqual(B.N_CLASSES, len(B.CHARSET))
        self.assertEqual(B.BLANK, B.N_CLASSES)
        self.assertEqual(len(set(B.CHARSET)), len(B.CHARSET))

    def test_case_sensitive(self):
        self.assertIn("a", B.CHARSET)
        self.assertIn("A", B.CHARSET)


class TestInput(unittest.TestCase):
    def _img(self):
        a = np.full((B.H, B.W, 3), 255, np.uint8)
        a[5:35, 40:46] = (60, 90, 160)            # a stroke crossing both bars
        a[2, 100] = (200, 180, 230)               # an isolated pastel speck
        for y0, y1 in B.BAR_ROWS:
            a[y0:y1, 1:150] = (220, 200, 240)     # the bars, drawn last
        return a

    def test_bar_rows_are_blind(self):
        d = B.ink_map(self._img())
        for y0, y1 in B.BAR_ROWS:
            self.assertEqual(float(d[y0:y1].max()), 0.0)

    def test_isolated_speck_removed_ink_kept(self):
        d = B.ink_map(self._img())
        self.assertEqual(float(d[2, 100]), 0.0)
        self.assertGreater(float(d[8, 42]), 0.5)
        self.assertGreater(float(d[20, 42]), 0.5)  # between the bars

    def test_input_shape(self):
        x = B._to_input(Image.fromarray(self._img()))
        self.assertEqual(x.shape, (B.IN_H, B.IN_W))
        self.assertEqual(x.dtype, np.uint8)

    def test_rejects_wrong_size(self):
        with self.assertRaises(ValueError):
            B.load_real(Image.new("RGB", (150, 42), "white"))


class TestGenerator(unittest.TestCase):
    def test_render_has_bars_at_measured_rows(self):
        rng = np.random.default_rng(3)
        im = B.make_rgb("aB3dEf", rng)
        self.assertEqual(im.size, (B.W, B.H))
        a = np.asarray(im).astype(int)
        for y0, y1 in B.BAR_ROWS:
            for y in range(y0, y1):
                self.assertTrue((a[y, 1:150] == a[y, 1]).all())
                self.assertFalse((a[y, 1] == 255).all())

    def test_synthetic_input_matches_real_treatment(self):
        rng = np.random.default_rng(4)
        x = B.make_input(B.random_label(rng), rng)
        self.assertEqual(x.shape, (B.IN_H, B.IN_W))
        self.assertGreater(int((x > 100).sum()), 500)   # there is ink


class TestVote(unittest.TestCase):
    def test_recovers_word_no_single_render_reads(self):
        word = "K8AyRy"
        lps = [_lp_for(word, 2, "4"), _lp_for(word, 0, "k"), _lp_for(word, 5, "v")]
        for lp in lps:                       # every render alone is wrong
            self.assertNotEqual(B.beam(lp, top=1)[0][0], word)
        got, conf = B.vote(lps)
        self.assertEqual(got, word)
        self.assertGreater(conf, 0.5)

    def test_order_invariant(self):
        word = "pf8dkY"
        lps = [_lp_for(word, 1, "t"), _lp_for(word, 4, "K"), _lp_for(word)]
        a = B.vote(lps)
        b = B.vote(lps[::-1])
        self.assertEqual(a[0], b[0])
        self.assertAlmostEqual(a[1], b[1], places=5)

    def test_single_render_vote_is_its_beam(self):
        lp = _lp_for("h38J5Q")
        self.assertEqual(B.vote([lp])[0], "h38J5Q")


class TestModel(unittest.TestCase):
    def test_forward_shape(self):
        m = B.BharatkoshCRNN()
        x = torch.zeros(2, 1, B.IN_H, B.IN_W)
        out = m(x)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[2], B.N_CLASSES + 1)
        self.assertGreaterEqual(out.shape[1], 2 * B.LENGTH + 1)


class TestApi(unittest.TestCase):
    def test_routes_by_size(self):
        from solver import api
        self.assertEqual(api._SIZES[(150, 40)], "bharatkosh")
        self.assertEqual(api._SIZES[(150, 42)], "itat")

    def test_rerender_helper_spends_renders_before_texts(self):
        from solver import api
        calls = []
        confs = iter([0.3, 0.5, 0.95])
        orig = api.solve_bharatkosh_group
        api.solve_bharatkosh_group = lambda ims: ("abc123", next(confs))
        try:
            out = api.solve_bharatkosh_with_rerender(
                lambda new: calls.append(new) or b"", min_conf=0.9)
        finally:
            api.solve_bharatkosh_group = orig
        self.assertEqual(calls, [True, False, False])    # one text, three looks
        self.assertEqual(out, ("abc123", 0.95, 3, 1))


@unittest.skipUnless(os.path.isdir(CORPUS), "real corpus not present")
class TestRealCorpus(unittest.TestCase):
    def test_bars_where_measured(self):
        fs = sorted(f for f in os.listdir(CORPUS) if f.endswith(".png"))[:25]
        for f in fs:
            a = np.asarray(Image.open(os.path.join(CORPUS, f)).convert("RGB")).astype(int)
            for y0, y1 in B.BAR_ROWS:
                for y in range(y0, y1):
                    self.assertTrue((a[y, 1:150] == a[y, 1]).all(), f)


if __name__ == "__main__":
    unittest.main()
