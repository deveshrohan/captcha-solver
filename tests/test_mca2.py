"""MCA's captcha since Nov 2026 (kind `mca2`), same 200x80 as the old `mca`."""
import json
import unittest

import numpy as np
from PIL import Image

from solver import mca as M
from solver import mca2 as M2
from solver.api import _SIZES, solve_bytes, warmup
from tests.test_hc import ROOT, png


class TestMca2(unittest.TestCase):
    def test_200x80_is_split_by_colour_count(self):
        route = _SIZES[(200, 80)]
        self.assertTrue(callable(route))
        rng = np.random.default_rng(5)
        new = M2.make_rgb(M2.random_label(rng), rng)
        self.assertEqual(route(new), "mca2")
        # the old format: three flat colours, black ink over grey lines
        old = np.full((80, 200, 3), 230, np.uint8)
        old[30:33, :] = 150
        old[40:55, 70:120] = 0
        self.assertEqual(route(Image.fromarray(old)), "mca")

    def test_charset_matches_the_census(self):
        self.assertEqual(M2.CHARSET, "234679ABCDEFGHJKLMNPQRSTUVWXYZ")
        for c in "0158IO":
            self.assertNotIn(c, M2.CHARSET)

    def test_labels_use_the_charset_and_six_characters(self):
        labels = json.loads((ROOT / "mca2_labels.json").read_text())
        self.assertGreaterEqual(len(labels), 150)
        for text in labels.values():
            self.assertEqual(len(text), 6)
            self.assertTrue(set(text) <= set(M2.CHARSET), text)

    def test_split_is_pinned_and_disjoint(self):
        split = json.loads((ROOT / "mca2_split.json").read_text())
        test, val, train = (set(split[k]) for k in ("test", "validation", "train"))
        self.assertGreaterEqual(len(test), 60)
        self.assertFalse(test & val or test & train or val & train)

    def test_input_is_native_resolution(self):
        rng = np.random.default_rng(1)
        x = M2.make_input("ABC234", rng)
        self.assertEqual(x.shape, (M2.IN_H, M2.IN_W))
        self.assertEqual(x.dtype, np.uint8)

    def test_model_reads_its_own_renderer(self):
        rng = np.random.default_rng(99)
        hits = 0
        for _ in range(20):
            w = M2.random_label(rng)
            text, _, kind = solve_bytes(png(np.asarray(M2.make_rgb(w, rng))))
            self.assertEqual(kind, "mca2")
            hits += text == w
        self.assertGreaterEqual(hits, 14)

    def test_warmup(self):
        warmup(securimage=False, mca2=True)


if __name__ == "__main__":
    unittest.main()
