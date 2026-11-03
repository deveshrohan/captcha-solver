"""CESTAT reader. The synthetic tests compose captchas from the shipped glyph
renders, so they run anywhere. The real tests score the gitignored
cestat_raw/ captures and skip when those are absent."""
import io
import json
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from solver import cestat as C

ROOT = Path(__file__).resolve().parents[1]
BG = np.array((72, 240, 240), np.float32)


def cestat_image(word, dx=0, dy=0, quality=None):
    """The portal's composition: black glyphs on cyan, pen 5 + 23*i,
    baseline 26. Cells 4 and 5 fall (almost) off the 100 px image, as on
    the real site."""
    _, tp = C.templates()
    cov = np.zeros((C.H, C.W), np.float32)
    for i, c in enumerate(word):
        left, g = tp[c]
        x0 = C.PEN0 + C.PITCH * i + left + dx
        a, b = max(0, x0), min(C.W, x0 + g.shape[1])
        if b > a:
            cov[:, a:b] = np.maximum(cov[:, a:b], g[:, a - x0:b - x0])
    cov = np.roll(cov, dy, 0)
    rgb = BG[None, None, :] * (1 - cov[:, :, None])
    out = io.BytesIO()
    im = Image.fromarray(np.rint(rgb).astype(np.uint8))
    if quality:
        im.save(out, format="JPEG", quality=quality)
    else:
        im.save(out, format="PNG")
    return out.getvalue()


class TestCestatSynthetic(unittest.TestCase):
    def test_reads_the_four_visible_characters(self):
        for word in ("L1kM6x", "G0fXab", "WMmw7J", "Il1jy9"):
            text, conf = C.solve_image(cestat_image(word))
            self.assertEqual(text, word[:C.VISIBLE], word)
            self.assertGreaterEqual(conf, 0.35, word)

    def test_survives_jpeg(self):
        text, conf = C.solve_image(cestat_image("Q3Xbzz", quality=75))
        self.assertEqual(text, "Q3Xb")
        self.assertGreaterEqual(conf, 0.35)

    def test_charset_has_no_O(self):
        self.assertNotIn("O", C.CHARSET)
        self.assertNotIn("o", C.CHARSET)
        self.assertEqual(len(C.CHARSET), 60)

    def test_moved_text_is_not_confident(self):
        for dx, dy in ((2, 0), (-2, 0), (0, 2)):
            self.assertEqual(C.solve_image(cestat_image("A1qPxx", dx, dy))[1], 0.0)

    def test_blank_and_wrong_size(self):
        blank = Image.new("RGB", (C.W, C.H), (72, 240, 240))
        self.assertEqual(C.solve_image(blank), ("", 0.0))
        self.assertEqual(C.solve_image(Image.new("RGB", (120, 35), "white")), ("", 0.0))
        self.assertEqual(C.solve_image(Image.new("RGB", (C.W, C.H), "black")), ("", 0.0))


class TestCestatReal(unittest.TestCase):
    def setUp(self):
        if not (ROOT / "cestat_raw").is_dir():
            self.skipTest("cestat_raw/ not present")
        self.labels = json.loads((ROOT / "cestat_labels.json").read_text())
        self.split = json.loads((ROOT / "cestat_split.json").read_text())

    def test_held_out_exact(self):
        bad = []
        for p in self.split["test"]:
            text, conf = C.solve_image(str(ROOT / p))
            if text != self.labels[p] or conf < 0.35:
                bad.append((p, self.labels[p], text, conf))
        self.assertEqual(bad, [])
        self.assertGreaterEqual(len(self.split["test"]), 30)

    def test_splits_are_disjoint(self):
        self.assertFalse(set(self.split["train"]) & set(self.split["test"]))


if __name__ == "__main__":
    unittest.main()
