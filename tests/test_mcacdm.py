"""MCA CDM export captcha reader (solver/mcacdm.py).

Synthetic tests compose captchas from the shipped glyph table, the way GD's
imagestring() draws them, so they run anywhere. The held-out test scores the
gitignored mcacdm_raw/ captures and skips when they are absent.
"""
import io
import json
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from solver import mcacdm as C

ROOT = Path(__file__).resolve().parents[1]
INK, BG = (204, 0, 0), (224, 224, 224)


def compose(word, origin=C.ORIGIN):
    chars, stack = C.glyphs()
    table = dict(zip(chars, stack))
    rgb = np.empty((C.H, C.W, 3), np.uint8)
    rgb[:] = BG
    x0, y0 = origin
    for k, c in enumerate(word):
        g = table[c].astype(bool)
        rgb[y0:y0 + C.CELL_H, x0 + C.CELL_W * k:x0 + C.CELL_W * (k + 1)][g] = INK
    out = io.BytesIO()
    Image.fromarray(rgb).save(out, format="PNG")
    return out.getvalue()


class TestMcaCdm(unittest.TestCase):
    def test_charset_is_full_alnum(self):
        chars, stack = C.glyphs()
        self.assertEqual(len(chars), 62)
        self.assertEqual(stack.shape[1:], (C.CELL_H, C.CELL_W))

    def test_digits_and_lowercase_are_libgd_font5(self):
        ref = json.loads((ROOT / "tests" / "gdfontgiant_alnum.json").read_text())["glyphs"]
        ours = json.loads((ROOT / "solver" / "mcacdm_glyphs.json").read_text())["glyphs"]
        for c, rows in ref.items():
            self.assertEqual(ours[c], rows, c)

    def test_no_two_glyphs_collide(self):
        _, stack = C.glyphs()
        s = stack.reshape(len(stack), -1).astype(int)
        d = np.abs(s[:, None] - s[None]).sum(-1) + np.eye(len(s), dtype=int) * 999
        self.assertGreaterEqual(d.min(), 5)          # I/l, the closest pair

    def test_composition_reads_back_exactly(self):
        for word in ("4BXM1N", "Il1i0O", "QbEztM", "zzzzzz"):
            self.assertEqual(C.solve_image(compose(word)), (word, 1.0), word)

    def test_moved_pen_is_found(self):
        self.assertEqual(C.solve_image(compose("aB3dE9", (31, 11))), ("aB3dE9", 1.0))

    def test_stray_ink_lowers_confidence(self):
        im = Image.open(io.BytesIO(compose("aB3dE9"))).convert("RGB")
        a = np.asarray(im).copy()
        a[30:34, 90:110] = INK
        text, conf = C.solve_image(Image.fromarray(a))
        self.assertEqual(text, "aB3dE9")
        self.assertLess(conf, 0.9)

    def test_blank_and_wrong_size(self):
        blank = Image.fromarray(np.full((C.H, C.W, 3), 224, np.uint8))
        self.assertEqual(C.solve_image(blank), ("", 0.0))
        self.assertEqual(C.solve_image(Image.new("RGB", (120, 40), BG)), ("", 0.0))

    def test_held_out_real_captures(self):
        if not (ROOT / "mcacdm_raw").is_dir():
            self.skipTest("mcacdm_raw/ not present")
        labels = json.loads((ROOT / "mcacdm_labels.json").read_text())
        test = json.loads((ROOT / "mcacdm_split.json").read_text())["test"]
        test = [p for p in test if (ROOT / p).exists()]
        self.assertGreaterEqual(len(test), 30)
        for p in test:
            self.assertEqual(C.solve_image(str(ROOT / p)), (labels[p], 1.0), p)


if __name__ == "__main__":
    unittest.main()
