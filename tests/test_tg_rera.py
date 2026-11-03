"""Telangana RERA reader (solver/tg_rera.py).

The synthetic test composes captchas from the shipped templates and lattice,
so it runs anywhere. The real test scores the gitignored tg_rera_raw/ captures
and skips when they are absent.
"""
import io
import json
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from solver import tg_rera as T

ROOT = Path(__file__).resolve().parents[1]


def compose(word):
    """Templates at their slots by max, lattice dots in navy, white elsewhere."""
    chars, stack, lat = T.templates()
    d = np.zeros((T.H, T.W), np.float32)
    for i, c in enumerate(word):
        d = np.maximum(d, T._place(stack[chars.index(c)], i))
    rgb = np.repeat((255 * (1 - d))[:, :, None], 3, 2)
    rgb[d > 0.05, 2] = np.maximum(rgb[d > 0.05, 2], 128)       # navy-ish ink
    rgb[lat] = (0, 0, 128)
    out = io.BytesIO()
    Image.fromarray(np.rint(rgb).astype(np.uint8)).save(out, format="PNG")
    return out.getvalue()


class TestTgRera(unittest.TestCase):
    def test_charset_is_the_census(self):
        self.assertEqual("".join(T.templates()[0]), "ABCEFGHJKNPRSTUVXYZ")

    def test_composition_reads_back(self):
        for word in ("KKPTF", "JBRSE", "ZXTUV", "GJZZB"):
            text, conf = T.solve_image(compose(word))
            self.assertEqual(text, word)
            self.assertGreaterEqual(conf, 0.85, word)

    def test_blank_and_wrong_size(self):
        white = Image.new("RGB", (T.W, T.H), "white")
        self.assertEqual(T.solve_image(white), ("", 0.0))
        self.assertEqual(T.solve_image(Image.new("RGB", (150, 48), "white")), ("", 0.0))

    def test_a_missing_letter_is_not_confident(self):
        a = np.asarray(Image.open(io.BytesIO(compose("KKPTF"))).convert("RGB")).copy()
        x = T.SLOT0 + 2 * T.STEP
        a[:, x + 4:x + 24] = 255
        self.assertLess(T.solve_image(Image.fromarray(a))[1], 0.85)

    def test_real_held_out(self):
        labels, split = ROOT / "tg_rera_labels.json", ROOT / "tg_rera_split.json"
        if not (ROOT / "tg_rera_raw").is_dir() or not labels.exists():
            self.skipTest("tg_rera_raw/ not present")
        L = json.loads(labels.read_text())
        test = [p for p in json.loads(split.read_text())["test"] if (ROOT / p).exists()]
        self.assertGreaterEqual(len(test), 30)
        bad = [(p, L[p], T.solve_image(str(ROOT / p))) for p in test]
        bad = [b for b in bad if b[2][0] != b[1] or b[2][1] < 0.85]
        self.assertEqual(bad, [])


if __name__ == "__main__":
    unittest.main()
