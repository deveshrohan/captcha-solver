"""Goa and Uttar Pradesh RERA readers (solver/ga_rera.py, solver/up_rera.py).

Synthetic tests render captchas with the training renderer (train_rera.py).
It needs the macOS fonts the CNNs were pretrained on, so those tests skip
elsewhere. Routing, blank and size tests run anywhere. Real-image tests skip
when the gitignored *_raw/ dirs are absent.
"""
import io
import json
import os
import random
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from solver import ga_rera as G
from solver import up_rera as U

ROOT = Path(__file__).resolve().parents[1]
SUP = "/System/Library/Fonts/Supplemental/"


def need_fonts(*names):
    if not all(os.path.exists(SUP + n) for n in names):
        raise unittest.SkipTest("training fonts not installed")


def up_image(word, rng):
    """Five synthetic MSCaptcha glyphs, one per 30 px slot, green on white."""
    import train_rera as T
    rgb = np.full((U.H, U.W, 3), 255, np.uint8)
    for i, ch in enumerate(word):
        while True:
            c = T.up_glyph(rng, ch) > 0.5
            ys, xs = np.nonzero(c)
            if len(xs):
                break
        g = c[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        s = min(1.0, 24 / max(g.shape))
        g = np.asarray(Image.fromarray(np.uint8(g) * 255).resize(
            (max(1, int(g.shape[1] * s)), max(1, int(g.shape[0] * s)))), np.uint8) > 127
        y0, x0 = (U.H - g.shape[0]) // 2, i * U.SLOT + (U.SLOT - g.shape[1]) // 2
        rgb[y0:y0 + g.shape[0], x0:x0 + g.shape[1]][g] = (90, 160, 30)
    return Image.fromarray(rgb)


def real(name):
    lab, sp = ROOT / ("%s_labels.json" % name), ROOT / ("%s_split.json" % name)
    if not (ROOT / ("%s_raw" % name)).is_dir() or not lab.exists():
        raise unittest.SkipTest("%s_raw/ not present" % name)
    L = json.loads(lab.read_text())
    return {p: L[p] for p in json.loads(sp.read_text())["test"] if (ROOT / p).exists()}


class TestGaRera(unittest.TestCase):
    def test_routes_cersai_and_goa_apart(self):
        grey = Image.new("RGB", (200, 45), (208, 208, 208))
        self.assertEqual(G.route_200x45(grey), "garera")
        green = np.full((45, 200, 3), 255, np.uint8)
        green[10:30, 20:180] = (10, 100, 10)
        self.assertEqual(G.route_200x45(Image.fromarray(green)), "cersai")

    def test_blank_and_wrong_size(self):
        self.assertEqual(G.solve_image(Image.new("RGB", (200, 45), (208, 208, 208))), ("", 0.0))
        self.assertEqual(G.solve_image(Image.new("RGB", (200, 50), "white")), ("", 0.0))

    def test_reads_its_renderer(self):
        need_fonts("Arial Italic.ttf")
        import train_rera as T
        rng = random.Random(7)
        ok = 0
        for _ in range(20):
            im, text = T.ga_render(rng)
            ok += G.solve_image(im)[0] == text.replace("I", "l")
        self.assertGreaterEqual(ok, 17)

    def test_real_held_out(self):
        rows = real("ga_rera")
        self.assertGreaterEqual(len(rows), 30)
        ok = sum(G.solve_image(str(ROOT / p))[0] == v for p, v in rows.items())
        self.assertGreaterEqual(ok / len(rows), 0.9)


class TestUpRera(unittest.TestCase):
    def test_blank_and_wrong_size(self):
        self.assertEqual(U.solve_image(Image.new("RGB", (150, 35), "white")), ("", 0.0))
        self.assertEqual(U.solve_image(Image.new("RGB", (150, 38), "white")), ("", 0.0))

    def test_noise_dot_holes_are_filled(self):
        a = np.full((35, 150, 3), 255, np.uint8)
        a[10:25, 10:20] = (90, 160, 30)
        a[17, 15] = (0, 0, 0)                     # a black dot inside the glyph
        self.assertTrue(U.ink(Image.fromarray(a))[17, 15])

    def test_one_glyph_per_slot(self):
        need_fonts("Arial Bold.ttf")
        im = up_image("EZQ4E", random.Random(3))
        self.assertEqual(len(U.segment(im)), 5)

    def test_reads_synthetic(self):
        need_fonts("Arial Bold.ttf")
        rng = random.Random(11)
        ok = 0
        for _ in range(20):
            word = "".join(rng.choice(U.CHARSET) for _ in range(5))
            ok += U.solve_image(up_image(word, rng))[0] == word
        self.assertGreaterEqual(ok, 12)

    def test_real_held_out(self):
        rows = real("up_rera")
        if len(rows) < 30:
            self.skipTest("fewer than 30 labelled test captures")
        ok = sum(U.solve_image(str(ROOT / p))[0] == v for p, v in rows.items())
        self.assertGreaterEqual(ok / len(rows), 0.7)


if __name__ == "__main__":
    unittest.main()
