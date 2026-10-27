"""Offline tests for the High Court readers (gujhc, madhc, calhc).

No network and no captcha images: synthetic compositions are built from the
shipped templates / the training renderer, so these run anywhere.
"""
import io
import json
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from solver import calhc as CH
from solver import gujhc as GH
from solver import karhc as KH
from solver import madhc as MH
from solver.api import GATES, solve_bytes, warmup

ROOT = Path(__file__).resolve().parents[1]


def png(arr):
    out = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8)).save(out, format="PNG")
    return out.getvalue()


def madhc_image(word, x=12, steps=None):
    """Compose a Madras captcha from the shipped templates: 80%-opaque
    #323232 digits, overlaps multiplying, white background."""
    tp, st = MH.templates(), MH.steps()
    L = np.zeros((MH.H, MH.W), np.float32)
    for i, d in enumerate(word):
        MH._add(L, tp[d], x)
        x += (steps[i] if steps else st[d][1])
    I = MH.INK + (255 - MH.INK) * np.exp(-L)
    return png(np.repeat(np.rint(I)[:, :, None], 3, 2))


def gujhc_image(word, colours, line=(20, 30)):
    """Compose a Gujarat captcha: glyphs over #dbf7ff in order, each pure
    colour composited through its template's alpha, then the noise line and
    the border."""
    tp = GH.templates()
    rgb = np.empty((GH.H, GH.W, 3), np.float32)
    rgb[:] = (219, 247, 255)
    x = GH.X0
    for c, col in zip(word, colours):
        top, left, g, adv = tp[c]
        a = 1 - np.exp(-g)
        h, w = g.shape
        win = rgb[top:top + h, x + left:x + left + w]
        win[:] = win * (1 - a[:, :, None]) + np.array(col, np.float32) * a[:, :, None]
        x += adv
    y1, y2 = line
    for cx in range(GH.W):
        yc = y1 + (y2 - y1) * cx / GH.W
        rgb[int(round(yc - 2)):int(round(yc + 2)), cx] = 0
    rgb[0, :] = rgb[-1, :] = rgb[:, 0] = rgb[:, -1] = 0
    return png(np.rint(rgb))


class TestMadras(unittest.TestCase):
    def test_composition_reads_back_and_routes(self):
        for word, steps in [("385320", None), ("111111", None),
                            ("080215", [13, 15, 17, 13, 15, 17])]:
            text, conf, kind = solve_bytes(madhc_image(word, steps=steps))
            self.assertEqual((text, kind), (word, "madhc"))
            self.assertGreaterEqual(conf, GATES["madhc"])

    def test_blank_and_wrong_size(self):
        blank = png(np.full((MH.H, MH.W, 3), 255))
        self.assertEqual(solve_bytes(blank), ("", 0.0, "madhc"))
        with self.assertRaises(ValueError):
            MH.solve_image(png(np.full((50, 120, 3), 255)))

    def test_foreign_glyphs_are_not_confident(self):
        # A different font (the generator changed) must fall below the gate.
        from PIL import ImageDraw, ImageFont
        im = Image.new("RGB", (MH.W, MH.H), "white")
        ImageDraw.Draw(im).text((10, 41), "385320", fill=(91, 91, 91), anchor="ls",
                                font=ImageFont.truetype(str(ROOT / "fonts/DejaVuSans.ttf"), 30))
        _, conf = MH.solve_image(im)
        self.assertLess(conf, GATES["madhc"])


class TestGujarat(unittest.TestCase):
    RGB = [(0, 0, 0), (0, 0, 255), (255, 0, 0)]

    def test_composition_reads_back_and_routes(self):
        for word, cols, line in [("p5eov", [1, 2, 0, 1, 2], (25, 47)),
                                 ("0o1li", [0, 0, 0, 0, 0], (5, 45)),
                                 ("wmwmw", [2, 1, 2, 1, 2], (30, 10))]:
            img = gujhc_image(word, [self.RGB[c] for c in cols], line)
            text, conf, kind = solve_bytes(img)
            self.assertEqual((text, kind), (word, "gujhc"))
            self.assertGreaterEqual(conf, GATES["gujhc"])

    def test_blank_and_wrong_size(self):
        blank = np.empty((GH.H, GH.W, 3))
        blank[:] = (219, 247, 255)
        self.assertEqual(solve_bytes(png(blank)), ("", 0.0, "gujhc"))
        with self.assertRaises(ValueError):
            GH.solve_image(png(np.full((40, 200, 3), 255)))

    def test_line_mask_covers_the_line_not_the_text(self):
        img = Image.open(io.BytesIO(gujhc_image("abcde", [self.RGB[1]] * 5, (10, 40))))
        m = GH.line_mask(np.asarray(img.convert("RGB")))
        self.assertTrue(m[10, 2] and m[40, GH.W - 3])
        self.assertFalse(m[40, 60])


class TestCalcutta(unittest.TestCase):
    def test_renderer_shape_and_charset(self):
        rng = np.random.default_rng(0)
        self.assertEqual(CH.make_rgb("abc234", rng).shape, (CH.H, CH.W, 3))
        self.assertEqual(CH.load_real(png(np.full((CH.H, CH.W, 3), CH.BG))).shape,
                         (CH.IN_H, CH.IN_W))
        # mews' default list minus 0 1 5 i k l o s v w, lowercased
        self.assertEqual(CH.CHARSET, "2346789abcdefghjmnpqrtuxyz")

    def test_model_reads_its_own_renderer(self):
        rng = np.random.default_rng(1234)
        words = [CH.random_label(rng) for _ in range(20)]
        hits = 0
        for w in words:
            out = io.BytesIO()
            Image.fromarray(CH.make_rgb(w, rng)).save(out, format="JPEG", quality=95)
            text, _, kind = solve_bytes(out.getvalue())
            self.assertEqual(kind, "calhc")
            hits += text == w
        self.assertGreaterEqual(hits, 17)


def karhc_image(word, origin=KH.ORIGIN, quality=75):
    """GD font-5 digits in black at the grid, faint lavender noise, JPEG."""
    rgb = np.full((KH.H, KH.W, 3), 255, np.float32)
    rgb[5, :] = rgb[14, 10:90] = (200, 200, 235)              # noise lines
    for x in (3, 50, 95):
        rgb[16, x - 1:x + 2] = rgb[15:18, x] = (190, 190, 230)  # '+' specks
    x0, y0 = origin
    for k, d in enumerate(word):
        g = KH.glyphs()[d]
        x = x0 + KH.CELL_W * k
        ys, xs = np.nonzero(g)
        for y, xx in zip(ys + y0, xs + x):
            if 0 <= y < KH.H and 0 <= xx < KH.W:
                rgb[y, xx] = 0
    out = io.BytesIO()
    Image.fromarray(rgb.astype(np.uint8)).save(out, format="JPEG", quality=quality)
    return out.getvalue()


class TestKarnataka(unittest.TestCase):
    def test_composition_reads_back_and_routes(self):
        for word in ("548382", "515054", "000000", "898989", "123456"):
            text, conf, kind = solve_bytes(karhc_image(word))
            self.assertEqual((text, kind), (word, "karhc"))
            self.assertGreaterEqual(conf, GATES["karhc"])

    def test_moved_text_is_not_confident(self):
        # the grid origin moved by more than the 1 px slack: stray ink shows
        _, conf = KH.solve_image(karhc_image("548382", origin=(10, 3)))
        self.assertLess(conf, GATES["karhc"])

    def test_blank_and_wrong_size(self):
        self.assertEqual(solve_bytes(png(np.full((KH.H, KH.W, 3), 255))),
                         ("", 0.0, "karhc"))
        with self.assertRaises(ValueError):
            KH.solve_image(png(np.full((21, 100, 3), 255)))

    def test_glyphs_are_libgd_font5(self):
        ref = json.loads((ROOT / "tests" / "gdfontgiant_alnum.json").read_text())["glyphs"]
        for d, g in KH.glyphs().items():
            self.assertEqual(["".join(str(int(v)) for v in row) for row in g], ref[d])


class TestItat2(unittest.TestCase):
    """ITAT's format since October 2026 (150x38); the old 150x42 stays `itat`."""

    def test_routes_by_size_and_keeps_the_old_format(self):
        from solver.api import _SIZES
        self.assertEqual(_SIZES[(150, 38)], "itat2")
        self.assertEqual(_SIZES[(150, 42)], "itat")

    def test_charset_matches_the_census(self):
        from solver import itat2 as T2
        self.assertEqual(T2.N_CLASSES, 34)
        self.assertNotIn("O", T2.CHARSET)
        self.assertNotIn("0", T2.CHARSET)
        for c in "1IL":                      # back in, unlike the old format
            self.assertIn(c, T2.CHARSET)

    def test_model_reads_its_own_renderer(self):
        from solver import itat2 as T2
        rng = np.random.default_rng(99)
        words = [T2.random_label(rng) for _ in range(20)]
        hits = 0
        for w in words:
            text, _, kind = solve_bytes(png(np.asarray(T2.make_rgb(w, rng))))
            self.assertEqual(kind, "itat2")
            hits += text == w
        self.assertGreaterEqual(hits, 14)


class TestProvenance(unittest.TestCase):
    def test_splits_are_pinned_and_disjoint(self):
        for kind in ("gujhc", "madhc", "calhc", "karhc", "itat2"):
            split = json.loads((ROOT / (kind + "_split.json")).read_text())
            self.assertGreaterEqual(len(split["test"]), 30)
            held = set(split["test"]) | set(split["validation"])
            self.assertFalse(held & set(split["train"]))
            self.assertFalse(set(split["test"]) & set(split["validation"]))

    def test_templates_come_from_train_only(self):
        for kind in ("gujhc", "madhc"):
            split = json.loads((ROOT / (kind + "_split.json")).read_text())
            meta = json.loads((ROOT / "solver" / (kind + "_glyphs.json")).read_text())["_meta"]
            self.assertEqual(meta["built_from"], "train-split")
            self.assertEqual(meta["train_images"], len(split["train"]))

    def test_warmup_new_kinds(self):
        warmup(securimage=False, gujhc=True, madhc=True, calhc=True, karhc=True, itat2=True)


class TestDownloader(unittest.TestCase):
    def test_truncated_bodies_are_rejected(self):
        import download_hc as D
        good = png(np.zeros((4, 4, 3)))
        self.assertTrue(D.complete("madhc", good))
        self.assertFalse(D.complete("madhc", good[:-12]))
        self.assertFalse(D.complete("gujhc", b""))      # ct missing -> empty 200
        jpg = io.BytesIO()
        Image.new("RGB", (4, 4)).save(jpg, format="JPEG")
        self.assertTrue(D.complete("calhc", jpg.getvalue()))
        self.assertFalse(D.complete("calhc", jpg.getvalue()[:-2]))


if __name__ == "__main__":
    unittest.main()
