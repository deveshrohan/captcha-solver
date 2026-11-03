"""Tests for the readers ported from other projects on this machine: CERSAI,
APHC, K2 (Khajane-II) and the RERA readers (hrrera, pbrera, hprera).

The synthetic tests compose captchas from each reader's shipped glyphs, so
they run anywhere. The real-image tests score the gitignored `*_raw/`
captures and skip when those are absent (they are never distributed).
"""
import io
import json
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

import captcha_solver as cs
from solver import aphc as AP
from solver import cersai as CE
from solver import k2 as K2
from solver.api import GATES, solve_bytes, warmup
from tests.test_hc import gujhc_image, png

ROOT = Path(__file__).resolve().parents[1]


def labelled(name):
    p = ROOT / ("%s_labels.json" % name)
    if not p.exists() or not (ROOT / ("%s_raw" % name)).is_dir():
        raise unittest.SkipTest("%s captures not present" % name)
    rows = {ROOT / k: v for k, v in json.loads(p.read_text()).items()}
    return {k: v for k, v in rows.items() if k.exists()}


# ---------------------------------------------------------------- composers
def k2_image(word, alpha=True):
    """SimpleCaptcha as K2 renders it: black ink on transparent, pen from
    x = 10 advancing by each glyph's ink width, overlaps multiplying."""
    _, tp = K2.templates()
    T = np.ones((K2.H, K2.W), np.float32)
    x = K2.X0
    for c in word:
        left, g, adv = tp[c]
        x0 = x + left
        a, b = max(0, x0), min(K2.W, x0 + g.shape[1])
        T[:, a:b] *= 1 - g[:, a - x0:b - x0]
        x += adv
    rgba = np.zeros((K2.H, K2.W, 4), np.uint8)
    rgba[:, :, 3] = np.rint((1 - T) * 255)
    im = Image.fromarray(rgba)
    out = io.BytesIO()
    (im if alpha else im.convert("RGB")).save(out, format="PNG")
    return out.getvalue()


def cersai_image(word, gap=3, x=8):
    """Native-height CERSAI templates side by side, green ink on white."""
    by_char = {}
    for w, (chars, stack) in CE.templates().items():
        for c, g in zip(chars, stack):
            by_char.setdefault(c, g)
    ink = np.zeros((CE.H, CE.W), bool)
    for c in word:
        g = by_char[c] > 0
        ink[:, x:x + g.shape[1]] |= g
        x += g.shape[1] + gap
    rgb = np.full((CE.H, CE.W, 3), 255, np.uint8)
    rgb[ink] = (10, 100, 10)
    return png(rgb)


def aphc_image(expr, gap=3, x=2, top=9):
    """White glyph bitmaps on the portal's dark grey."""
    a = np.full((AP.H, AP.W), 60, np.uint8)
    for c in expr:
        t = AP.templates()[c].astype(bool)
        cols = np.flatnonzero(t.any(0))
        t = t[:, cols[0]:cols[-1] + 1]
        a[top:top + t.shape[0], x:x + t.shape[1]][t] = 255
        x += t.shape[1] + gap
    return png(np.repeat(a[:, :, None], 3, 2))


# ---------------------------------------------------------------- synthetic
class TestK2(unittest.TestCase):
    def test_composition_reads_back_and_routes(self):
        for word in ("7kmxf", "2mpdm", "bhkfy", "a5w8r"):
            img = k2_image(word)
            self.assertEqual(cs.identify(img), "k2")
            r = cs.solve(img)
            self.assertEqual((r.text, r.kind, r.accepted), (word, "k2", True), word)

    def test_shares_200x50_with_gujarat_hc(self):
        self.assertEqual(cs.identify(gujhc_image("ab3de", [(0, 0, 255)] * 5)), "gujhc")
        # alpha stripped: all black, still K2 (and unreadable), never gujhc
        flat = k2_image("7kmxf", alpha=False)
        self.assertEqual(cs.identify(flat), "k2")
        self.assertFalse(cs.solve(flat).accepted)

    def test_blank_is_not_confident(self):
        rgba = np.zeros((K2.H, K2.W, 4), np.uint8)
        self.assertEqual(K2.solve_image(Image.fromarray(rgba)), ("", 0.0))


class TestCersai(unittest.TestCase):
    def test_composition_reads_back(self):
        for word in ("RokKAW", "96TW8o", "Hq4Zmx"):
            r = cs.solve(cersai_image(word))
            self.assertEqual((r.text, r.kind, r.accepted), (word, "cersai", True), word)

    def test_blank_and_wrong_size(self):
        self.assertEqual(CE.solve_image(png(np.full((45, 200, 3), 255))), ("", 0.0))
        self.assertEqual(CE.solve_image(png(np.zeros((40, 200, 3)))), ("", 0.0))


class TestAphc(unittest.TestCase):
    def test_answer_is_the_sum(self):
        for expr, ans in (("77plus8", "85"), ("100plus9", "109"), ("4plus2", "6")):
            img = aphc_image(expr)
            self.assertEqual(AP.read(img)[:2], (expr, int(ans)))
            r = cs.solve(img)
            self.assertEqual((r.text, r.kind, r.accepted), (ans, "aphc", True), expr)

    def test_no_plus_is_unreadable(self):
        self.assertEqual(AP.solve_image(aphc_image("1234567")), ("", 0.0))
        self.assertFalse(cs.solve(png(np.full((33, 110, 3), 60))).accepted)


class TestRegistry(unittest.TestCase):
    def test_new_kinds_are_wired(self):
        for kind in ("k2", "cersai", "aphc", "hrrera", "pbrera", "hprera"):
            self.assertIn(kind, cs.KINDS)
            self.assertIn(kind, GATES)

    def test_tesseract_kinds_fail_loudly_without_it(self):
        img = png(np.full((38, 100, 3), 255))
        with patch("shutil.which", return_value=None):
            with self.assertRaises(cs.MissingDependency):
                cs.solve(img)
            with self.assertRaises(cs.MissingDependency):
                cs.solve_with_retry(lambda: img)

    def test_warmup_new_kinds(self):
        cs.warmup(["k2", "cersai", "aphc", "hrrera", "pbrera"])


# ---------------------------------------------------------------- real captures
class TestRealCaptures(unittest.TestCase):
    def score(self, rows, kind):
        bad = []
        for p, label in rows.items():
            r = cs.solve(p.read_bytes())
            if r.kind != kind or not r.accepted or r.text != label:
                bad.append((p.name, label, r.text, round(r.confidence, 3)))
        return bad

    def test_cersai_held_out(self):
        rows = labelled("cersai")
        test = json.loads((ROOT / "cersai_split.json").read_text())["test"]
        rows = {p: v for p, v in rows.items() if str(p.relative_to(ROOT)) in test}
        self.assertEqual(len(rows), 40)
        bad = self.score(rows, "cersai")
        # 40/40 read exactly, one falls under the gate
        self.assertLessEqual(len(bad), 1, bad)
        self.assertTrue(all(t == lab for _, lab, t, _ in bad), bad)

    def test_aphc(self):
        rows = labelled("aphc")
        expect = {p: str(AP.read(p)[1]) for p in rows}
        for p, expr in rows.items():
            a, b = expr.split("plus")
            self.assertEqual(expect[p], str(int(a) + int(b)))
        bad = [x for x in self.score({p: expect[p] for p in rows}, "aphc")]
        self.assertEqual(bad, [])

    def test_k2(self):
        rows = {p: v for p, v in labelled("k2").items() if Image.open(p).size == (200, 50)}
        self.assertEqual(self.score(rows, "k2"), [])

    def test_hrrera_gold(self):
        self.assertEqual(self.score(labelled("hrrera"), "hrrera"), [])

    def test_pbrera_gold(self):
        if shutil.which("tesseract") is None:
            self.skipTest("tesseract not installed")
        rows = labelled("pbrera")
        wrong = [p for p, v in rows.items() if cs.solve(p.read_bytes()).text != v]
        self.assertEqual(wrong, [])


if __name__ == "__main__":
    unittest.main()
