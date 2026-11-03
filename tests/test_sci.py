"""Supreme Court of India reader (solver/sci.py).

The synthetic test composes an expression from the shipped exemplars, so it
runs anywhere. The real test scores the held-out split and skips when the
gitignored sci_raw/ captures are absent.
"""
import io
import json
import re
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from solver import sci as S

ROOT = Path(__file__).resolve().parents[1]


def exemplar(cls):
    """First (unrotated) exemplar of a class, back at its native box size."""
    chars, vecs, sizes, _ = S.templates()
    i = chars.index(cls)
    h, w = (int(v) for v in sizes[i])
    s = max(h, w)
    sq = Image.fromarray(np.uint8(vecs[i].reshape(S.NORM, S.NORM) * 255)).resize((s, s), Image.BILINEAR)
    a = np.asarray(sq) > 127
    y0, x0 = (s - h) // 2, (s - w) // 2
    return a[y0:y0 + h, x0:x0 + w]


def compose(tokens, x=60, gap=12):
    img = np.full((S.H, S.W), 242, np.uint8)
    for t in tokens:
        g = exemplar(t)
        y = (S.H - g.shape[0]) // 2
        img[y:y + g.shape[0], x:x + g.shape[1]][g] = 87
        x += g.shape[1] + gap
    out = io.BytesIO()
    Image.fromarray(np.repeat(img[:, :, None], 3, 2)).save(out, format="PNG")
    return out.getvalue()


class TestSynthetic(unittest.TestCase):
    def test_composition_reads_back(self):
        for tokens, expr, ans in ((["9", "+", "4"], "9+4", "13"),
                                  (["6", "-", "2"], "6-2", "4"),
                                  (["1", "0", "-", "1"], "10-1", "9")):
            img = compose(tokens)
            self.assertEqual(S.read(img)[:2], (expr, int(ans)))
            self.assertEqual(S.solve_image(img)[0], ans)

    def test_blank_wrong_size_and_garbage(self):
        self.assertEqual(S.solve_image(compose([])), ("", 0.0))
        self.assertEqual(S.solve_image(Image.new("RGB", (215, 80), (242,) * 3)), ("", 0.0))
        self.assertEqual(S.solve_image(compose(["9", "4"])), ("", 0.0))   # no operator

    def test_group_vote(self):
        img = compose(["6", "-", "2"])
        self.assertEqual(S.solve_group([img, img, compose([])])[0], "4")
        self.assertEqual(S.solve_group([]), ("", 0.0))


class TestReal(unittest.TestCase):
    def test_held_out(self):
        if not (ROOT / "sci_raw").is_dir():
            self.skipTest("sci captures not present")
        labels = json.loads((ROOT / "sci_labels.json").read_text())
        test = [p for p in json.loads((ROOT / "sci_split.json").read_text())["test"]
                if (ROOT / p).exists()]
        self.assertGreaterEqual(len(test), 30)
        wrong = 0
        for rel in test:
            a, op, b = re.fullmatch(r"(\d+)([+\-x])(\d+)", labels[rel]).groups()
            want = str(S.OPS[op](int(a), int(b)))
            text, conf = S.solve_image(ROOT / rel)
            wrong += text != want
            if conf >= 0.6:                  # the gate: an accepted read must be right
                self.assertEqual(text, want, rel)
        self.assertLessEqual(wrong / len(test), 0.10)

    def test_templates_come_from_train_only(self):
        if not (ROOT / "sci_split.json").exists():
            self.skipTest("no split")
        meta = json.loads((ROOT / "solver" / "sci_glyphs.json").read_text())["_meta"]
        self.assertIn("train", meta["built_from"])


if __name__ == "__main__":
    unittest.main()
