"""District courts (securimage-wp) reader: renderer, routing, voting and the
sticky-code retry. The real-image tests skip without dcourts_raw/."""
import io
import json
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

import captcha_solver as cs
from solver import dcourts as D

ROOT = Path(__file__).resolve().parents[1]


def png(gray):
    out = io.BytesIO()
    Image.fromarray(gray).convert("RGB").save(out, format="PNG")
    return out.getvalue()


class TestRenderer(unittest.TestCase):
    def test_shape_and_two_tone(self):
        rng = np.random.default_rng(0)
        g = D.render(D.random_code(rng), rng)
        self.assertEqual(g.shape, (D.H, D.W))
        self.assertEqual(set(np.unique(g)), {D.BG, D.INK})

    def test_charset_has_no_l_I_O(self):
        self.assertFalse(set("lIO") & set(D.DRAW_CHARSET))
        self.assertEqual(len(D.CLASSES), 36)


class TestReader(unittest.TestCase):
    def test_routes_by_size(self):
        rng = np.random.default_rng(1)
        self.assertEqual(cs.identify(png(D.render("WOy2Z", rng))), "dcourts")

    def test_reads_its_own_renders(self):
        rng = np.random.default_rng(7)
        codes = [D.random_code(rng) for _ in range(12)]
        hits = sum(cs.solve(png(D.render(c, rng))).text == c.upper() for c in codes)
        self.assertGreaterEqual(hits, 10)

    def test_votes_over_renders_of_one_code(self):
        rng = np.random.default_rng(11)
        imgs = [png(D.render("xoCRn", rng)) for _ in range(3)]
        text, conf = D.solve_group(imgs)
        self.assertEqual(text, "XOCRN")
        self.assertGreater(conf, 0.5)

    def test_wrong_size_is_unreadable(self):
        self.assertEqual(D.solve_image(png(np.full((80, 225), 242, np.uint8))), ("", 0.0))


class TestStickyRetry(unittest.TestCase):
    def test_runner_up_is_submitted_before_refetching(self):
        """A rejected district-court guess keeps the code, so the next
        reading of the same renders goes out without a new fetch."""
        from unittest.mock import patch
        from captcha_solver import core
        fetched, submitted = [], []

        def fetch(new):
            fetched.append(new)
            return b"img"
        with patch.object(core._api, "solve_bytes", return_value=("9ABJZ", 0.9, "dcourts")), \
                patch.dict(core._api.STICKY, {"dcourts": lambda ims: [("9ABJZ", .69), ("GABJZ", .31)]}):
            r = cs.solve_with_retry(fetch, kind="dcourts",
                                    submit=lambda t: submitted.append(t) or (t == "GABJZ" and "ok"))
        self.assertEqual(submitted, ["9ABJZ", "GABJZ"])
        self.assertEqual(fetched, [True])
        self.assertEqual((r.text, r.response, r.attempts), ("GABJZ", "ok", 1))


class TestRealCodes(unittest.TestCase):
    def test_held_out_codes(self):
        if not (ROOT / "dcourts_raw").is_dir():
            self.skipTest("dcourts captures not present")
        import sys
        sys.path.insert(0, str(ROOT))
        from download_dcourts import load_manifest
        man = load_manifest(str(ROOT / "dcourts_raw"))
        labels = json.loads((ROOT / "dcourts_labels.json").read_text())
        test = json.loads((ROOT / "dcourts_split.json").read_text())["test"]
        voted = within2 = 0
        for code in test:
            paths = [str(ROOT / "dcourts_raw" / n) for n in man[code]["renders"][:3]]
            ranked = [w for w, _ in D.rank_group(paths)]
            voted += ranked[0] == labels[code]
            within2 += labels[code] in ranked[:2]
        self.assertGreaterEqual(voted, 39)
        self.assertEqual(within2, len(test))


if __name__ == "__main__":
    unittest.main()
