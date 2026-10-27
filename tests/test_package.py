"""The public captcha_solver package: identification and the retry contract."""
import io
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

import captcha_solver as cs
from captcha_solver import core
from tests.test_hc import karhc_image, madhc_image, png


def blank(kind):
    w, h = cs.KINDS[kind]["size"]
    return png(np.full((h, w, 3), 255))


class TestIdentifyAndSolve(unittest.TestCase):
    def test_identifies_by_image(self):
        self.assertEqual(cs.identify(madhc_image("385320")), "madhc")
        self.assertEqual(cs.identify(karhc_image("548382")), "karhc")

    def test_unknown_size_and_non_images_are_errors_not_guesses(self):
        with self.assertRaises(cs.UnsupportedCaptcha):
            cs.identify(png(np.zeros((33, 77, 3))))
        with self.assertRaises(cs.UnsupportedCaptcha):
            cs.solve(b"<html>not an image</html>")

    def test_solve_accepts_inputs_of_every_shape(self):
        img = madhc_image("385320")
        for form in (img, io.BytesIO(img), Image.open(io.BytesIO(img))):
            r = cs.solve(form)
            self.assertEqual((r.text, r.kind, r.accepted, r.reason),
                             ("385320", "madhc", True, None))

    def test_blank_is_not_accepted(self):
        r = cs.solve(blank("karhc"))
        self.assertFalse(r.accepted)
        self.assertEqual(r.reason, "no-text")

    def test_kinds_table_matches_the_engine(self):
        for kind, info in cs.KINDS.items():
            route = core._api._SIZES[info["size"]]
            self.assertTrue(route == kind or callable(route), kind)
            self.assertIn(kind, core._api.GATES)


class TestRetry(unittest.TestCase):
    def feed(self, images):
        served = []

        def fetch():
            served.append(images[len(served)])
            return served[-1]
        return fetch, served

    def test_unreadable_captchas_are_refetched_never_submitted(self):
        fetch, served = self.feed([blank("madhc"), blank("madhc"), madhc_image("111222")])
        submitted = []
        r = cs.solve_with_retry(fetch, submit=lambda t: submitted.append(t) or "page")
        self.assertEqual((r.text, r.attempts, r.response), ("111222", 3, "page"))
        self.assertEqual(submitted, ["111222"])          # the blanks never went out

    def test_site_rejection_triggers_a_fresh_captcha(self):
        fetch, served = self.feed([madhc_image("123456"), madhc_image("654321")])
        verdicts = iter([False, "results"])
        r = cs.solve_with_retry(fetch, submit=lambda t: next(verdicts))
        self.assertEqual((r.text, r.attempts, r.response), ("654321", 2, "results"))

    def test_without_submit_the_first_accepted_read_returns(self):
        fetch, served = self.feed([blank("karhc"), karhc_image("548382")])
        r = cs.solve_with_retry(fetch)
        self.assertEqual((r.text, r.attempts), ("548382", 2))

    def test_exhaustion_raises_with_every_attempt(self):
        fetch, _ = self.feed([madhc_image("123456")] * 3)
        with self.assertRaises(cs.CaptchaUnsolved) as ctx:
            cs.solve_with_retry(fetch, submit=lambda t: False, max_attempts=3)
        self.assertEqual([a.reason for a in ctx.exception.attempts],
                         ["rejected-by-site"] * 3)
        fetch, _ = self.feed([blank("madhc")] * 2)
        last = cs.solve_with_retry(fetch, max_attempts=2, raise_on_failure=False)
        self.assertEqual((last.accepted, last.attempts), (False, 2))

    def test_mca_ambiguous_glyph_is_retried(self):
        reads = [("abIcd1", 0.99, "mca"), ("abcde1", 0.99, "mca")]
        submitted = []
        with patch.object(core._api, "solve_bytes", side_effect=reads):
            r = cs.solve_with_retry(lambda: b"img", kind="mca",
                                    submit=lambda t: submitted.append(t) or True)
        self.assertEqual((r.text, r.attempts), ("abcde1", 2))
        self.assertEqual(submitted, ["abcde1"])

    def test_bharatkosh_rerenders_before_drawing_a_new_answer(self):
        calls = []

        def fetch(new):
            calls.append(new)
            return b"render"
        with patch.object(core._api, "solve_bytes", return_value=("aB3dE7", 0.5, "bharatkosh")), \
                patch.object(core._api, "solve_bharatkosh_group",
                             side_effect=[("aB3dE7", 0.7), ("aB3dE7", 0.95)]):
            r = cs.solve_with_retry(fetch, kind="bharatkosh")
        self.assertEqual(calls, [True, False, False])    # one answer, three looks
        self.assertEqual((r.text, r.accepted, r.attempts), ("aB3dE7", True, 1))


if __name__ == "__main__":
    unittest.main()
