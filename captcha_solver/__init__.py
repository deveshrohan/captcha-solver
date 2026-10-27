"""captcha_solver: offline readers for Indian government-portal captchas.

    import captcha_solver as cs

    cs.identify(image_bytes)            # -> "mca"
    r = cs.solve(image_bytes)           # -> Result(text, confidence, kind, accepted, ...)

    # let the package retry until the site accepts an answer
    r = cs.solve_with_retry(fetch=lambda: session.get(URL).content,
                            submit=lambda text: post_form(text))   # truthy = accepted

Everything runs locally on CPU; no image ever leaves the process.
"""
from .core import (KINDS, CaptchaError, CaptchaUnsolved, Result,
                   UnsupportedCaptcha, identify, solve, solve_url,
                   solve_with_retry, warmup)

__version__ = "0.2.0"
__all__ = ["KINDS", "CaptchaError", "CaptchaUnsolved", "Result",
           "UnsupportedCaptcha", "identify", "solve", "solve_url",
           "solve_with_retry", "warmup", "__version__"]
