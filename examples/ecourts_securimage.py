"""Template: solve a Securimage captcha inside a requests-based flow.

Shows the correct SESSION-BOUND pattern: the captcha code lives in the server
session, so you must fetch the captcha, solve it, and submit the form THROUGH THE
SAME session, and (on a low-confidence read) re-fetch a fresh captcha rather than
reusing the old one.

Run once at process startup:  from solver.api import warmup; warmup()
Then call solve_ecourts_form(session, ...) per request.

Only needs: torch, numpy, pillow  (no scipy, no GPU, no network to Anthropic).
"""
import time

import os

import requests

from solver.api import solve_with_retry, warmup

# Set to your own Securimage endpoint. Only automate a service you are authorized
# to use (see README acceptable use).
CAPTCHA_URL = os.environ.get("SECURIMAGE_URL", "https://YOUR-SECURIMAGE-ENDPOINT/securimage_show.php")


def make_session():
    """A session whose cookies bind the captcha code to the form submission."""
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0"})
    return s


def solve_ecourts_captcha(session, min_conf=0.90, max_tries=4):
    """Fetch + solve the captcha for `session`. Returns (code, confidence, tries).

    Each call to `fetch` re-requests securimage_show.php on the SAME session,
    which rotates the server-side code — so the returned code always matches the
    captcha the session currently expects. Submit it immediately (below)."""
    def fetch():
        # cache-buster keeps proxies/browsers from returning a stale image;
        # the server still rotates the session code on every hit.
        r = session.get(CAPTCHA_URL, params={"_": int(time.time() * 1000)}, timeout=15)
        r.raise_for_status()
        return r.content

    code, conf, _kind, tries = solve_with_retry(
        fetch, min_conf=min_conf, max_tries=max_tries, kind="securimage"
    )
    return code, conf, tries


def submit_search(session, form_fields, captcha_code):
    """Replace with your real search POST. The captcha_code MUST be the one
    returned by solve_ecourts_captcha for THIS session, submitted without
    fetching another captcha in between."""
    payload = dict(form_fields)
    payload["captcha"] = captcha_code                 # <-- your form's captcha field name
    # resp = session.post(SEARCH_URL, data=payload, timeout=30)
    # return resp
    raise NotImplementedError("wire this to your existing search POST")


def run_one(form_fields):
    session = make_session()
    # (do whatever GETs your flow needs to establish the session first)
    code, conf, tries = solve_ecourts_captcha(session)
    if conf < 0.90:
        # even after retries it stayed low-confidence: still submit (the code
        # matches the session) but flag it — a wrong submit just means re-running.
        print(f"[warn] low-confidence captcha {code!r} conf={conf:.2f} after {tries} tries")
    return submit_search(session, form_fields, code)


if __name__ == "__main__":
    warmup()                                          # load model once, up front
    # run_one({...your search fields...})
    print("solver ready; wire run_one() to your scraper's search flow")
