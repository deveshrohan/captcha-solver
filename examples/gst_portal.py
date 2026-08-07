"""Example: solve the GST portal captcha in-process, inside a scraper session.

Like the eCourts securimage captcha, the GST captcha is **session-bound** — and
here the mechanism is visible: every response sets a fresh
`CaptchaCookie=<32 hex>` (Domain=.gst.gov.in, Secure, HttpOnly), and the value
rotates on *every* GET. The server keeps the answer for the most recent cookie,
so only the last image fetched is valid to submit. Always fetch through the same
`requests.Session` the form is posted with, and post before fetching again.

Note the `rnd` parameter is only cache-busting: the same `rnd` returns different
images on successive calls, and omitting it works fine. It does not seed the
captcha, so there is nothing to be gained by choosing particular values.

    python3 examples/gst_portal.py
"""
import time

import requests

from solver.api import solve_with_retry, warmup

CAPTCHA_URL = "https://services.gst.gov.in/services/captcha"


def make_session():
    s = requests.Session()
    # The endpoint sits behind an F5 WAF that rejects a *spoofed* browser
    # User-Agent lacking the matching client hints. Either send a complete,
    # consistent browser header set or leave the default client UA alone.
    s.headers.update({"Accept": "image/avif,image/webp,image/*,*/*;q=0.8"})
    return s


def run_once(session):
    def fetch():
        # rnd is cache-busting only; the session binding comes from CaptchaCookie,
        # which this Session carries and the server rotates on every call.
        r = session.get(CAPTCHA_URL, params={"rnd": time.time()}, timeout=20)
        r.raise_for_status()
        return r.content

    # solve_with_retry returns the LAST read, which is the code the session holds
    code, conf, kind, tries = solve_with_retry(fetch, min_conf=0.95, kind="gst")
    print(f"captcha={code!r} conf={conf:.3f} kind={kind} tries={tries}")

    if conf < 0.80:
        print("low confidence - the search will just fail; re-run for a fresh captcha")
    # ...now POST your search form with `code` on this same `session`.
    return code


def main():
    warmup(securimage=False, gst=True)     # load the GST model once at startup
    session = make_session()
    run_once(session)


if __name__ == "__main__":
    main()
