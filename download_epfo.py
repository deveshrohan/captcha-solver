#!/usr/bin/env python3
"""Download EPFO employer-portal captchas (unifiedportal-emp.epfindia.gov.in).

The captcha URL carries an `_HDIV_STATE_` token from HDIV (a Java request-
integrity framework). Measured behaviour:

  * the token is **session-bound** — replaying a copied URL without the matching
    JSESSIONID 302s to /publicPortal/error.jsp and returns nothing;
  * but it is **reusable** within its session: three fetches on one token
    returned three different captchas;
  * so the flow is: GET the establishment-search page with a cookie jar, scrape
    the freshly-minted captcha URL out of the HTML, then fetch repeatedly on
    that same session.

    python3 download_epfo.py 400 epfo_raw
"""
import os
import re
import subprocess
import sys
import time

BASE = "https://unifiedportal-emp.epfindia.gov.in"
ENTRY = BASE + "/publicPortal/no-auth/misReport/home/loadEstSearchHome"
CAPTCHA_RE = re.compile(r"/publicPortal/no-auth/captcha/createCaptcha\?_HDIV_STATE_=[0-9A-F-]+")
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def new_session(jar):
    """Load the entry page to mint a JSESSIONID + HDIV token; return captcha URL."""
    if os.path.exists(jar):
        os.remove(jar)
    html = subprocess.run(
        ["curl", "-s", "-L", "--max-time", "30", "-c", jar, "-b", jar, ENTRY],
        capture_output=True).stdout.decode("utf-8", "replace")
    m = CAPTCHA_RE.search(html)
    return BASE + m.group(0) if m else None


def fetch(url, jar):
    out = subprocess.run(
        ["curl", "-s", "--max-time", "25", "-b", jar, "-c", jar, url],
        capture_output=True).stdout
    return out if out.startswith(PNG_MAGIC) else None


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    outdir = sys.argv[2] if len(sys.argv) > 2 else "epfo_raw"
    delay = float(os.environ.get("EPFO_DELAY", "0.4"))
    os.makedirs(outdir, exist_ok=True)
    jar = os.path.join(outdir, ".cookies")

    url = new_session(jar)
    if not url:
        sys.exit("could not mint an HDIV token from the entry page")
    print("session ok, token url: %s" % url.split("=")[-1], flush=True)

    got = miss = 0
    for i in range(n):
        png = fetch(url, jar)
        if png is None:                       # session or token expired -> renew
            miss += 1
            url = new_session(jar) or url
            png = fetch(url, jar)
        if png:
            with open(os.path.join(outdir, "e%04d.png" % i), "wb") as f:
                f.write(png)
            got += 1
        time.sleep(delay)
    print("saved %d captchas to %s (%d session renewals)" % (got, outdir, miss))


if __name__ == "__main__":
    main()
