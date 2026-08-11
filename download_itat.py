#!/usr/bin/env python3
"""Download ITAT portal captchas (itat.gov.in).

    python3 download_itat.py 400 itat_raw

Endpoint behaviour (measured by ablation, not assumed):

* **there is no user-agent gate at all.** `curl/8.x`, `python-requests/2.31`,
  an empty UA, a bare `Mozilla/5.0` and even `x` every one returns 200 with a
  valid PNG. That is a third distinct anti-bot posture in this repo: Kaveri and
  Udyam run a UA *denylist* (curl/python refused), MCA wants most of a real
  Chrome header set, GST rejects a spoofed browser UA — and ITAT checks the UA
  not at all. So no header recipe is reused here; the gate was re-measured.
* the stack is **CodeIgniter** (PHP). Every response sets `ci_session`,
  `csrf_cookie_name` and a `uid` cookie, and the PNG carries the libgd
  fingerprint (`pHYs` = 3780 ppm = 96 DPI), i.e. it is drawn by PHP's GD — the
  `create_captcha()` helper lineage, heavily customised (truecolor, black
  antialiased text, light-blue line noise, no border).
* the response's `Content-Type` is `text/html` even though the body is a PNG;
  the controller just echoes the image bytes. Validity is checked by the PNG
  magic, not the header.
* the captcha is almost certainly **session-bound**. Within one cookie jar the
  `ci_session` is minted on the first fetch and then persists, and this is a
  CodeIgniter app, whose idiomatic captcha stores the expected word in the
  session (or a DB row keyed by it). The natural consequence is that only the
  MOST RECENTLY fetched image is submittable on a given session — the
  Securimage / Udyam pattern, not Kaveri's independent-id one. That is inferred
  from the cookie behaviour, NOT confirmed by a form submission: the captcha
  guards ITAT's case-status / e-filing forms, and confirming the validate
  contract would mean posting to a live government portal, which this script
  does not do (the same line download_udyam.py and download_kaveri.py draw).

* because of the above this downloader fetches each captcha on a **fresh
  session** (no cookie jar), so the saved PNGs are independent samples rather
  than a chain in which only the last was ever valid. 40 fresh fetches returned
  40 distinct images.

How the answer is submitted is deliberately NOT documented here.
"""
import os
import subprocess
import sys
import time

URL = "https://itat.gov.in//captcha/show"
# The endpoint has no UA gate, so this value is not load-bearing; a plain,
# honest UA is sent rather than a spoofed browser fingerprint the server never
# checks.
UA = "Mozilla/5.0"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def fetch():
    """-> png bytes | None. No cookie jar: each fetch is an independent session."""
    out = subprocess.run(
        ["curl", "-s", "--max-time", "25", "-A", UA, URL],
        capture_output=True).stdout
    return out if out.startswith(PNG_MAGIC) else None


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    outdir = sys.argv[2] if len(sys.argv) > 2 else "itat_raw"
    delay = float(os.environ.get("ITAT_DELAY", "0.3"))
    os.makedirs(outdir, exist_ok=True)

    got = miss = 0
    for i in range(n):
        png = fetch()
        if png is None:                       # transient error / rate limiting
            miss += 1
            time.sleep(min(30.0, 2.0 * (1 + miss)))
            png = fetch()
        if png:
            with open(os.path.join(outdir, "t%04d.png" % i), "wb") as f:
                f.write(png)
            got += 1
        time.sleep(delay)

    print("saved %d captchas to %s (%d retries)" % (got, outdir, miss))


if __name__ == "__main__":
    main()
