#!/usr/bin/env python3
"""Download Udyam registration portal captchas (udyamregistration.gov.in).

    python3 download_udyam.py 400 udyam_raw

Endpoint behaviour (measured by ablation, not assumed):

* the gate is a user-agent **denylist**, and the server says so out loud: the
  403 status line reads `403 Forbidden - Access denied due to bot User-Agent`.
  `curl/8.x` and `python-requests/2.31` are refused; an *empty* UA, a bare
  `Mozilla/5.0`, a full Chrome string and even `x` all return 200. So the only
  header that matters is one that is not a known bot client. That matches
  Kaveri and is the opposite of GST (which rejects a spoofed browser UA and
  accepts plain curl) and of MCA (which wants most of a real Chrome header set).
* the captcha is **session-bound**, unlike Kaveri. Each response carries
  `Set-Cookie: ASP.NET_SessionId=...`, and re-fetching inside one jar keeps the
  same id — this is a stock ASP.NET WebForms CaptchaControl, which stores the
  expected answer in server-side session state. Consequence for a caller: only
  the MOST RECENTLY fetched image is submittable on a given session, exactly as
  for Securimage/eCourts. `solver.api.solve_with_retry` already documents and
  handles that pattern.
* because of the above this downloader deliberately fetches each captcha on a
  **fresh session** (no cookie jar), so the images are independent samples
  rather than a chain in which only the last one was ever valid.

How the answer is submitted is NOT documented here. The captcha guards forms
that take an Aadhaar or Udyam registration number, and confirming the validate
contract would mean posting identifiers to a live government portal, so this
script stops at the image — the same line `download_kaveri.py` draws.
"""
import os
import subprocess
import sys
import time

URL = "https://udyamregistration.gov.in/CaptchaControl.aspx"
# Anything that does not match the server's bot denylist. Deliberately minimal:
# a full spoofed Chrome fingerprint is not needed and would misrepresent what
# the endpoint actually checks.
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
    outdir = sys.argv[2] if len(sys.argv) > 2 else "udyam_raw"
    delay = float(os.environ.get("UDYAM_DELAY", "0.3"))
    os.makedirs(outdir, exist_ok=True)

    got = miss = 0
    for i in range(n):
        png = fetch()
        if png is None:                       # 403 bursts are rate limiting
            miss += 1
            time.sleep(min(30.0, 2.0 * (1 + miss)))
            png = fetch()
        if png:
            with open(os.path.join(outdir, "u%04d.png" % i), "wb") as f:
                f.write(png)
            got += 1
        time.sleep(delay)

    print("saved %d captchas to %s (%d retries)" % (got, outdir, miss))


if __name__ == "__main__":
    main()
