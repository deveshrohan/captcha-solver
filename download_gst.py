#!/usr/bin/env python3
"""Download GST portal captchas (services.gst.gov.in).

Measured behaviour of the endpoint (verified, not assumed):

* the `rnd` query parameter is **pure cache-busting, not a seed**. Requesting
  the same `rnd` twice returns two *different* images, and omitting `rnd`
  entirely still returns a valid captcha. Its only job is to stop the browser
  reusing a cached response.
* each response sets `CaptchaCookie=<32 hex>` (Domain=.gst.gov.in, Secure,
  HttpOnly) and that value **rotates on every request**. That is what binds a
  captcha to a session: the server remembers the answer for the most recent
  cookie, so only the last image fetched is valid to submit. Fetch and submit
  through the same cookie jar.
* `TS0134d082` is the F5 BIG-IP ASM (WAF) cookie. That WAF rejects a *spoofed*
  browser User-Agent — a bare "Mozilla/5.0" with none of the matching client
  hints gets a 200-with-HTML "Request Rejected" page. Plain curl with its
  default UA is accepted, so this shells out to curl rather than using requests.

    python3 download_gst.py 400 gst_raw
"""
import os
import random
import subprocess
import sys
import time

URL = "https://services.gst.gov.in/services/captcha"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def fetch():
    rnd = "0.%d" % random.randint(10 ** 15, 10 ** 16 - 1)
    out = subprocess.run(
        ["curl", "-s", "--max-time", "20", f"{URL}?rnd={rnd}"],
        capture_output=True,
    ).stdout
    return out if out.startswith(PNG_MAGIC) else None


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    outdir = sys.argv[2] if len(sys.argv) > 2 else "gst_raw"
    delay = float(os.environ.get("GST_DELAY", "0.3"))
    os.makedirs(outdir, exist_ok=True)

    got = fails = 0
    for i in range(n):
        png = fetch()
        if png:
            with open(os.path.join(outdir, "g%04d.png" % i), "wb") as f:
                f.write(png)
            got += 1
            fails = 0
        else:
            fails += 1
            print("  [%d] rejected" % i, flush=True)
            if fails >= 10:
                print("rejected 10x in a row - stopping", flush=True)
                break
            time.sleep(2 * fails)
        time.sleep(delay)
    print("saved %d captchas to %s" % (got, outdir))


if __name__ == "__main__":
    main()
