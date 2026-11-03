#!/usr/bin/env python3
"""Download CESTAT captchas for labelling.

    python3 download_cestat.py 150 cestat_raw

Image GETs only, DELAY seconds (default 2) between requests. Stops (no retry)
on the first non-image response, so a block or a changed endpoint is noticed
rather than hammered.

Endpoint (from the litigation scraper's crawlers/cestat.py, 2026-06-30):
`cestat.gov.in/captcha?rand=<random>` after one GET of `/casestatus`, which
sets the ci_session and csrf cookies. A 100x35 RGB JPEG.
"""
import os
import random
import sys
import time

import requests

DELAY = float(os.environ.get("CESTAT_DELAY", "2"))
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126 Safari/537.36")


def main():
    n = int(sys.argv[1])
    outdir = sys.argv[2] if len(sys.argv) > 2 else "cestat_raw"
    os.makedirs(outdir, exist_ok=True)
    s = requests.Session()
    s.headers["User-Agent"] = UA
    s.get("https://cestat.gov.in/casestatus", timeout=(10, 30))
    time.sleep(DELAY)
    i = got = 0
    while got < n:
        path = os.path.join(outdir, "cestat_%04d.jpg" % i)
        i += 1
        if os.path.exists(path):          # resume
            continue
        r = s.get("https://cestat.gov.in/captcha?rand=%.16f" % random.random(),
                  timeout=(10, 30))
        body = r.content
        if r.status_code != 200 or not body.startswith(b"\xff\xd8\xff") \
                or not body.endswith(b"\xff\xd9"):
            sys.exit("cestat: stopped at %d: http %d, %d bytes, %s" % (
                i - 1, r.status_code, len(body), r.headers.get("content-type")))
        with open(path, "wb") as f:
            f.write(body)
        got += 1
        time.sleep(DELAY)
    print("cestat downloaded", got, "->", outdir)


if __name__ == "__main__":
    main()
