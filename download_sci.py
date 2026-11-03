#!/usr/bin/env python3
"""Download Supreme Court of India captchas (securimage-wp, arithmetic).

    python3 download_sci.py 60 sci_raw        # 60 sessions x 2 renders
    SCI_RENDERS=6 SCI_START=100 python3 download_sci.py 30 sci_raw

Endpoint behaviour (measured 2026-11-03, not assumed):

* GET https://www.sci.gov.in/case-status-party-name/ carries the hidden
  `scid` (and a `tok_<hash>`). The page sets no cookie, and the plain URL is
  served from a page cache, so every visitor gets the SAME scid.
* The expression is stored server-side PER SCID, not per session. A fresh
  session that fetches a cached scid gets the expression already stored for
  it (9 sessions in a row read `6 - 2`). Adding a throwaway query string
  (`?nc=<random>`) gets a page with a fresh scid, and so a new expression.
* GET https://www.sci.gov.in/?_siwp_captcha&id=<scid> -> 220x80 RGB PNG,
  e.g. `10 - 2`, and sets PHPSESSID. Fetching the same scid again
  RE-RENDERS THE SAME expression with fresh placement, rotation and size,
  so each answer can be read from several renders.

Each session here is one page GET and two image GETs, DELAY seconds apart.
The two renders share a label. Stops on the first non-image response.
Nothing is submitted.
"""
import os
import random
import re
import sys
import time

import requests

DELAY = float(os.environ.get("SCI_DELAY", "2"))
BASE = "https://www.sci.gov.in"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126 Safari/537.36")
RENDERS = int(os.environ.get("SCI_RENDERS", "2"))
QUIET = 60          # silence after a dropped connection before a retry


def get(s, url):
    for attempt in range(3):
        try:
            return s.get(url, timeout=(10, 40))
        except requests.RequestException as e:
            if attempt == 2:
                sys.exit("sci: stopped: %s" % e)
            time.sleep(QUIET)


def is_png(body):
    return body.startswith(b"\x89PNG\r\n\x1a\n") and body[-8:-4] == b"IEND"


def main():
    n = int(sys.argv[1])
    outdir = sys.argv[2] if len(sys.argv) > 2 else "sci_raw"
    os.makedirs(outdir, exist_ok=True)
    got, i = 0, int(os.environ.get("SCI_START", "0"))
    while got < n:
        paths = [os.path.join(outdir, "sci_%04d_%d.png" % (i, r)) for r in range(RENDERS)]
        i += 1
        if os.path.exists(paths[0]):                  # resume
            continue
        s = requests.Session()
        s.headers["User-Agent"] = UA
        html = get(s, BASE + "/case-status-party-name/?nc=%d" % random.getrandbits(32)).text
        m = re.search(r'name=["\']scid["\']\s+value=["\']([^"\']+)', html)
        if not m:
            sys.exit("sci: stopped at %d: no scid on the page" % (i - 1))
        for p in paths:
            time.sleep(DELAY)
            r = get(s, BASE + "/?_siwp_captcha&id=" + m.group(1))
            if r.status_code != 200 or not is_png(r.content):
                sys.exit("sci: stopped at %d: http %d, %d bytes, %s" % (
                    i - 1, r.status_code, len(r.content), r.headers.get("content-type")))
            with open(p, "wb") as f:
                f.write(r.content)
        got += 1
        s.close()
        time.sleep(DELAY)
    print("sci downloaded", got, "sessions ->", outdir)


if __name__ == "__main__":
    main()
