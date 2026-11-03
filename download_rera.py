#!/usr/bin/env python3
"""Download RERA captchas (UP, TG, GA) for labelling.

    python3 download_rera.py up 120        # -> up_rera_raw/
    python3 download_rera.py tg 120        # -> tg_rera_raw/
    python3 download_rera.py ga 120        # -> ga_rera_raw/

Every captcha is bound to its session cookie, so each one is fetched after a
page GET in the same session (one session per captcha). Page and image GETs
only, at least DELAY seconds apart. Stops on the first non-image response.
Endpoints are from the RERA harvesters (rera-states/<st>/harvest/):

* up  GET https://up-rera.in/frm_Rc_Report.aspx, then the <img> whose src
      contains "CaptchaImage" (MSCaptcha). 150x35 JPEG.
* tg  GET https://rerait.telangana.gov.in/SearchList/Search for the session,
      then /SearchList/SearchCaptcha. 158x48 PNG.
* ga  GET https://rera.goa.gov.in/reraApp/ for the session, then
      captcha.jpg?x=<time_ns>. 200x45 JPEG. Each GET holds the GA
      harvester's flock (rera-states/ga/harvest/host.lock) so the two
      never overlap on that host.
"""
import contextlib
import fcntl
import html
import os
import re
import sys
import time
from urllib.parse import urljoin

import requests

DELAY = float(os.environ.get("RERA_DELAY", "2"))
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126 Safari/537.36")
GA_LOCK = os.path.expanduser("~/blrMent-corpus/rera-states/ga/harvest/host.lock")

JPEG, PNG = b"\xff\xd8\xff", b"\x89PNG\r\n\x1a\n"


@contextlib.contextmanager
def ga_lock():
    if not os.path.exists(os.path.dirname(GA_LOCK)):
        yield
        return
    with open(GA_LOCK, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def fetch_up(s):
    page = "https://up-rera.in/frm_Rc_Report.aspx"
    body = s.get(page, timeout=40).text
    m = re.search(r'src=["\']([^"\']*CaptchaImage[^"\']*)["\']', body)
    if not m:
        raise RuntimeError("no CaptchaImage <img> on %s" % page)
    time.sleep(DELAY)
    return s.get(urljoin(page, html.unescape(m.group(1))), timeout=40, headers={"Referer": page}), JPEG


def fetch_tg(s):
    base = "https://rerait.telangana.gov.in"
    s.get(base + "/SearchList/Search", timeout=40)
    time.sleep(DELAY)
    return s.get(base + "/SearchList/SearchCaptcha", timeout=40,
                 headers={"Referer": base + "/SearchList/Search"}), PNG


def fetch_ga(s):
    base = "https://rera.goa.gov.in/reraApp/"
    with ga_lock():
        s.get(base, timeout=40)
    time.sleep(DELAY)
    with ga_lock():
        return s.get(base + "captcha.jpg?x=%d" % time.time_ns(), timeout=40,
                     headers={"Referer": base}), JPEG


FETCH = {"up": (fetch_up, "jpg"), "tg": (fetch_tg, "png"), "ga": (fetch_ga, "jpg")}


def main():
    st, n = sys.argv[1], int(sys.argv[2])
    fetch, ext = FETCH[st]
    out = sys.argv[3] if len(sys.argv) > 3 else "%s_rera_raw" % st
    os.makedirs(out, exist_ok=True)
    i = got = 0
    while got < n:
        path = os.path.join(out, "%s_%04d.%s" % (st, i, ext))
        i += 1
        if os.path.exists(path):
            continue
        s = requests.Session()
        s.headers["User-Agent"] = UA
        r, magic = fetch(s)
        if r.status_code != 200 or not r.content.startswith(magic):
            sys.exit("%s: stopped at %d: http %d, %d bytes, %s" % (
                st, i - 1, r.status_code, len(r.content), r.headers.get("content-type")))
        with open(path, "wb") as f:
            f.write(r.content)
        got += 1
        time.sleep(DELAY)
    print(st, "downloaded", got, "->", out)


if __name__ == "__main__":
    main()
