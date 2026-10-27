#!/usr/bin/env python3
"""Download High Court captchas (Gujarat, Madras, Calcutta, Karnataka) for labelling.

    python3 download_hc.py gujhc 150 gujhc_raw

One worker per portal, image GETs only, DELAY seconds between responses.
Stops (no retry) on the first non-image response, so a block or a changed
endpoint is noticed rather than hammered.

Endpoint behaviour (measured 2026-10-27, not assumed):

* gujhc  `gujarathc-casestatus.nic.in/gujarathc/CaptchaServlet`
  Needs `ct=S`. Without it, and with `ct=A`, the servlet answers
  200 with an EMPTY text/html body. `tm` (the page's ms timestamp) is
  optional. No UA or cookie gate: curl's default UA with no cookie gets the
  image. 200x50 RGB PNG, no ancillary chunks.
* madhc  `hcmadras.tn.gov.in/securimage_show.php?namespace=contact&<hex>`
  Securimage (PHP). Sets PHPSESSID on the first fetch. No UA gate.
  120x60 RGB PNG.
* calhc  `www.calcuttahighcourt.gov.in/captcha/default?<8 random chars>`
  Laravel mews/captcha (the `captcha_src('default')` URL shape). Sets the
  Laravel session + XSRF cookies. libgd JPEG, quality 75, 120x36.
* karhc  `judiciary.karnataka.gov.in/captcha.php`
  No UA gate, sets PHPSESSID. A 100x20 libgd JPEG ("default quality")
  served under a WRONG `Content-Type: text/html`, so sniff the magic.
  The server speaks TLS 1.3 ONLY: a Python linked against LibreSSL 2.8 (the
  macOS system python3) fails with TLSV1_ALERT_PROTOCOL_VERSION, so this
  kind is fetched through curl.

How each answer is submitted is deliberately not documented here.
"""
import os
import random
import string
import subprocess
import sys
import time

import requests

DELAY = float(os.environ.get("HC_DELAY", "3"))
# Silence after a dropped connection before trying again.
QUIET = float(os.environ.get("HC_QUIET", "60"))

KINDS = {
    "gujhc": dict(ext="png", magic=b"\x89PNG\r\n\x1a\n",
                  url=lambda: "https://gujarathc-casestatus.nic.in/gujarathc/"
                              "CaptchaServlet?ct=S&tm=%d" % (time.time() * 1000)),
    "madhc": dict(ext="png", magic=b"\x89PNG\r\n\x1a\n",
                  url=lambda: "https://hcmadras.tn.gov.in/securimage_show.php"
                              "?namespace=contact&%032x" % random.getrandbits(128)),
    "calhc": dict(ext="jpg", magic=b"\xff\xd8\xff",
                  url=lambda: "https://www.calcuttahighcourt.gov.in/captcha/"
                              "default?" + "".join(random.choices(
                                  string.ascii_letters + string.digits, k=8))),
    "karhc": dict(ext="jpg", magic=b"\xff\xd8\xff", curl=True,
                  url=lambda: "https://judiciary.karnataka.gov.in/captcha.php"),
}


def complete(kind, body):
    k = KINDS[kind]
    if not body.startswith(k["magic"]):
        return False
    # a cut-off body can start with the right magic; check the trailer too
    return body[-8:-4] == b"IEND" if k["ext"] == "png" else body.endswith(b"\xff\xd9")


class _Response:
    def __init__(self, status_code, content, ctype):
        self.status_code, self.content = status_code, content
        self.headers = {"content-type": ctype}


class _CurlSession:
    """The two calls main() makes on a requests.Session, done with curl and
    a cookie jar, for servers this Python's TLS stack cannot reach."""

    def __init__(self, jar):
        self.jar = jar

    def get(self, url, timeout=(10, 30)):
        out = subprocess.run(
            ["curl", "-sS", "-A", "Mozilla/5.0", "-b", self.jar, "-c", self.jar,
             "--connect-timeout", str(timeout[0]), "--max-time", str(sum(timeout)),
             "-w", "\n%{http_code} %{content_type}", url],
            capture_output=True)
        if out.returncode:
            raise requests.ConnectionError(out.stderr.decode(errors="replace").strip())
        body, _, meta = out.stdout.rpartition(b"\n")
        code, _, ctype = meta.decode().partition(" ")
        return _Response(int(code), body, ctype)

    def close(self):
        pass


def main():
    kind = sys.argv[1]
    n = int(sys.argv[2])
    outdir = sys.argv[3] if len(sys.argv) > 3 else kind + "_raw"
    os.makedirs(outdir, exist_ok=True)
    k = KINDS[kind]
    if k.get("curl"):
        s = _CurlSession(os.path.join(outdir, ".cookies"))
    else:
        s = requests.Session()
        s.headers["User-Agent"] = "Mozilla/5.0"
    i = 0
    got = 0
    while got < n:
        path = os.path.join(outdir, "%s_%04d.%s" % (kind, i, k["ext"]))
        i += 1
        if os.path.exists(path):          # resume
            continue
        for attempt in range(4):
            try:
                r = s.get(k["url"](), timeout=(10, 30))
                break
            except requests.RequestException as e:
                # Gujarat sometimes drops a kept-alive connection. Go silent
                # (do not poll) and retry on a fresh connection.
                if attempt == 3:
                    sys.exit("%s: stopped at %d: %s" % (kind, i - 1, e))
                time.sleep(QUIET)
                s.close()
        if r.status_code != 200 or not complete(kind, r.content):
            sys.exit("%s: stopped at %d: http %d, %d bytes, %s" % (
                kind, i - 1, r.status_code, len(r.content),
                r.headers.get("content-type")))
        with open(path, "wb") as f:
            f.write(r.content)
        got += 1
        time.sleep(DELAY)
    print(kind, "downloaded", got, "->", outdir)


if __name__ == "__main__":
    main()
