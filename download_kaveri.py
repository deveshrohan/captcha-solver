#!/usr/bin/env python3
"""Download Kaveri portal captchas (kaveri.karnataka.gov.in).

    python3 download_kaveri.py 400 kaveri_raw

Endpoint behaviour (measured by ablation, not assumed):

* the gate is a user-agent **denylist**, not an allowlist. `curl/8.x` and
  `python-requests/2.31` are refused with 403 and an empty body; an *empty*
  UA, a bare `Mozilla/5.0`, and even `x` all return 200. This is the opposite
  of both neighbours in this repo — MCA needs most of a real Chrome header set,
  and GST rejects a spoofed browser UA while accepting plain curl's. So the
  only header that matters here is one that does not look like a known bot
  client; nothing else is required.
* **no cookies are involved at all.** The server sets none, and none are sent.
* instead each response carries the captcha id in an `i` response header,
  published to browser JS via `Access-Control-Expose-Headers: i`. 186 fetches
  returned 186 distinct GUIDs, so every fetch is an independent captcha.
  That is unlike Securimage and GST, where a fetch rotates the code held in the
  session and only the newest image is submittable.

The ids are written to `ids.json` in the output directory alongside the PNGs,
since a caller validating a solve needs the id that came with that image.
(How the id is submitted is NOT documented here: the portal's Angular bundle is
obfuscated and I could not confirm the validate call, so this script records the
id and stops short of claiming a contract it did not verify.)
"""
import json
import os
import subprocess
import sys
import time

URL = "https://kaveri.karnataka.gov.in/api/Generate"
# Anything that does not match the server's bot denylist. Deliberately minimal:
# a full spoofed Chrome fingerprint is not needed and would misrepresent what
# the endpoint actually checks.
UA = "Mozilla/5.0"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def fetch(hdr_path):
    """-> (png bytes | None, captcha id | None)."""
    out = subprocess.run(
        ["curl", "-s", "--max-time", "25", "-A", UA, "-D", hdr_path, URL],
        capture_output=True).stdout
    if not out.startswith(PNG_MAGIC):
        return None, None
    cid = None
    try:
        with open(hdr_path, "r", errors="replace") as f:
            for line in f:
                if line.lower().startswith("i:"):
                    cid = line.split(":", 1)[1].strip()
    except OSError:
        pass
    return out, cid


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    outdir = sys.argv[2] if len(sys.argv) > 2 else "kaveri_raw"
    delay = float(os.environ.get("KAVERI_DELAY", "0.3"))
    os.makedirs(outdir, exist_ok=True)

    ids_path = os.path.join(outdir, "ids.json")
    ids = json.load(open(ids_path)) if os.path.exists(ids_path) else {}
    hdr = os.path.join(outdir, ".headers")

    got = miss = 0
    for i in range(n):
        png, cid = fetch(hdr)
        if png is None:                       # 403 bursts are rate limiting
            miss += 1
            time.sleep(min(30.0, 2.0 * (1 + miss)))
            png, cid = fetch(hdr)
        if png:
            name = "k%04d.png" % i
            with open(os.path.join(outdir, name), "wb") as f:
                f.write(png)
            if cid:
                ids[name] = cid
            got += 1
        time.sleep(delay)

    json.dump(ids, open(ids_path, "w"), indent=1, sort_keys=True)
    if os.path.exists(hdr):
        os.remove(hdr)
    print("saved %d captchas to %s (%d retries), %d ids -> %s"
          % (got, outdir, miss, len(ids), ids_path))


if __name__ == "__main__":
    main()
