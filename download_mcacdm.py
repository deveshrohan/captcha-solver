#!/usr/bin/env python3
"""Download MCA CDM export captchas (mcacdm.nic.in) for labelling.

    python3 download_mcacdm.py 150 mcacdm_raw

`https://mcacdm.nic.in/captcha.php?context=company|llp`: a two-colour 120x38
PNG, served since ~2026-09-08 on the company/LLP master-data export pages.
The host speaks TLS 1.3 only, so a Python linked against old LibreSSL (the
macOS system python3) cannot reach it; this goes through curl with a cookie
jar. Image GETs only, MCACDM_DELAY seconds apart (default 1.5), alternating
the two contexts. Stops at the first non-image response.

How an answer is submitted is deliberately not documented here.
"""
import os
import subprocess
import sys
import time

DELAY = float(os.environ.get("MCACDM_DELAY", "1.5"))
UA = "Mozilla/5.0"
MAGIC = b"\x89PNG\r\n\x1a\n"


def fetch(jar, context):
    url = "https://mcacdm.nic.in/captcha.php?context=%s&r=%f" % (context, time.time())
    out = subprocess.run(["curl", "-sS", "-A", UA, "-b", jar, "-c", jar,
                          "--max-time", "40", "-w", "\n%{http_code}", url],
                         capture_output=True)
    if out.returncode:
        raise RuntimeError(out.stderr.decode(errors="replace").strip())
    body, _, code = out.stdout.rpartition(b"\n")
    return int(code), body


def main():
    n = int(sys.argv[1])
    outdir = sys.argv[2] if len(sys.argv) > 2 else "mcacdm_raw"
    os.makedirs(outdir, exist_ok=True)
    jar = os.path.join(outdir, ".cookies")
    i = got = 0
    while got < n:
        context = ("company", "llp")[i % 2]
        path = os.path.join(outdir, "mcacdm_%s_%04d.png" % (context, i))
        i += 1
        if os.path.exists(path):          # resume
            continue
        code, body = fetch(jar, context)
        if code != 200 or not body.startswith(MAGIC) or body[-8:-4] != b"IEND":
            sys.exit("stopped at %d: http %d, %d bytes" % (i - 1, code, len(body)))
        with open(path, "wb") as f:
            f.write(body)
        got += 1
        time.sleep(DELAY)
    print("downloaded", got, "->", outdir)


if __name__ == "__main__":
    main()
