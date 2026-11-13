#!/usr/bin/env python3
"""Download MCA (mca.gov.in) captchas in the format served since ~Nov 2026.

Same endpoint as download_mca.py (`/bin/mca/generateCaptchaWithHMAC`, a
multipart/mixed body with an image/png and an audio/wav part), but the image
is a new generator at the same 200x80 size: coloured antialiased letters at
random heights under three wavy lines, speckle dots and a rounded border.
Each fetch saves `<name>.png`, `<name>.wav` and
appends the `pre_ct` response header to `pre_ct.jsonl` as {"file", "pre_ct"}.

    python3 download_mca2.py 200 mca2_raw
"""
import json
import os
import subprocess
import sys
import time

from download_mca import URL, split_multipart

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/157.0.0.0 Safari/537.36")
HEADERS = [
    ("accept", "*/*"),
    ("accept-language", "en-GB,en-US;q=0.9,en;q=0.8"),
    ("priority", "u=1, i"),
    ("referer", "https://www.mca.gov.in/content/mca/global/en/foportal/fologin.html"),
    ("sec-ch-ua", '"Not-A.Brand";v="99", "Google Chrome";v="157", "Chromium";v="157"'),
    ("sec-ch-ua-mobile", "?0"),
    ("sec-ch-ua-platform", '"macOS"'),
    ("sec-fetch-dest", "empty"),
    ("sec-fetch-mode", "cors"),
    ("sec-fetch-site", "same-origin"),
    ("user-agent", UA),
    ("x-requested-with", "XMLHttpRequest"),
]


def fetch():
    """-> (png, wav, pre_ct) or (None, None, why)."""
    cmd = ["curl", "-s", "--max-time", "30", "-D", "-", URL]
    for k, v in HEADERS:
        cmd += ["-H", "%s: %s" % (k, v)]
    out = subprocess.run(cmd, capture_output=True).stdout
    head, _, body = out.partition(b"\r\n\r\n")
    pre_ct = None
    for line in head.split(b"\r\n")[1:]:
        k, _, v = line.partition(b":")
        if k.strip().lower() == b"pre_ct":
            pre_ct = v.strip().decode()
    if b"--boundary" not in body[:200]:
        return None, None, "blocked"
    png, wav = split_multipart(body)
    if not png or b"IEND" not in png[-16:]:
        return None, None, "truncated"
    return png, wav, pre_ct


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    outdir = sys.argv[2] if len(sys.argv) > 2 else "mca2_raw"
    delay = float(os.environ.get("MCA_DELAY", "1.5"))
    os.makedirs(outdir, exist_ok=True)
    start = len([f for f in os.listdir(outdir) if f.endswith(".png")])
    got = fails = 0
    log = open(os.path.join(outdir, "pre_ct.jsonl"), "a")
    for i in range(start, start + n):
        png, wav, why = fetch()
        if png:
            log.write(json.dumps({"file": "n%04d" % i, "pre_ct": why}) + "\n")
            log.flush()
            name = os.path.join(outdir, "n%04d" % i)
            with open(name + ".png", "wb") as f:
                f.write(png)
            if wav:
                with open(name + ".wav", "wb") as f:
                    f.write(wav)
            got += 1
            fails = 0
        else:
            fails += 1
            print("  [%d] %s" % (i, why), flush=True)
            if fails >= 5:
                print("blocked 5x in a row, stopping", flush=True)
                break
            time.sleep(10 * fails)
        time.sleep(delay)
    log.close()
    print("saved %d captchas to %s" % (got, outdir))


if __name__ == "__main__":
    main()
