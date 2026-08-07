#!/usr/bin/env python3
"""Download MCA (mca.gov.in) captchas.

The endpoint returns multipart/mixed with an image/png part and an audio/wav
part (the accessibility audio captcha).

It sits behind Akamai Bot Manager, but — measured by ablation, not assumed — the
Akamai cookies are **not** what gates it: dropping `ak_bmsc` and `bm_sv`
entirely still returns 200. What gates it is the *header fingerprint*. Findings:

* `user-agent` and `referer` are each individually necessary (removing either
  from an otherwise complete set gives 403), and the referer must be an
  mca.gov.in URL — pointing it at example.com is rejected;
* those two alone are not sufficient. Akamai scores the whole set, so the
  request has to carry enough of a real Chrome XHR's client hints and fetch
  metadata. HEADERS below is a known-good set;
* no cookie of any kind is required, so MCA_COOKIE is optional. It is still
  honoured if you set it.

Occasional 403 bursts are rate limiting, not cookie expiry; the loop backs off.

    python3 download_mca.py 400 mca_raw
"""
import os
import subprocess
import sys
import time

URL = "https://www.mca.gov.in/bin/mca/generateCaptchaWithHMAC"
REFERER = "https://www.mca.gov.in/content/mca/global/en/foportal/fologin.html"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/151.0.0.0 Safari/537.36")


HEADERS = [
    ("accept", "*/*"),
    ("accept-language", "en-GB,en-US;q=0.9,en;q=0.8"),
    ("referer", REFERER),
    ("sec-ch-ua", '"Not=A?Brand";v="99", "Google Chrome";v="151", "Chromium";v="151"'),
    ("sec-ch-ua-mobile", "?0"),
    ("sec-ch-ua-platform", '"macOS"'),
    ("sec-fetch-dest", "empty"),
    ("sec-fetch-mode", "cors"),
    ("sec-fetch-site", "same-origin"),
    ("user-agent", UA),
    ("x-requested-with", "XMLHttpRequest"),
]


def split_multipart(body):
    """-> (png_bytes, wav_bytes). Parts are separated by the '--boundary' marker."""
    png = wav = None
    for part in body.split(b"--boundary"):
        if b"\r\n\r\n" not in part:
            continue
        head, data = part.split(b"\r\n\r\n", 1)
        data = data.rstrip(b"\r\n")
        if b"image/png" in head:
            png = data
        elif b"audio/wav" in head:
            wav = data
    return png, wav


def fetch(cookie=None):
    """Uses curl: the host negotiates a TLS version that the stdlib ssl shipped
    with older system Pythons rejects."""
    cmd = ["curl", "-s", "--max-time", "30", URL]
    if cookie:
        cmd += ["-b", cookie]
    for k, v in HEADERS:
        cmd += ["-H", f"{k}: {v}"]
    body = subprocess.run(cmd, capture_output=True).stdout
    if b"--boundary" not in body[:200]:
        return None, None, "blocked"
    png, wav = split_multipart(body)
    return png, wav, 200


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    outdir = sys.argv[2] if len(sys.argv) > 2 else "mca_raw"
    cookie = os.environ.get("MCA_COOKIE")     # optional; see module docstring
    os.makedirs(outdir, exist_ok=True)
    save_wav = os.environ.get("MCA_SAVE_WAV") == "1"

    got = fails = 0
    for i in range(n):
        png, wav, code = fetch(cookie)
        if png:
            with open(os.path.join(outdir, "m%04d.png" % i), "wb") as f:
                f.write(png)
            if save_wav and wav:
                with open(os.path.join(outdir, "m%04d.wav" % i), "wb") as f:
                    f.write(wav)
            got += 1
            fails = 0
        else:
            fails += 1
            print("  [%d] blocked (HTTP %s)" % (i, code), flush=True)
            if fails >= 5:
                print("blocked 5x in a row - cookie is stale, stopping", flush=True)
                break
            time.sleep(5 * fails)
        time.sleep(float(os.environ.get("MCA_DELAY", "1.0")))
    print("saved %d captchas to %s" % (got, outdir))


if __name__ == "__main__":
    main()
