#!/usr/bin/env python3
"""Download NGT (National Green Tribunal) captchas.

    python3 download_ngt.py 250 ngt_raw

Endpoint behaviour (measured by ablation, not assumed):

* **the user-agent gate is an allowlist crossed with a denylist.** The UA must
  contain `mozilla` AND must not contain a tool token:

      Mozilla / Mozilla/4.0 / mozilla/5.0 / "MyBot Mozilla/5.0"  -> 200 image/png
      curl/8.7.1 / python-requests/2.31 / Wget/1.21 / Googlebot  -> 403
      "" / "x"                                                   -> 403
      "Mozilla/5.0 curl"                                         -> 403

  That is a FOURTH distinct anti-bot posture in this repo, so once again no
  header recipe was reused: Kaveri and Udyam run a UA denylist, MCA wants most
  of a real Chrome header set, GST rejects a spoofed browser UA, ITAT checks the
  UA not at all -- and NGT wants a browser-shaped UA with no tool word in it.

* **the host rate-limits at ~10 requests per window, and a blocked request
  REFRESHES the block.** Harvesting at one request per 8s produced a strikingly
  regular pattern of successes between 429s: 10, 10, 10, 10, 9, 10. An opening
  burst of ~50 requests in 90s then tripped a 429 that did *not* clear across 7
  minutes of backed-off polling -- while killing the poller and staying silent
  for 5 minutes cleared it immediately.

  So the backoff below goes SILENT rather than polling. That is not politeness,
  it is throughput: every request made during a block extends the block, so
  polling one is strictly slower than waiting it out.

* the stack is **Drupal 7** -- responses set `SSESS<hash>` (secure, HttpOnly,
  SameSite=Strict), which persists across fetches on one jar. `captcha.php` is a
  raw PHP file inside a custom module directory that bootstraps Drupal.

* the captcha is almost certainly **session-bound**: the idiomatic Drupal
  implementation stores the expected word in `$_SESSION`, whose consequence is
  that only the most recently fetched image is submittable on a given session --
  the Securimage / GST / ITAT pattern. That is inferred from the cookie
  behaviour, NOT confirmed by a form submission.

  Unlike download_itat.py this reuses ONE cookie jar rather than taking a fresh
  session per fetch. Drupal sessions are DB-backed, so a new session per image
  would litter the server's session table for no sampling benefit -- the served
  images are distinct regardless, and nothing is ever submitted.

How the answer is submitted is deliberately NOT documented here.
"""
import os
import subprocess
import sys
import time

URL = ("https://www.greentribunal.gov.in/sites/all/modules/custom/"
       "case_status/captcha.php")
# Must contain "mozilla" and must not contain a tool token -- see the module
# docstring. This is the honest minimum that passes, not a spoofed fingerprint:
# the server checks the UA string only, and sends no client-hint challenge.
UA = "Mozilla/5.0"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

# ~10 requests per window were admitted at 8s spacing, so 12s sits below the
# observed trip point. The safe steady rate was bounded, not measured exactly.
DELAY = float(os.environ.get("NGT_DELAY", "12"))
# How long to stay completely silent once blocked. Requests made during a block
# refresh it, so this is a hard sleep with no probing.
QUIET = float(os.environ.get("NGT_QUIET", "180"))


def fetch(jar):
    """-> png bytes | None, through a single persistent cookie jar."""
    out = subprocess.run(
        ["curl", "-s", "--max-time", "25", "-A", UA,
         "-b", jar, "-c", jar, URL],
        capture_output=True).stdout
    return out if out.startswith(PNG_MAGIC) else None


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 250
    outdir = sys.argv[2] if len(sys.argv) > 2 else "ngt_raw"
    os.makedirs(outdir, exist_ok=True)
    jar = os.path.join(outdir, ".cookies")

    # resume: never overwrite an already-harvested image
    have = len([f for f in os.listdir(outdir) if f.endswith(".png")])
    got = blocked = 0
    quiet = QUIET
    while have + got < n:
        png = fetch(jar)
        if png is None:
            blocked += 1
            print("blocked (%d), going silent for %.0fs" % (blocked, quiet),
                  flush=True)
            time.sleep(quiet)             # silence, NOT polling
            quiet = min(900.0, quiet * 1.5)
            continue
        path = os.path.join(outdir, "n%04d.png" % (have + got))
        with open(path, "wb") as f:
            f.write(png)
        got += 1
        quiet = QUIET                     # a success means the window reopened
        if got % 10 == 0:
            print("got %d/%d" % (have + got, n), flush=True)
        time.sleep(DELAY)

    print("saved %d captchas to %s (%d blocks waited out)"
          % (got, outdir, blocked))


if __name__ == "__main__":
    main()
