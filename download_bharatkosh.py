#!/usr/bin/env python3
"""Download Bharatkosh (NTRP) captchas, in same-answer groups.

    python3 download_bharatkosh.py 40 5 bharatkosh_raw

fetches 40 groups of 5 renders each -> b0000_r0.png .. b0039_r4.png. Every
render in a group shows the SAME text (see `New` below).

Endpoint behaviour (measured by ablation, not assumed):

* **no request gate.** No UA, an empty UA, `curl/8.x` and
  `python-requests/2.31` all get 200. No cookie or referer is needed for a
  first fetch. Yet another posture, so again no header recipe was reused.

* **`New` selects new-text vs re-render, and it is the whole point.**

      ?New=1, or no param   -> draws a NEW random text into the session
      ?New=0, with session  -> RE-RENDERS the session's existing text: same
                               characters, fresh fonts, colours, warp, noise
      ?New=0, no session    -> 500 (nothing to re-render)

  Five consecutive fetches on one jar (New=1, then New=0 x3, then no param)
  read g2D4aK, g2D4aK, g2D4aK, g2D4aK, then a different text. So one answer
  can be observed through as many independent renders as wanted -- which
  makes hand-labelling reliable (a glyph cut by a bar in one render is whole
  in the next) and lets a reader vote across renders.

  That a re-render leaves the submittable answer unchanged is inferred from
  the text staying constant, NOT confirmed by a form submission.

* the stack is **ASP.NET MVC** -- `ASP.NET_SessionId` (secure, HttpOnly,
  SameSite=Lax) is set on the first fetch and persists on the jar. The body is
  a PNG (GDI+ fingerprint: sRGB + gAMA + pHYs chunks, zlib `78 5e`) served
  under a wrong `Content-Type: image/gif`, so sniff the magic, not the header.

* each group takes a FRESH jar: the session is what binds a group together,
  so a new group is by construction a new session.

How the answer is submitted is deliberately NOT documented here.
"""
import os
import subprocess
import sys
import time

URL = "https://bharatkosh.gov.in/NTRPHome/GenerateCaptcha"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

# No rate limit was hit at this spacing; the trip point was not searched for.
DELAY = float(os.environ.get("BK_DELAY", "2"))
# How long to stay completely silent after a non-PNG response.
QUIET = float(os.environ.get("BK_QUIET", "120"))


def fetch(jar, new):
    """-> png bytes | None, through the group's cookie jar."""
    out = subprocess.run(
        ["curl", "-s", "--max-time", "25", "-b", jar, "-c", jar,
         "%s?New=%d" % (URL, 1 if new else 0)],
        capture_output=True).stdout
    # a complete PNG, not just a PNG-shaped start: a cut-off body must not be
    # saved into a group as if it were a render
    ok = out.startswith(PNG_MAGIC) and out[-8:-4] == b"IEND"
    return out if ok else None


def main():
    # RENDERS is optional so `selfimprove.py harvest` can call this with the
    # (n, outdir) shape every other downloader takes.
    rest = sys.argv[2:]
    groups = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    renders = int(rest.pop(0)) if rest and rest[0].isdigit() else 5
    outdir = rest[0] if rest else "bharatkosh_raw"
    os.makedirs(outdir, exist_ok=True)
    jar = os.path.join(outdir, ".cookies")

    # resume: a group counts only once its last render is on disk
    g = 0
    while os.path.exists(os.path.join(
            outdir, "b%04d_r%d.png" % (g, renders - 1))):
        g += 1
    blocked = 0
    while g < groups:
        if os.path.exists(jar):
            os.remove(jar)                # fresh session = fresh text
        pngs = []
        for r in range(renders):
            png = fetch(jar, new=(r == 0))
            if png is None:
                break
            pngs.append(png)
            time.sleep(DELAY)
        if len(pngs) < renders:           # never keep a partial group
            blocked += 1
            print("blocked (%d), going silent for %.0fs" % (blocked, QUIET),
                  flush=True)
            time.sleep(QUIET)
            continue
        for r, png in enumerate(pngs):
            with open(os.path.join(outdir, "b%04d_r%d.png" % (g, r)),
                      "wb") as f:
                f.write(png)
        g += 1
        if g % 5 == 0:
            print("got %d/%d groups" % (g, groups), flush=True)

    print("saved %d groups x %d renders to %s (%d blocks waited out)"
          % (groups, renders, outdir, blocked))


if __name__ == "__main__":
    main()
