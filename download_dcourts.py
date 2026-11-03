#!/usr/bin/env python3
"""Download district-court (dcourts.gov.in) Securimage-WP captchas.

    python3 download_dcourts.py 150 [renders] [host ...]

In the live flow (the litigation scraper's dcourts.py), the case-search page
carries a `scid` and `/?_siwp_captcha&id=<scid>` draws that scid's code. The
code is sticky: re-fetching re-renders the SAME code with fresh noise and
warp, and a wrong guess does not change it.

The pages are cached, so every visitor sees the same scid until the cache
turns over. Measured 2026-11-03: kota served one scid on 7 fresh sessions.
The plugin's siwp_captcha.php creates a new code for any 40-hex id it does
not know. So this script asks for a random id per code and saves `renders`
images of it (`<host>_<id10>_<r>.png`). Several renders of one code make
eye-labelling reliable and let the reader vote.

The sites run the securimage-wp plugin's defaults (checked against the
plugin source): 225x85, 5 characters, bg #F2F2F2, text/lines/noise #7D7D7D,
3 lines, noise level 5, perturbation 0.75, AHGBold.

Page and image GETs only, DELAY seconds apart, cycling over the hosts.
Stops on the first non-image response.
"""
import json
import os
import secrets
import sys
import time

import requests

DELAY = float(os.environ.get("DC_DELAY", "1.5"))
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126 Safari/537.36")
HOSTS = ["kota", "jaipur", "pune", "lucknow", "patna", "bangalore", "bhopal", "kolkata"]
OUT = "dcourts_raw"


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    renders = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    hosts = sys.argv[3:] or HOSTS
    os.makedirs(OUT, exist_ok=True)
    # one manifest per host list, so one process per host can run side by side
    manifest_path = os.path.join(OUT, "manifest_%s.json" % "-".join(hosts))
    manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else {}
    s = requests.Session()
    s.headers["User-Agent"] = UA
    got, i, dead = 0, 0, set()
    while got < n and len(dead) < len(hosts):
        host = hosts[i % len(hosts)]
        i += 1
        if host in dead:
            continue
        scid = secrets.token_hex(20)        # an unknown id gets a new code
        code_id = "%s_%s" % (host, scid[:10])
        names = []
        for r in range(renders):
            try:
                resp = s.get("https://%s.dcourts.gov.in/?_siwp_captcha&id=%s" % (host, scid),
                             timeout=40)
            except requests.RequestException as e:
                print(host, "failed:", e, file=sys.stderr)
                dead.add(host)
                break
            body = resp.content
            if resp.status_code != 200 or not body.startswith(b"\x89PNG"):
                sys.exit("%s: stopped: http %d, %d bytes, %s" % (
                    host, resp.status_code, len(body), resp.headers.get("content-type")))
            name = "%s_%d.png" % (code_id, r)
            with open(os.path.join(OUT, name), "wb") as f:
                f.write(body)
            names.append(name)
            time.sleep(DELAY)
        if len(names) == renders:
            manifest[code_id] = dict(host=host, scid=scid, renders=names)
            json.dump(manifest, open(manifest_path, "w"), indent=1)
            got += 1
    print("downloaded", got, "codes ->", OUT, "dead hosts:", sorted(dead))


def load_manifest(out=OUT):
    """All codes from every manifest_*.json: {code_id: {host, scid, renders}}."""
    import glob
    m = {}
    for p in sorted(glob.glob(os.path.join(out, "manifest*.json"))):
        m.update(json.load(open(p)))
    return m


if __name__ == "__main__":
    main()
