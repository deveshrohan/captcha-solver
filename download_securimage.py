"""Download distinct real securimage captchas from the eCourts endpoint.

Usage: python3 download_securimage.py [TARGET] [OUTDIR]
Dedupes by content hash (the endpoint caches within short time windows), paces
requests politely, and stops at TARGET distinct images or a request cap.
NOTE: downloaded images are UNLABELED (no ground truth from the server).
"""
import hashlib
import os
import sys
import time
import urllib.request

# Point this at your own Securimage endpoint (or set SECURIMAGE_URL). Only use it
# against a service you are authorized to automate — see README acceptable use.
BASE = os.environ.get("SECURIMAGE_URL", "https://YOUR-SECURIMAGE-ENDPOINT/securimage_show.php")
TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
OUTDIR = sys.argv[2] if len(sys.argv) > 2 else "captcha_2/downloads"
MAX_ATTEMPTS = TARGET * 4
DELAY = 0.5

os.makedirs(OUTDIR, exist_ok=True)
seen = set()
# seed dedup set with anything already downloaded
for f in os.listdir(OUTDIR):
    if f.endswith(".png"):
        seen.add(hashlib.md5(open(os.path.join(OUTDIR, f), "rb").read()).hexdigest())

kept = len(seen)
attempts = 0
log = open(os.path.join(OUTDIR, "_progress.log"), "w")


def note(msg):
    log.write(msg + "\n"); log.flush()
    print(msg, flush=True)


note(f"start: target={TARGET} already={kept}")
while kept < TARGET and attempts < MAX_ATTEMPTS:
    attempts += 1
    url = f"{BASE}?ns={attempts}_{int(time.time()*1000) % 100000}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (captcha-ocr-eval)"})
        data = urllib.request.urlopen(req, timeout=15).read()
    except Exception as e:
        note(f"attempt {attempts}: error {e}")
        time.sleep(1.0)
        continue
    h = hashlib.md5(data).hexdigest()
    if h not in seen:
        seen.add(h)
        with open(os.path.join(OUTDIR, f"img_{kept:04d}.png"), "wb") as fh:
            fh.write(data)
        kept += 1
        if kept % 50 == 0:
            note(f"kept {kept}/{TARGET}  (attempts {attempts})")
    time.sleep(DELAY)

note(f"DONE: kept {kept} distinct in {attempts} attempts")
log.close()
