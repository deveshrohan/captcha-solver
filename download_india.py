"""Image-only public captcha collector: one worker, >=3s after each response.

Set HRMUT_CAPTCHA_URL / MHIGR_CAPTCHA_URL to an authorized image endpoint.
No page refreshes, proxy/route rotation, automatic retries or login/OTP pages.
The shared cumulative ledger counts failures toward the 150-attempt cap.
"""
import argparse
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import time
import requests
from PIL import Image


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('kind',choices=['hrmut','mhigr']);p.add_argument('n',type=int);p.add_argument('folder',type=Path);a=p.parse_args()
 url=os.environ[a.kind.upper()+'_CAPTCHA_URL']
 if not url.startswith('https://'):raise ValueError('HTTPS required')
 if a.n<1:raise ValueError('positive count required')
 a.folder.mkdir(exist_ok=True,parents=True)
 root=Path(__file__).resolve().parent;ledger=root/'india_capture_ledger.jsonl'
 with (root/'.india-captcha.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  prior=[json.loads(l) for l in ledger.read_text().splitlines()] if ledger.exists() else []
  count=sum(e['kind']==a.kind for e in prior)
  if count+a.n>150:raise ValueError('cumulative 150-image budget exceeded')
  s=requests.Session();s.headers['User-Agent']='Mozilla/5.0'
  for i in range(count,count+a.n):
   time.sleep(3.1)
   event={'kind':a.kind,'n':i,'at':time.time()}
   try:
    r=s.get(url,timeout=(10,45),allow_redirects=False)
    if r.status_code!=200:raise RuntimeError('http-'+str(r.status_code))
    im=Image.open(io.BytesIO(r.content))
    if im.size!={'hrmut':(60,20),'mhigr':(100,30)}[a.kind] or im.format!='GIF':raise RuntimeError('image-shape-changed')
    path=a.folder/f'{a.kind}_{i:03}.gif';path.write_bytes(r.content);event.update(status='ok',path=str(path),sha256=hashlib.sha256(r.content).hexdigest())
   except Exception:
    event['status']='fetch-failed'
    with ledger.open('a') as f:f.write(json.dumps(event)+'\n')
    raise SystemExit('source stopped; no retry')
   with ledger.open('a') as f:f.write(json.dumps(event)+'\n')

if __name__=='__main__':main()
