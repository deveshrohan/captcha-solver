"""HRERA six-character monochrome NIC captcha (130x40).

Imports existing glyph templates; adds only site-server-accepted TRAIN glyphs.
Whitespace segmentation and cropped bitmap distances. Tesseract bootstraps unseen
characters; portal acceptance, never OCR confidence, supplies gold labels.
"""
import io,json,subprocess,tempfile
from pathlib import Path
import numpy as np
from PIL import Image
from .fixedgreen import library
CHARSET='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
LENGTH=6

def ink(image):
 im=image if isinstance(image,Image.Image) else Image.open(io.BytesIO(image) if isinstance(image,(bytes,bytearray)) else image)
 if im.size!=(130,40):raise ValueError('HRERA captcha shape changed')
 a=np.array(im.convert('L'))
 return (a<80).astype(np.uint8)

def trim(a):
 ys,xs=np.where(a)
 return a[ys.min():ys.max()+1,xs.min():xs.max()+1] if len(xs) else a[:0,:0]

def segments(image):
 a=ink(image);xs=np.where(a.any(0))[0]
 if not len(xs):return []
 runs=[];start=xs[0];last=start
 for x in xs[1:]:
  if x-last>2:runs.append(trim(a[:,start:last+1]));start=x
  last=x
 runs.append(trim(a[:,start:last+1]))
 return runs

def glyphs():
 data={c:[trim((g>128).astype(np.uint8)) for g in gs] for c,gs in library('hrmut').items()}
 ref=Path(__file__).with_name('hrrera_reference_glyphs.json')
 if ref.exists():
  for c,gs in json.loads(ref.read_text())['glyphs'].items():data.setdefault(c,[]).extend(trim(np.array(g,np.uint8)) for g in gs)
 path=Path(__file__).with_name('hrrera_glyphs.json')
 if path.exists():
  for c,gs in json.loads(path.read_text()).get('glyphs',{}).items():data.setdefault(c,[]).extend(np.array(g,np.uint8) for g in gs)
 return data

def solve_image(image):
 segs=segments(image)
 if len(segs)!=6:return '',0.
 lib=glyphs();word='';confidence=1.
 for a in segs:
  opts=[]
  for char,gs in lib.items():
   errs=[float(np.abs(a.astype(float)-g.astype(float)).sum())/max(a.sum(),g.sum(),1) for g in gs if g.shape==a.shape]
   if errs:opts.append((min(errs),char))
  opts.sort()
  if not opts:
   confidence=0.;word='';break
  word+=opts[0][1];confidence=min(confidence,1-opts[0][0])
  if len(opts)>1 and opts[1][0]==opts[0][0]:confidence=0.
 if confidence>=.95:return word,confidence
 # Tesseract sees enlarged, widely spaced glyphs on pure white.
 canvas=Image.new('L',(6*35,35),255)
 for i,a in enumerate(segs):canvas.paste(Image.fromarray((1-a)*255).resize((a.shape[1]*2,a.shape[0]*2)),(i*35+5,5))
 # Pass PNG through stdin; no filesystem writes from this reader.
 stream=io.BytesIO();canvas.resize((630,105)).save(stream,format='PNG')
 p=subprocess.run(['tesseract','stdin','stdout','--psm','7','-c','tessedit_char_whitelist='+CHARSET],input=stream.getvalue(),capture_output=True)
 pred=''.join(c for c in p.stdout.decode().strip() if c in CHARSET)
 return (pred,.5) if len(pred)==6 else (word,max(0,confidence))

def add_accepted(image,label,split='train'):
 if len(label)!=6:return
 segs=segments(image)
 if len(segs)!=6:return
 if split!='train':return
 path=Path(__file__).with_name('hrrera_glyphs.json')
 data=json.loads(path.read_text()) if path.exists() else {'_meta':{'built_from':'train-split','label_source':'HRERA server accepted'},'glyphs':{}}
 for c,a in zip(label,segs):
  gs=data['glyphs'].setdefault(c,[]);g=a.tolist()
  if g not in gs:gs.append(g)
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data));tmp.replace(path)
