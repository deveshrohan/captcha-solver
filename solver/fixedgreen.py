"""Small fixed-font green GIF readers: glyph cover, never stored answers.

Libraries contain individual glyph pixels from the TRAIN split only. A dynamic
program covers all visible ink with exactly six glyphs. Unexplained ink, an
unknown glyph or an ambiguous cover lowers confidence instead of being hidden.
"""
from functools import lru_cache
import io
import json
from pathlib import Path
import numpy as np
from PIL import Image

LENGTH = 6
SIZES = {'hrmut':(60,20), 'mhigr':(100,30)}


def open_image(image):
    if isinstance(image,Image.Image): return image
    if isinstance(image,(bytes,bytearray)): return Image.open(io.BytesIO(image))
    return Image.open(image)


def strength(image,kind):
    im=open_image(image).convert('RGB')
    if im.size != SIZES[kind]:
        raise ValueError('captcha image shape changed')
    a=np.asarray(im).astype(np.float32)
    if kind=='hrmut':
        out=(a[:,:,0]>=128).astype(np.float32)
    else:
        # Font ink has red 0..51, green <=170; background noise has
        # red >=102 and green >=213. Preserve font antialiasing only.
        out=np.clip((153-a[:,:,0])/153,0,1)
        out[(a[:,:,1]>190)|(a[:,:,0]>=102)]=0
    return np.rint(out*255).astype(np.uint8)


def trim(a):
    xs=np.where(a.any(0))[0]
    return a[:,xs[0]:xs[-1]+1] if len(xs) else a[:,:0]


@lru_cache(maxsize=2)
def library(kind):
    data=json.loads((Path(__file__).parent/(kind+'_glyphs.json')).read_text())
    if data.get('_meta',{}).get('built_from')!='train-split':
        raise ValueError('reader library must be built from the train split')
    return {c:[np.array(g,np.uint8) for g in gs] for c,gs in data['glyphs'].items()}


def cover(a, glyphs, label=None):
    """Best and second-best DISTINCT text covers; optionally force a train label."""
    a=trim(a).astype(np.float32)/255.
    if not a.size: return [], []
    h,w=a.shape
    # Group glyph variants by width, vectorising all distances per position.
    by_width={}
    for char, variants in glyphs.items():
        for g in variants:
            if g.shape[0] != h: continue
            by_width.setdefault(g.shape[1],[]).append((char,g.astype(np.float32)/255.))
    stacks={wid:(np.stack([g for c,g in items]),[c for c,g in items]) for wid,items in by_width.items()}
    active=a.any(0)
    transitions={}
    for x in range(w):
        if not active[x]: continue
        opts={}
        for wid,(templates,chars) in stacks.items():
            end=x+wid
            if end>w: continue
            crop=a[:,x:end]
            # Error relative to glyph ink, rather than area: blank surroundings
            # cannot inflate confidence for narrow letters or a changed font.
            errs=np.abs(templates-crop).sum((1,2))/np.maximum(templates.sum((1,2)),crop.sum()).clip(1)
            nxt=end
            while nxt<w and not active[nxt]: nxt+=1
            for char,err in zip(chars,errs):
                key=(char,nxt)
                if key not in opts or err<opts[key][0]:
                    opts[key]=(float(err),end)
        transitions[x]=[(char,nxt,err,end) for (char,nxt),(err,end) in opts.items()]
    # Two candidates per state suffice for the distinct-word margin; variant
    # segmentations of one text are deduplicated at each state.
    states={(0,''):(0.,[])}
    for i in range(LENGTH):
        new={}
        for (x,word),(score,cuts) in states.items():
            for char,nxt,err,end in transitions.get(x,[]):
                if label is not None and char!=label[i]: continue
                key=(nxt,word+char)
                value=(score+err,cuts+[(x,end)])
                if key not in new or value[0]<new[key][0]: new[key]=value
        grouped={}
        for (x,word),value in new.items(): grouped.setdefault(x,[]).append((word,value))
        states={}
        for x,items in grouped.items():
            for word,value in sorted(items,key=lambda t:t[1][0])[:2]:states[(x,word)]=value
    finals=sorted([(score,word,cuts) for (x,word),(score,cuts) in states.items() if x==w])
    return finals[:2],a


def solve_image(image,kind):
    a=strength(image,kind)
    candidates,_=cover(a,library(kind))
    if not candidates:return '',0.
    cost,word,cuts=candidates[0]
    fidelity=max(0.,1-cost/LENGTH)
    margin=candidates[1][0]-cost if len(candidates)>1 else 1.
    # Confidence measures cover fidelity and the distinct-word margin. The
    # retry threshold was selected on validation; final test images stay held out.
    conf=min(fidelity, max(0.,margin*20))
    if kind == 'hrmut':
        # Uppercase I and lowercase l are identical solid bars in this font.
        # They were visually unreadable and excluded from labels. Do not snap
        # an unseen bar to a nearby 1/i template with a high word-average score.
        native=trim(a)
        for x,end in cuts:
            glyph=trim(native[:,x:end])
            occupied=np.where(glyph.any(1))[0] if glyph.size else []
            if glyph.shape[1] <= 2 and len(occupied) >= 7 and len(occupied) == occupied[-1]-occupied[0]+1:
                conf=0.
    return word,float(conf)
