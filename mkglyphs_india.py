"""Build fixed-font readers from hand labels; held-out images are never opened.

python mkglyphs_india.py hrmut --font /path/to/Arial-Bold.ttf
Only individual train glyph pixels ship, following kaveri/udyam/ngt.
"""
import argparse
import json
from pathlib import Path
from PIL import ImageFont
import numpy as np
from solver.fixedgreen import strength, trim, cover, library


def initial_cuts(a,label,font,scale):
    # GDI fractional advances are approximated by an oversampled font. The
    # boundary cost then finds low-ink columns near those advance positions.
    a=trim(a); width=a.shape[1]
    adv=np.array([font.getlength(c)/scale for c in label],float)
    ideal=np.r_[0,np.cumsum(adv)]/adv.sum()*width
    states={0:(0.,[0])}
    for i in range(1,len(label)):
        new={}
        for cut in range(max(i,int(ideal[i]-3)),min(width-(len(label)-i),int(ideal[i]+3))+1):
            border=float(a[:,cut-1:cut+1].sum())/255.
            for prev,(cost,cuts) in states.items():
                if not 1<=cut-prev<=18:continue
                score=cost+border*.15+(cut-ideal[i])**2*.3
                if cut not in new or score<new[cut][0]:new[cut]=(score,cuts+[cut])
        states=new
    if not states:raise ValueError('unable to segment training image')
    _,cuts=min(states.values(),key=lambda x:x[0])
    return a, list(zip(cuts,cuts[1:]+[width]))


def build(kind,font_path):
    labels=json.loads(Path(kind+'_labels.json').read_text())
    split=json.loads(Path(kind+'_split.json').read_text())
    # No test paths can enter the extraction loop, including by accidental
    # overlap in a malformed split.
    train=split['train']; test=set(split['test'])
    if set(train)&test:raise ValueError('overlapping split')
    font=ImageFont.truetype(font_path,32 if kind=='hrmut' else 60)
    scale=3
    glyphs={};samples=[]
    for p in train:
        label=labels[p]
        a,cuts=initial_cuts(strength(p,kind),label,font,scale)
        samples.append((a,label,cuts))
        for char,(x,y) in zip(label,cuts):
            g=trim(a[:,x:y])
            if g.size:glyphs.setdefault(char,[]).append(g)
    # Keep variants, deduplicating identical glyph pixels. Training-only forced
    # covers refine the boundary estimates without consulting eval images.
    for iteration in range(2):
        revised={}
        for a,label,old in samples:
            candidates,_=cover(a,glyphs,label=label)
            cuts=candidates[0][2] if candidates else old
            for char,(x,y) in zip(label,cuts):
                g=trim(a[:,x:y])
                if g.size:revised.setdefault(char,[]).append(g)
        glyphs={c:list({(g.shape,g.tobytes()):g for g in gs}.values()) for c,gs in revised.items()}
    out={'_meta':{'built_from':'train-split','train_images':len(train),'held_out':len(test),'method':'fixed-font glyph cover','label_passes':2},'glyphs':{c:[g.tolist() for g in gs] for c,gs in glyphs.items()}}
    Path('solver/'+kind+'_glyphs.json').write_text(json.dumps(out,separators=(',',':'))+'\n')
    print(kind,'train',len(train),'test',len(test),'classes',len(glyphs),'variants',sum(map(len,glyphs.values())))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('kind',choices=['hrmut','mhigr']);p.add_argument('--font',required=True);a=p.parse_args();build(a.kind,a.font)
