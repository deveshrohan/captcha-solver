"""Held-out exact accuracy and real-image replay of <=3 fresh captcha policy.

The final test split is NEVER opened by mkglyphs_india.py. Sequential groups of
three here use a confidence gate, not an oracle that picks a correct answer.
Returned answers are compared to the label of the most recently fetched image.
"""
import argparse
import hashlib
import json
from pathlib import Path
from solver.api import solve_bytes,solve_with_retry
from solver.fixedgreen import library


def evaluate(kind,part='test'):
    split=json.loads(Path(kind+'_split.json').read_text());labels=json.loads(Path(kind+'_labels.json').read_text())
    paths=split[part]
    if part=='test' and len(paths)<30:raise ValueError('need at least 30 held-out images')
    artifact=Path('solver/'+kind+'_glyphs.json')
    meta=json.loads(artifact.read_text())['_meta']
    if meta['built_from']!='train-split':raise ValueError('not held out')
    rows=[]
    for path in paths:
        prediction,confidence,route=solve_bytes(path)
        if route!=kind:raise ValueError('wrong route')
        rows.append(dict(path=path,label=labels[path],prediction=prediction,confidence=confidence,correct=prediction==labels[path]))
    groups=[]
    for i in range(0,len(paths)-2,3):
        group=paths[i:i+3];used=[]
        def fetch():
            p=group[len(used)];used.append(p);return Path(p).read_bytes()
        threshold=.97 if kind=='hrmut' else .90
        t,c,k,tries=solve_with_retry(fetch,kind=kind,max_tries=3,min_conf=threshold)
        accepted=c>=threshold
        groups.append(dict(paths=group,used=len(used),accepted=accepted,correct=accepted and t==labels[used[-1]],confidence=c))
    threshold=.97 if kind=='hrmut' else .9
    accepted=[r for r in rows if r['confidence']>=threshold]
    return dict(kind=kind,split=part,artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),n=len(rows),exact=sum(r['correct'] for r in rows),single_shot_exact=sum(r['correct'] for r in rows)/len(rows),accepted=len(accepted),accepted_correct=sum(r['correct'] for r in accepted),retry_groups=len(groups),within_three_exact=sum(r['correct'] for r in groups),within_three_rate=sum(r['correct'] for r in groups)/len(groups),mean_fresh_fetches=sum(r['used'] for r in groups)/len(groups),threshold=threshold,rows=rows,groups=groups)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('kind',choices=['hrmut','mhigr']);p.add_argument('--split',default='test',choices=['validation','test']);p.add_argument('--output',type=Path);a=p.parse_args();r=evaluate(a.kind,a.split)
    if a.output:a.output.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps({k:v for k,v in r.items() if k not in ('rows','groups')}))
