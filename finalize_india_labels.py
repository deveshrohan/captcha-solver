"""Adjudicate two BLIND visual passes and pin disjoint train/validation/test.

No model prediction is consulted when writing labels. Nulls are unreadable and
are dropped before splitting; split seed is fixed before any evaluation.
"""
import argparse
import json
from pathlib import Path
import random

def main(kind):
    one=json.loads(Path(kind+'_pass1.json').read_text());two=json.loads(Path(kind+'_pass2.json').read_text())
    adjudication=Path(kind+'_adjudication.json')
    decisions=json.loads(adjudication.read_text()) if adjudication.exists() else {}
    if one.keys()!=two.keys():raise ValueError('each image needs both passes')
    labels={};dropped=[]
    for path,value in one.items():
        if path in decisions:final=decisions[path]['final']
        elif value==two[path]:final=value
        else:raise ValueError('re-view disagreement: '+path)
        if final is None:dropped.append(path)
        elif len(final)!=6:raise ValueError('length: '+path)
        else:labels[path]=final
    if len(labels)<50:raise ValueError('need >=50 readable images, including >=30 held out')
    paths=sorted(labels);random.Random(20261023).shuffle(paths)
    split_path=Path(kind+'_split.json')
    if split_path.exists():
        old=json.loads(split_path.read_text())
        if not set(old['test']+old['validation']) <= set(labels):
            raise ValueError('pinned eval image removed; explicit new evaluation required')
        test=old['test']; validation=old['validation']
        train=[p for p in paths if p not in set(test+validation)]
    else:
        test=paths[:30];validation=paths[30:40];train=paths[40:]
    split={'seed':20261023,'test':test,'validation':validation,'train':train,'dropped':dropped,'rule':'two independent visual passes; disagreements re-viewed; unreadable dropped; no model labels'}
    Path(kind+'_labels.json').write_text(json.dumps(labels,indent=1,sort_keys=True)+'\n')
    Path(kind+'_split.json').write_text(json.dumps(split,indent=1)+'\n')
    print(kind,'labelled',len(labels),'dropped',len(dropped),'train',len(split['train']),'val',len(split['validation']),'test',len(split['test']))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('kind',choices=['hrmut','mhigr']);main(p.parse_args().kind)
