"""Offline provenance, routing, drift and session-budget guards for India GIFs."""
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from solver.api import solve_bytes,solve_with_retry,warmup
from solver.fixedgreen import library,SIZES

ROOT=Path(__file__).resolve().parents[1]


def synthetic(kind,word):
    w,h=SIZES[kind];ink=np.zeros((h,w),np.uint8);x=5
    for char in word:
        glyph=library(kind)[char][0];end=x+glyph.shape[1]
        ink[:,x:end]=glyph;x=end+1
    if kind=='hrmut':
        a=np.zeros((h,w,3),np.uint8);a[:,:,0]=ink;a[:,:,1]=128+(ink.astype(np.uint16)*127//255);a[:,:,2]=ink
    else:
        a=np.empty((h,w,3),np.uint8);a[:,:,0]=np.rint(153-ink.astype(float)*153/255);a[:,:,1]=np.where(ink>0,128,255);a[:,:,2]=np.where(ink>0,0,153)
    output=io.BytesIO();Image.fromarray(a).save(output,format='GIF');return output.getvalue()


class TestIndia(unittest.TestCase):
    def test_native_gif_auto_routes_and_covers_synthetic_glyph_compositions(self):
        for kind,word in [('hrmut','2H3A79'),('mhigr','ABC279')]:
            got,conf,route=solve_bytes(synthetic(kind,word))
            self.assertEqual((got,route),(word,kind))
            self.assertGreaterEqual(conf,0.)
            self.assertLessEqual(conf,1.)

    def test_blank_and_wrong_size_images_cannot_look_confident(self):
        for kind,color in [('hrmut',(0,128,0)),('mhigr',(153,255,153))]:
            got,conf,route=solve_bytes(Image.new('RGB',SIZES[kind],color))
            self.assertEqual((got,conf,route),('',0.,kind))
        with self.assertRaises(ValueError):solve_bytes(Image.new('RGB',(100,30)),kind='hrmut')

    def test_templates_never_contain_complete_answers_or_held_out_images(self):
        for kind in SIZES:
            split=json.loads((ROOT/(kind+'_split.json')).read_text())
            self.assertGreaterEqual(len(split['test']),30)
            self.assertFalse(set(split['test'])&set(split['train']))
            self.assertFalse(set(split['validation'])&set(split['train']))
            artifact=json.loads((ROOT/'solver'/(kind+'_glyphs.json')).read_text())
            self.assertEqual(artifact['_meta']['built_from'],'train-split')
            self.assertEqual(artifact['_meta']['train_images'],len(split['train']))
            self.assertTrue(all(len(char)==1 for char in artifact['glyphs']))

    def test_retry_uses_hr_threshold_and_only_current_session_answer(self):
        reads=[('first',.96,'hrmut'),('second',.969,'hrmut'),('third',.99,'hrmut')]
        fetches=[]
        def fetch():fetches.append(1);return b'new session code'
        with patch('solver.api.solve_bytes',side_effect=reads):
            result=solve_with_retry(fetch,kind='hrmut',max_tries=3)
        self.assertEqual(result,('third',.99,'hrmut',3));self.assertEqual(len(fetches),3)
        with patch('solver.api.solve_bytes',side_effect=[(str(i),.5,'hrmut') for i in range(3)]):
            self.assertEqual(solve_with_retry(lambda:b'new',kind='hrmut',max_tries=3),('2',.5,'hrmut',3))

    def test_warmup_new_kinds_needs_no_neural_weights(self):
        warmup(securimage=False,hrmut=True,mhigr=True)

if __name__=='__main__':unittest.main()


class TestCollectionBudget(unittest.TestCase):
    def test_downloader_respects_the_same_cumulative_ledger_as_c4(self):
        import os
        import tempfile
        import download_india
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'india_capture_ledger.jsonl').write_text('\n'.join(json.dumps({'kind':'hrmut','status':'ok'}) for _ in range(150))+'\n')
            with patch.object(download_india,'__file__',str(root/'download_india.py')), patch('sys.argv',['download_india.py','hrmut','1',str(root/'images')]), patch.dict(os.environ,{'HRMUT_CAPTCHA_URL':'https://authorized.example.test/captcha'}), patch.object(download_india.requests,'Session') as http:
                with self.assertRaisesRegex(ValueError,'150-image budget'):
                    download_india.main()
                http.assert_not_called()
