"""Punjab RERA six-character captcha: local dark-ink isolation and Tesseract voting.

No remote services. Training/evaluation labels must come only from an explicit
portal acceptance, never from the OCR's own prediction.
"""
import io, re, subprocess
from collections import Counter
from PIL import Image

def solve(image):
    im=Image.open(io.BytesIO(image)).convert('RGB') if isinstance(image,bytes) else Image.open(image).convert('RGB')
    answers=[]
    for threshold,psm in [(60,8),(60,7),(90,8),(120,13)]:
        mask=im.convert('L').point(lambda p: 255 if p>threshold else 0)
        mask=mask.resize((im.width*5,im.height*5))
        buf=io.BytesIO(); mask.save(buf,format='PNG')
        p=subprocess.run(['tesseract','stdin','stdout','--psm',str(psm),'-c','tessedit_char_whitelist=0123456789ABCDEF'],input=buf.getvalue(),stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,timeout=20)
        v=re.sub(r'\s','',p.stdout.decode())
        if re.fullmatch(r'[0-9A-F]{6}',v):answers.append(v)
    if not answers:return '',0.0
    value,count=Counter(answers).most_common(1)[0]
    return value,count/4
