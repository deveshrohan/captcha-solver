"""Build an adjudication page for every MCA image where the model disagrees
with the stored label.

Self-contained (no scratch helpers). Writes mca_adjudicate.html, which embeds
the images as data URIs so it opens standalone.

    python3 mkadj_mca.py
"""
import base64
import io
import json
import os

import numpy as np
import torch
from PIL import Image

from solver import mca as M

OUT = "mca_adjudicate.html"
CROP = (56, 22, 176, 62)          # same text window solver/mca.py reads


def b64(img, scale, nearest=False):
    img = img.resize((img.width * scale, img.height * scale),
                     Image.NEAREST if nearest else Image.LANCZOS)
    b = io.BytesIO()
    img.save(b, "PNG")
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def views(path):
    rgb = Image.open(path).convert("RGB")
    l, t, r, bm = CROP
    ink = M.text_mask(rgb)[t:bm, l:r]
    clean = Image.fromarray(((~ink) * 255).astype(np.uint8))     # black on white
    return b64(rgb.crop((40, 14, 190, 66)), 5), b64(clean, 7, nearest=True)


def main():
    labels = json.load(open("mca_labels.json"))
    sp = json.load(open("mca_split.json")) if os.path.exists("mca_split.json") else {"test": [], "dev": []}
    test, dev = set(sp["test"]), set(sp["dev"])

    items = [(p, l) for p, l in sorted(labels.items()) if os.path.exists(p)]
    model = M.McaCRNN()
    model.load_state_dict(torch.load("solver/mca_model.pt", map_location="cpu"))
    model.eval()
    X = np.stack([M.load_real(p) for p, _ in items])
    preds, confs = M.predict(model, X, "cpu")

    cases = [(p, l, pr, c) for (p, l), pr, c in zip(items, preds, confs) if pr != l]
    cases.sort(key=lambda x: x[3])

    cards = []
    for p, mine, pred, c in cases:
        diff = [i for i, (a, b) in enumerate(zip(mine, pred)) if a != b]
        raw, clean = views(p)

        def spell(s):
            return "".join(f'<b class="x">{ch}</b>' if i in diff else f"<b>{ch}</b>"
                           for i, ch in enumerate(s))

        which = "test" if p in test else "dev" if p in dev else "train"
        bars = any({mine[i], pred[i]} <= set("lI1") for i in diff)
        cards.append(f"""
<section>
  <header><code class="id">{os.path.basename(p)}</code>
    <span class="tag">{which}</span>
    {'<span class="bar">l / I / 1 bar — I could not tell either</span>' if bars else ''}
  </header>
  <div class="imgs">
    <figure><figcaption>as served (5&times;)</figcaption><img src="{raw}"></figure>
    <figure><figcaption>ink only, noise removed (7&times;)</figcaption><img src="{clean}"></figure>
  </div>
  <div class="guesses">
    <div><span>my label</span><div class="digits">{spell(mine)}</div></div>
    <div><span>model says</span><div class="digits">{spell(pred)}</div></div>
    <div><span>confidence</span><div class="digits conf">{c:.2f}</div></div>
  </div>
</section>""")

    html = f"""<title>MCA captcha — {len(cases)} disagreements</title>
<style>
 :root {{ --bg:#fff; --fg:#1a1a1a; --mut:#6b7280; --line:#e5e7eb; --card:#fafafa; --hi:#b91c1c; --tag:#3730a3; }}
 @media (prefers-color-scheme: dark) {{ :root {{ --bg:#0f1115; --fg:#e8eaed; --mut:#9aa0a6; --line:#2a2f3a; --card:#161922; --hi:#f87171; --tag:#a5b4fc; }} }}
 :root[data-theme="dark"] {{ --bg:#0f1115; --fg:#e8eaed; --mut:#9aa0a6; --line:#2a2f3a; --card:#161922; --hi:#f87171; --tag:#a5b4fc; }}
 :root[data-theme="light"] {{ --bg:#fff; --fg:#1a1a1a; --mut:#6b7280; --line:#e5e7eb; --card:#fafafa; --hi:#b91c1c; --tag:#3730a3; }}
 body {{ background:var(--bg); color:var(--fg); margin:0 auto; padding:2rem 1.25rem 4rem; max-width:62rem;
        font:16px/1.6 ui-sans-serif,-apple-system,system-ui,sans-serif; }}
 h1 {{ font-size:1.5rem; margin:0 0 .35rem; letter-spacing:-.01em; }}
 .lede {{ color:var(--mut); margin:0 0 2rem; max-width:48rem; }}
 section {{ border:1px solid var(--line); border-radius:12px; background:var(--card); padding:1rem 1.15rem 1.25rem; margin-bottom:1.4rem; }}
 header {{ display:flex; align-items:center; gap:.7rem; margin-bottom:.8rem; flex-wrap:wrap; }}
 .id {{ font:600 .95rem ui-monospace,SFMono-Regular,Menlo,monospace; }}
 .tag {{ font-size:.7rem; color:var(--tag); border:1px solid currentColor; border-radius:999px; padding:.1rem .5rem; text-transform:uppercase; letter-spacing:.05em; }}
 .bar {{ font-size:.72rem; color:var(--hi); border:1px solid currentColor; border-radius:999px; padding:.1rem .55rem; }}
 .imgs {{ display:flex; flex-wrap:wrap; gap:1.1rem; overflow-x:auto; }}
 figure {{ margin:0; }} figcaption {{ font-size:.72rem; color:var(--mut); margin-bottom:.3rem; }}
 img {{ max-width:100%; display:block; border-radius:6px; }}
 .guesses {{ display:flex; gap:2rem; flex-wrap:wrap; margin-top:1rem; padding-top:.9rem; border-top:1px dashed var(--line); }}
 .guesses span {{ display:block; font-size:.72rem; color:var(--mut); }}
 .digits {{ font:600 1.35rem ui-monospace,SFMono-Regular,Menlo,monospace; letter-spacing:.14em; }}
 .digits .x {{ color:var(--hi); }} .conf {{ font-weight:400; color:var(--mut); }}
</style>
<h1>MCA &mdash; the {len(cases)} captchas my label and the model disagree on</h1>
<p class="lede">Red characters are the positions in dispute. The right-hand image is
ink only, with the grey noise lines stripped &mdash; that is exactly what the model sees.
Sorted lowest-confidence first (most likely to be my mistake).
<br><br><strong>Case matters</strong> &mdash; the portal is case-sensitive, so
<code>s</code> vs <code>S</code> is a real difference. Cases flagged
<em>l / I / 1 bar</em> are ones I genuinely could not call; if you can't either,
say &ldquo;unknown&rdquo; and I'll drop them from the label set rather than train on a guess.
<br><br>Reply with the true 6 characters per id (or &ldquo;mine&rdquo; / &ldquo;model&rdquo;).</p>
{''.join(cards)}
"""
    open(OUT, "w").write(html)
    print("%d disagreements -> %s (%d KB)" % (len(cases), OUT, os.path.getsize(OUT) // 1024))


if __name__ == "__main__":
    main()
