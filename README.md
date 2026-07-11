# captcha-solver

Local, synthetic-trained readers for two PHP-GD captcha styles. `solve.py`
auto-routes by image size and runs on CPU with the included weights.

> Ships code, the synthetic data generator, and pretrained weights — no captcha
> images (models are trained on synthetic data only). Use only against services
> you are authorized to automate. Endpoint URLs are placeholders; set
> `SECURIMAGE_URL` to your own. MIT licensed.

| type | size | style | model | held-out real accuracy |
|---|---|---|---|---|
| numeric | 120×40 | 6 digits, thin `#ccc` lines | threshold + segment + per-digit CNN | 100% (48/48 digits) |
| securimage | 215×80 | 6 lowercase-alnum, wavy warp | CRNN+CTC, faithful renderer, length-6 beam | 98.8% exact (81/82) |

```bash
pip install torch numpy pillow       # inference deps (scipy is training-only)
python solve.py path/to/captcha.png  # auto-routes by size
```

Programmatic use: `solver/api.py` — `solve_bytes`, `solve_with_retry`, `warmup`.

---

## Numeric captcha (120×40)

Dark `#333` digits with lighter `#ccc` diagonal lines on white — a PHP-GD captcha
(confirmed by the PNG fingerprint: `pHYs 3780` = libgd 96 DPI, zlib `CINFO=6` from
libpng). Since the digits are darker than the lines, one brightness threshold
removes the lines, reducing the task to reading 6 isolated digits.

```
image ─► threshold (#ccc out) ─► segment (6 cells, 28×28) ─► CNN ─► "233120"
```

- `solver/preprocess.py` — threshold at gray 128.
- `solver/segment.py` — vertical-projection split into 6 digit cells.
- `solver/synth.py` — synthetic generator (8 fonts, ±13° rotation, per-digit
  jitter, `#ccc` lines drawn over the glyphs). Trained on synthetic only.
- `solver/model.py` — 3-conv CNN → 10 classes.

```bash
python3 gen_corpus.py 1000 corpus    # synthetic corpus
python3 train.py                     # -> solver/model.pt
python3 solve.py path/to/img.png     # prints text + min per-digit confidence
```

Trained on 1000 synthetic images, the model reaches 100% exact / 100% digit
accuracy on held-out real samples. A change of font, distortion, or length would
need `solver/synth.py` updated to match, then a re-train.

---

## Securimage captcha (215×80)

[Securimage](https://github.com/dapphp/securimage) grayscale captcha: 6 lowercase
alphanumeric characters (`a-z`, `2-9`), a sinusoidal warp, and curvy noise lines.
Threshold + segment fails here (lines are as dark as the text and the warp moves
characters off fixed positions), so the reader is whole-image: **CRNN + CTC**.

### A faithful renderer is the key lever

The model trains on synthetic images only; each accuracy gain came from porting
Securimage's own GD algorithm more faithfully (`solver/securimage.py`):

1. **Font** — AHGBold (`fonts/AHGBold.ttf`), matched to the letterforms.
2. **Exact 3-pole radial warp** (`distortedCopy`), computed in output space and
   sampled from an upscaled text canvas — the biggest single gain.
3. **Upright, tightly-packed text** (`use_text_angles=false`, `dist=rand(-2,0)`).
4. **Measured colours/opacity** — ink flat gray 50; lines `#707070` at 0.8
   opacity. The 0.8 opacity keeps descenders readable *through* a crossing line
   instead of erased (real histogram spikes at 50 / 99 / 140 = text /
   line-over-text / line-over-background).
5. **Length-6 CTC beam decode** (`ctc_beam_decode`) — best exactly-6-char
   decoding; separates adjacent repeats via the blank.

Architecture notes: CRNN+CTC generalises where a flatten+fixed-head CNN only
memorises (the warp moves characters off fixed positions). Input `192×64`
collapses to `T=24`; the 64px height preserves vertically-distinguished detail
(descenders, `6`/`8`, `a`/`2`). CTC loss runs on CPU (unimplemented on Apple MPS).

```bash
python3 train_securimage.py 40       # synthetic-only -> solver/securimage_model.pt
python3 eval_securimage.py           # score against labeled real sets
```

### Results (held-out real, never trained on any real image)

| set | exact | note |
|---|---|---|
| real_test (12) | 12/12 | |
| real_eval (40) | 39/40 | random draw |
| fresh random 30 | 30/30 | labeled blind |
| **total (82)** | **81/82 (98.8%)** | single-shot |
| coverage (1300) | 1300/1300 length-6 | mean confidence 0.99 |

Residual ~1%: a noise line *fully* covering a distinguishing feature (`g`'s hook →
`9`, `y`'s descender → `v`) — partly inherent once the feature is erased. These
are the lowest-confidence reads; since a captcha is re-requestable and always
length 6, a low-confidence read can trigger a retry on a fresh image.

(Labeled real evaluation images are not included in this repo.)

### Programmatic use

`solver/api.py` is a CPU, in-process API — no network, only needs
torch/numpy/pillow (scipy is training-only). See `examples/ecourts_securimage.py`.

```python
from solver.api import warmup, solve_bytes, solve_with_retry

warmup()                                     # load the model once at startup
text, conf, kind = solve_bytes(image_bytes)  # ('6yyzch', 0.998, 'securimage')

def fetch():                                 # re-request on the SAME session
    return session.get(url, params={"_": time.time()}).content
code, conf, kind, tries = solve_with_retry(fetch, min_conf=0.90, kind="securimage")
```

Securimage is **session-bound**: each captcha request rotates the server-side
code, so only the most recently fetched captcha is valid to submit.
`solve_with_retry` returns the last read via the same `fetch` session — submit it
before fetching again.

---

## Layout

- `solver/` — renderers, models, decode, pipeline, `api.py`
- `train_securimage.py`, `train.py`, `gen_corpus.py`, `eval_securimage.py`
- `examples/ecourts_securimage.py`
- `fonts/AHGBold.ttf` — Alte Haas Grotesk Bold (via the Securimage project)
