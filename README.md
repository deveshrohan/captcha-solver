# captcha-solver

Local, synthetic-trained captcha readers. `solve.py` auto-routes by image size
and runs on CPU with the included weights.

> Ships code, the synthetic data generators, and pretrained weights — no captcha
> images. Every *model* here is trained on synthetic data only. The two
> model-free readers (kaveri, udyam) are the exception worth naming: they ship a
> library of individual glyph bitmaps extracted from real captchas, since a
> template cover is only as good as the templates. Use only against services you
> are authorized to automate. Endpoint URLs are placeholders; set
> `SECURIMAGE_URL` to your own. MIT licensed.

| type | size | style | model | held-out real accuracy |
|---|---|---|---|---|
| numeric | 120×40 | 6 digits, thin `#ccc` lines | threshold + segment + per-digit CNN | 100% (48/48 digits) |
| securimage | 215×80 | 6 lowercase-alnum, wavy warp | CRNN+CTC, faithful renderer, length-6 beam | 98.8% exact (81/82) |
| gst | 182×50 | 6 digits, hatch grid + fisheye | CRNN+CTC on RGB, SimpleCaptcha port | 99.0% exact (97/98), 99.83% digits |
| mca | 200×80 | 6 mixed-case alnum, line noise | exact ink mask → CRNN+CTC + real fine-tune | 75.0% exact single-shot, ~97% within 3 fetches |
| epfo | 150×50 | 5 alnum, gradient background | background subtraction → sprite-exact synth → CRNN+CTC | **100% exact (44/44)** |
| kaveri | 200×60 | 6 uppercase-alnum, lines under text | exact ink mask → sprite cover, **no model** | **210/210 exact and unique** |
| udyam | 225×80 | 6 uppercase-alnum, lines over text | luminance ink mask → template cover, **no model** | **36/36 exact**, margin > 0 on 2520/2520 glyphs |
| itat | 150×42 | 6 mixed-case alnum, blue lines over text | darkness projection → CRNN+CTC, case-insensitive + augmented real fine-tune | 36.7% exact / 85.0% char (11/30), ~84% within 4 fetches |

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

---

## GST portal captcha (182×50)

Six black digits over a grey gradient, crossed by a black hatch grid and one red
curve. Unlike the two above this is **not** PHP-GD: there is no `pHYs` chunk and
the zlib header is `78 da` (`CINFO=7`, best-compression), the signature of Java's
`ImageIO` writer.

### Identifying the generator

The image is produced by **SimpleCaptcha** (Java, `nl.captcha`). Three
independent measurements pin it down:

1. **Grid pitch.** `FishEyeGimpyRenderer` computes
   `hspace = height/(height/7+1)` and `vspace = width/(width/7+1)`. For 182×50
   that is exactly `6` and `6` — the measured pitch — and the lines land on
   `x = 6,12,…,180`, `y = 6,12,…,48`, exactly as observed.
2. **A circular warp.** The same renderer then applies a fisheye inside a circle
   of radius `ranInt(width/4, width/3)` = 45..60 centred on (91, 25). Real
   captchas have *perfectly straight* grid lines outside that circle and warped
   ones inside — which is why whole-column blackness holds at `x ≤ 36` and
   `x ≥ 144` but breaks in between.
3. **The background.** `GradiatedBackgroundProducer` paints a Java
   `GradientPaint` from (0,0) `DARK_GRAY` to (w,h) `WHITE`, i.e.
   `64 + 191·(x·w + y·h)/(w²+h²)`. Predicted vs measured agrees to a mean
   absolute error of ~0.9 grey levels *at every radius* — which also proves the
   background is composited **behind** the already-fisheyed ink layer rather than
   being warped with it, matching `Captcha.Builder.build()`.

Layer order, therefore: transparent layer ← text ← red curve ← grid + fisheye;
then the gradient behind; then a 1px border on top. `solver/gst.py` ports this,
including Java's `(int)` truncation inside the fisheye resample.

Two details are *not* stock SimpleCaptcha and are fitted from real samples:

- the red curve spans the full width with **no antialiasing** (real images
  contain exactly one non-grey colour, `#ff0000`), whereas stock
  `CurvedLineNoiseProducer` spans `0.1w..0.9w` with antialiasing on;
- the font is a Verdana-lineage bold face, not the stock Arial/Courier.

### The font is the one open variable

The server is Linux, so the Java font name resolves through fontconfig to
whatever is installed. Matching real glyph shapes (IoU over the pristine strip
left of the fisheye circle, scoring only pixels the grid never touches) ranks
**DejaVu Sans Condensed Bold** first (0.775), then DejaVu Sans Bold / Tahoma
Bold / Verdana Bold (~0.74). Rather than bet on one face, training randomises
over that shortlist.

The *geometry*, by contrast, is measured and pinned: after undoing the fisheye
and masking the grid, real ink spans `x 13..125` (112px for 6 digits) at height
28 with baseline 37. Each face is auto-scaled to reproduce a mean digit ink width
of 18.67px, and the pen advances by each glyph's **ink** width (not its advance
width) — which is what makes real digits sit tightly packed.

### Endpoint behaviour (measured)

- **`?rnd=` is cache-busting, not a seed.** The same `rnd` requested twice
  returns two *different* images, and omitting it entirely still returns a valid
  captcha. Nothing about the image depends on the value.
- **The captcha is session-bound via `CaptchaCookie`.** Every response sets
  `CaptchaCookie=<32 hex>` (Domain=`.gst.gov.in`, `Secure`, `HttpOnly`) and the
  value rotates on *every* GET — three successive calls through one jar gave
  three distinct cookies. (That rotation is measured; that the server keys the
  expected answer off the newest cookie is the natural reading of it, but
  confirming it needs a real form submission, which I did not do.) Treat only
  the last image fetched as valid, and fetch and post through one cookie jar.
- `TS0134d082` is the F5 BIG-IP ASM (WAF) cookie. That WAF rejects a *spoofed*
  browser UA (a bare `Mozilla/5.0` without matching client hints returns a
  200-with-HTML "Request Rejected" page); plain curl's default UA is accepted.

```bash
python3 download_gst.py 400 gst_raw  # plain curl; a spoofed browser UA trips the WAF
python3 train_gst.py 55              # synthetic-only -> solver/gst_model.pt
python3 finetune_gst.py 12           # self-training on unlabeled reals
python3 eval_gst.py                  # score against gst_labels.json
```

### Results (98 real captchas)

| model | exact | per-digit |
|---|---|---|
| synthetic-only (no real image used at all) | 94/98 (95.9%) | 584/588 (99.32%) |
| + self-training on unlabeled reals, **held-out test half** | **49/49 (100%)** | 294/294 (100%) |
| + self-training, all 98 | 97/98 (99.0%) | 587/588 (99.83%) |

**The labels were the bottleneck, not the model.** These 98 were first labeled by
reading the fisheye-inverted images by eye, and 10 were flagged at the time as
not confidently readable. Scoring against those first-pass labels put the model
at 85.7% exact. Every disagreement was then re-examined against the image at 8×
zoom and adjudicated by the repo owner — and **10 of the 11 were labeling
mistakes, not model errors**. Re-scoring against the corrected labels moves the
model from 85.7% to 99.0%.

The single remaining error is `g0004`: truth `222646`, read as `222546`, one
digit at the fisheye's point of maximum magnification. Notably the adjudicated
truth there matched *neither* the original label (`227646`) nor the model
(`222546`) — evidence the adjudication was an independent read rather than a
rubber-stamp of the prediction.

Two caveats worth keeping: 49/49 on the test half is a small sample (95% Wilson
CI `[92.7, 100]`), so read it as "no observed errors", not as proof of
perfection; and the adjudication UI displayed the model's guess alongside the
image, which is not a blind protocol even though the `g0004` result argues
against anchoring. The epoch was selected on the dev half using the
pre-correction labels.

The one residual error sits at the fisheye's centre, where the image is
magnified ~4× and stroke detail is destroyed — the same region that produced
most of the *labeling* difficulty. The distortion is invertible (`undistort` in
`solver/gst.py`), which is what made the images readable enough to label and
adjudicate at all; inference still reads the raw image, because the per-image
radius estimator is biased by ~3px (Java truncates where the inverse rounds).

---

## MCA portal captcha (200×80)

Six mixed-case alphanumeric characters over a dense field of straight grey noise
lines. This one is by far the easiest, for a structural reason worth stating:
the image contains **exactly three colours** — background 230, noise 150, ink 0 —
with no antialiasing, and the noise is drawn *under* the text. So an exact
equality test on pure black recovers every glyph whole and unbroken, and the
noise cannot affect recognition at all. The generator does not model it.

What remains is genuinely variable: each character gets its own size and
weight/slant, with cap heights spanning 12–21px *within a single captcha*, so the
reader must be scale-invariant. `solver/mca.py` renders text only — random
DejaVu Sans face per character, per-image base size with per-character jitter —
onto the measured geometry (baseline y≈50, start x≈66) and reads it with a
CRNN+CTC over 62 classes.

### Endpoint behaviour: the Akamai cookies are a red herring

The endpoint returns `multipart/mixed` with both a PNG and a WAV (the
accessibility audio captcha); `download_mca.py` splits them. It sits behind
Akamai Bot Manager, and the obvious assumption — that the `ak_bmsc` / `bm_sv`
tokens are what let you through — is **wrong**. Ablating one cookie at a time:

| request | result |
|---|---|
| all cookies | 200 |
| drop `ak_bmsc` | 200 |
| drop `bm_sv` | 200 |
| **no cookies at all** | **200** |

What actually gates it is the *header fingerprint*. Dropping one header at a
time from a complete Chrome-XHR set, only two are individually fatal:
`user-agent` and `referer` (and the referer must be an mca.gov.in URL —
example.com is rejected). But those two alone still fail: Akamai scores the set
as a whole. Adding headers back one at a time, it took 8 of them (`accept`,
`accept-language`, the three `sec-ch-ua*` client hints and `sec-fetch-dest`)
before it returned 200 — that count is specific to the order I added them in, so
read it as "most of a real Chrome XHR's headers", not as a minimal basis.

So **no browser session is needed** and `MCA_COOKIE` is optional. Sporadic 403
bursts are rate limiting, not cookie expiry — the downloader backs off.

For reference, the tokens themselves are `~`-separated and opaque:
`ak_bmsc` = 32-hex GUID ~ a 30-char all-zero flag field ~ a 400-char base64
payload that decodes to a 300-byte AES blob; `bm_sv` = 32-hex GUID ~ 192-byte
blob ~ a counter. Both blobs start `0x6000…`, and `bm_sv`'s embeds `17af3b17`
— the serving edge node, `23.175.59.23`, the same address Akamai prints in its
`Reference #18.17af3b17.…` denial pages.

```bash
python3 download_mca.py 400 mca_raw  # no cookie needed
python3 train_mca.py 45              # synthetic-only -> solver/mca_model.pt
python3 finetune_mca.py 25           # fine-tune on hand-labeled reals
python3 eval_mca.py                  # scores the held-out split (--all for every label)
```

### The font could not be identified — so train on real labels instead

Segmenting glyphs out of the labeled reals and matching them against 30+
candidate faces gives a best mean IoU of only 0.67 (Arial Bold) with no face
clearly winning, and rendering aliased rather than antialiased barely moves it
(0.664 → 0.670). Side by side the real strokes are consistently *thinner* than
the bold templates. The face remains unidentified.

Since MCA ink is exactly separable, the captchas are trivial to read by eye and
therefore cheap to label in bulk. 184 were hand-labeled and split
110 train / 30 dev / 44 test (`finetune_mca.py`); the test split is never
trained on and never used to pick the epoch.

### Results (44 held-out real captchas)

| model | exact | per-character |
|---|---|---|
| synthetic-only | 15/44 (34.1%) | 214/264 (81.1%) |
| + fine-tuned on 110 real labels | 28/44 (63.6%) | 247/264 (93.6%) |
| + fine-tuned on 230 real labels | 33/44 (75.0%) | 250/264 (94.7%) |
| **+ fine-tuned on 321 real labels** | **33/44 (75.0%)** | **250/264 (94.7%)** |

Labelling is **saturated**: going from 230 to 321 training images (395 of the 397
corpus captchas are now labelled) moved neither metric by a single character.
The first doubling, 110 → 230, was worth +7 points; the next 40% was worth zero.
Whatever is left is not a data-volume problem.

The portal validates **case-sensitively**, so the strict column is the only one
that counts. Doubling the labelled training set moved exact accuracy 63.6% →
70.5%; the CIs (`[48.9, 76.2]` vs `[55.8, 81.8]`) still overlap at n=44, but the
direction is consistent and per-character improved too. The 44-image test set is
pinned by filename in `mca_split.json` rather than re-derived from a seed, so
adding training labels cannot silently reshuffle it.

### Skip the unreadable ones instead of guessing

`l` and `I` are both plain vertical bars at the same height, so a read
containing one is close to a coin flip — and since the portal is
case-sensitive, guessing costs a failed submission. Measured on held-out reals:

| | share of captchas | exact accuracy |
|---|---|---|
| read contains `l` or `I` | 16% | **28.6%** |
| read contains neither | 84% | **83.8%** |

So the right move is not a better model — more labels demonstrably do not help — but a cheaper decision: **throw that
captcha away and fetch another.** `solve_with_retry(..., avoid_ambiguous=True)`
(the default) does this — it refetches whenever the read contains a character
from `api.AMBIGUOUS[kind]`. Cost is ~1.4 fetches per solve:

| fetches | cumulative success |
|---|---|
| 1 | 68.2% |
| 2 | 89.9% |
| 3 | **96.8%** |
| 4 | 99.0% |

That turns a 75% single-shot reader into a ~97% pipeline without touching the
model. Captchas are free to re-request, so the only cost is latency.

Of the residual character errors, roughly a third are these bar confusions, and
the rest split between case slips (`s`→`S`, `W`→`w`, `V`→`v` — height-resolvable,
so more labels keep eating them) and genuine shape errors. ~94 corpus images
remain unlabelled; `mca_ambiguous.json` lists the labels that still rest on an
unresolved `l`/`I` call, so training can exclude them rather than learn a guess.

Most of the residual error is case and homoglyph collisions: `I`↔`l`, `s`↔`S`,
`c`↔`C`, `O`↔`0`, `V`↔`v`.

### Case is *partly* recoverable — via relative height

An earlier version of this note claimed case was unresolvable, on the grounds
that `s` and `S` are the same shape at different scales. That is true of an
isolated glyph and false in context: every character sits on one shared baseline
at one shared font size, so a lowercase `s` reaches only x-height while a
capital `S` reaches cap-height. Measured over the labelled glyphs, relative
height (glyph height ÷ tallest glyph in the same image) separates:

| pair | lowercase | uppercase | 1-D accuracy |
|---|---|---|---|
| o/O | 0.737 | 0.990 | 100% |
| c/C | 0.726 | 0.971 | 100% |
| x/X | 0.748 | 0.973 | 100% |
| u/U | 0.751 | 0.953 | 100% |
| s/S | 0.760 | 0.943 | 95% |
| k/K, l/I, 0/O, 1/l | ~1.0 | ~1.0 | 38–61% (chance) |

The split is exactly what typography predicts: the cue exists for x-height
letters and is genuinely absent for full-height ones (`k`/`K` and `l`/`I` are
both ascender-height; `0`/`O` and `1`/`l` are both full-height). So `I`↔`l` and
`O`↔`0` really are irreducible; the x-height cases are not.

**The generator was destroying this cue.** Within a real image the height spread
among full-height glyphs is only `(max-min)/mean = 0.044` — font size is
essentially constant per image, and what looks like per-character size
randomisation is mostly cap-height vs x-height. The generator was jittering size
±3pt, giving a spread of 0.208 — **4.8× too wide** — so in training a lowercase
`o` could out-tower an uppercase `O` and the model correctly learned to ignore
height. `make_mask` now scales by `base × (1 + N(0, 0.015))`, matching reality.

Retraining on the corrected generator cut pure-case errors from **7 to 4**, as
predicted, but genuine shape errors drifted 5 → 7, so the totals were flat:
28/44 vs 29/44 exact, 247/264 vs 246/264 characters. At n=44 those CIs overlap
almost entirely (`[48.9, 76.2]` vs `[51.1, 78.1]`) — a one-image swing is noise.
Separating a 5-point difference here would need ~700 labelled test images.

**The reliable lever from here is more labels, not more renderer archaeology:**
110 training images is thin for 62 classes, and the genuine-shape errors
(`C`→`O`, `a`→`9`, `r`→`m`) are the kind that more real data fixes. Also worth
confirming whether the portal validates case-insensitively — if it does, the
case column stops mattering and the effective rate is the case-insensitive row.

---

## EPFO portal captcha (150×50)

Five characters on a grey gradient with no noise lines at all. Two measurements
turn this from a modelling problem into a reconstruction problem.

### The background is a constant

The image is entirely greyscale and the background is a **fixed vertical
gradient that is byte-identical in every captcha** — 255 at y=0, falling to 128
at y=25, back to 249 at y=49, constant along each row (verified against 40
samples: zero variation). So ink extraction is exact subtraction rather than a
threshold heuristic.

### The font is a sprite sheet, not TrueType

The first attempt approximated the font with DejaVu and scored **27.3% exact**
despite 92.4% on its own synthetic validation. The confusions gave it away —
`M→I` ×9, `W→I` ×8, `V→I` ×3, wide glyphs collapsing to a narrow bar.

Measuring per-character ink widths explained why: they are quantised with *zero*
variance — `M`=8, `1`=8, `L`=8, `T`=8, `W`=10 — at every threshold tried. No
scalable font does that. Normalising to true alpha (`pixel = bg·(1−alpha)`)
showed all instances of each character are **pixel-identical, max deviation
0.004**. The renderer blits fixed bitmaps.

So `solver/epfo.py` does not approximate the font; it **replays the sprites
extracted from labelled images** (`solver/epfo_glyphs.json`, 32 glyphs) at the
measured layout (`solver/epfo_layout.json`). Every metric now matches:

| | real | synthetic |
|---|---|---|
| glyph height | 12.08 | 12.10 |
| glyph width | 8.13 | 8.17 |
| stroke density | 4.46 | 4.41 |
| 5-segment rate | 1.00 | 1.00 |

### The charset is 32, not 36

`I`, `N` and `O` never occur in 280 labelled characters while the other 33 are
uniform (χ² p = 0.52); P(three given characters absent by chance from a 36-char
set) is 5×10⁻¹¹. A further catch: the single labelled `0` had a descender tail
and was byte-identical to `Q` — a mislabel. The real charset is
`1-9` + `A-Z` minus `I N O` — **32 characters, with `0` excluded too.**

That matters beyond correctness: excluding `0`/`O` and `1`/`I` removes exactly
the homoglyph pairs that cap the MCA reader at 75%. EPFO has no ambiguous pairs,
which is why it reaches 100%.

### Endpoint behaviour (measured)

The captcha URL carries an `_HDIV_STATE_` token. Replaying a copied URL without
the matching JSESSIONID 302s to `error.jsp`; but the token is **reusable within
its session** (three fetches on one token gave three different captchas). So
`download_epfo.py` loads the establishment-search page, scrapes the freshly
minted token, and re-mints on expiry.

```bash
python3 download_epfo.py 400 epfo_raw
python3 train_epfo.py 35             # synthetic-only -> solver/epfo_model.pt
python3 eval_epfo.py                 # score against the held-out split
```

### Results (44 held-out real captchas, trained on synthetic only)

| model | exact | per-character |
|---|---|---|
| DejaVu approximation, 36-char set | 12/44 (27.3%) | 79.6% |
| **extracted sprites, 32-char set** | **44/44 (100%)** | **220/220 (100%)** |

No real image was used for training — the 56 labels exist only to evaluate and
to extract the sprites. 44/44 is a small sample (95% CI `[92.0, 100]`), so read
it as "no observed errors", not proof of perfection.

---

## Udyam portal captcha (225×80)

`udyamregistration.gov.in/CaptchaControl.aspx` — 6 characters, navy on a pale
blue gradient under the tricolour. Like Kaveri it is read by covering the ink
with fixed glyph bitmaps, so there is no model, no training script and no `.pt`;
`solver/udyam_glyphs.json` is 25KB.

**The noise lines are drawn over the text, and it does not matter.** This is the
one structural difference from Kaveri, where the lines sit underneath and
`rgb == (0,0,0)` recovers every glyph whole. Here they are *alpha-blended*: a
line crossing ink keeps the ink's red and green and lifts only blue —
`(25,60,153)`, `(25,60,175)`, `(25,100,196)` — so the crossed pixels stay dark
while the same line over the pale background stays light. A luminance cut
recovers them, and glyphs stop fragmenting.

**Subpixel phase, not occlusion, is what varies.** Recovering the occluded
pixels moves the exact-duplicate rate only 61.8% → 64.7%. Matching with ±2px
alignment collapses 147 apparent bitmap variants into 34 — the full 33-character
alphabet, plus a second phase for `5`. Those two `5`s are kept apart rather than
merged: at IoU 0.828 they sit *above* the closest genuinely distinct pair (`E`
vs `F`, 0.794), so any threshold that merged them would risk merging those.

**Glyphs overlap, so the reader covers rather than segments.** A merged `WX`
spans 55 columns where `W` (35px) and `X` (27px) need 62 — they share 7. No
vertical cut yields both, and cutting to complete the `W` leaves an
X-minus-left-edge that is a pixel-perfect `K`. Scoring each piece on its own
merit therefore ranks the wrong reading joint-first; only the X's orphaned
lower-left stroke, unexplained by `W` and `K` together, distinguishes them. So
each glyph is anchored on the leftmost ink its predecessors left over — as the
Kaveri reader anchors on a pixel — and whole readings are scored by cover.

| | |
|---|---|
| alphabet | 33 chars — no `0`, `I` or `O`, so no homoglyph pairs and no `AMBIGUOUS` entry |
| labelled | **36/36 exact**, of which **26/26** have touching glyphs |
| substitution margin | positive on **2520/2520** glyphs (min 0.17, median 0.70) |
| confidence | real 0.869…0.99 vs scaled-text 0.744, rotated 0.684, black 0.120 |

The headline is the **substitution margin**, not the accuracy: for every glyph,
swapping in the runner-up character explains the pixels measurably worse. That
is a property of the images, not an estimate from a sample. It is the graded
counterpart of Kaveri's exact-cover uniqueness — the text here is antialiased,
so no cover is pixel-exact and confidence never legitimately reaches 1.0.

What the margin cannot catch is the one human step: the 34 bitmaps were mapped
to characters by eye. A mislabelled class corrupts every read containing it
while leaving every margin healthy, which is what the labels and
`eval_udyam.py --sprites` are for.

Two measurement traps, both of which produced wrong readers before being fixed:

* the border `(100,130,180)` and the green bar `(0,128,0)` are both dark enough
  to pass any ink test, so the reader works on a fixed band, not the raw image;
* a 2px speck of line residue 20 rows below the text is too narrow to disturb
  segmentation but stretches a glyph's bounding box from 25 rows to 46, which
  silently corrupted the `U`, `H` and `F` templates built from those crops.

Endpoint: a user-agent **denylist** (`curl/8.x` and `python-requests` get a
`403 Forbidden - Access denied due to bot User-Agent`; an empty UA, `Mozilla/5.0`
and even `x` get 200) — the same shape as Kaveri, the opposite of GST. Unlike
Kaveri it is **session-bound**: a stock ASP.NET `CaptchaControl` holding the
answer in session state, so only the most recently fetched image is submittable,
as for Securimage. How the answer is submitted is deliberately not documented —
the captcha guards forms taking an Aadhaar or Udyam number, so confirming the
validate contract would mean posting identifiers to a live government portal.

---

## ITAT portal captcha (150×42)

`itat.gov.in/captcha/show` — six mixed-case alphanumeric characters, near-black
and antialiased, each at its own **random rotation**, drawn on white under a
dense field of **light-blue straight lines and colour speckle**. This is the
Income Tax Appellate Tribunal's case-status / e-filing captcha.

### Identifying the generator

Two fingerprints pin it to **CodeIgniter's `create_captcha()` GD helper**:

1. every response sets the CodeIgniter session cookies `ci_session`,
   `csrf_cookie_name` and `uid`; and
2. the PNG carries libgd's `pHYs` chunk = 3780 ppm (96 DPI) — the same PHP-GD
   signature as the numeric captcha at the top of this file, and the marker that
   it is drawn by GD rather than Java's `ImageIO` (GST) or .NET (Udyam).

The response's `Content-Type` is even `text/html`: the controller just echoes
the image bytes, so validity is the PNG magic, not the header. The helper is
heavily customised from stock — truecolor (3000+ colours per image, so
`imagecreatetruecolor`, not the palette default), black antialiased text, no
border, and light-blue **line** noise rather than the stock pink spiral — but
the create_captcha DNA is unmistakable: one TTF face for the whole word, each
glyph placed with a random angle.

### The ink is colour-separable, but the noise fragments it

The text is near-black `(0,0,0)` while every noise colour is bright — the lines
cluster around `(151,206,252)` and the speckle is scattered light dots. So a
**darkness projection** collapses the two:

```
d = 1 - max(r, g, b) / 255      # ink -> ~1, white -> 0, light-blue line -> ~0.01
```

That is the whole preprocessing story, and it is applied identically to real and
synthetic images, so there is no train/test skew. What it does **not** do is
repair the text: the noise is drawn **over** the glyphs (632 blue pixels sit
flanked by ink on both sides across 20 images — the signature of a line cutting
through a stroke), and where an opaque line crosses, the darkness there drops to
~0, punching a hole. This is Udyam's layering, but Udyam's lines are
alpha-blended so a luminance cut *recovers* the crossed pixel; ITAT's are opaque
enough to erase it. Rather than try to fill the holes, the generator
**reproduces** them — it draws the same blue-lines-over-text before projecting —
so the model trains on the same broken strokes it will read. That is the
Securimage lesson (model the noise that touches the ink), reached through a
colour projection that discards the noise that doesn't.

### Why a model, and why case-insensitive

Random per-character rotation rules out the template-cover approach that reads
Kaveri, Udyam and EPFO: those need pixel-identical fixed-pose glyphs, and a
random angle destroys that. So ITAT is a whole-image **CRNN + CTC** over the
darkness map, like Securimage and MCA.

The captcha renders **mixed case** — lowercase `p`, `u`, `w` with descenders and
x-heights appear next to capitals. Case is read **case-insensitively**: every
glyph folds to one uppercase class. Two reasons. First, for full-height letters
(`K`/`k`, `S`/`S` at cap size) case is not recoverable from the glyph at all —
the wall that caps the case-*sensitive* MCA reader at 75%. Second, the submit
contract could not be verified (that would mean posting to a live ITAT form), so
whether the portal even validates case is unknown, and folding is the choice
that fails safe if it does not. The generator therefore renders each letter as
lower- **or** upper-case at random while labelling the one folded class, so the
model learns both glyph shapes map to it.

Folding case also removes the case-homoglyphs (`c`/`C`, `s`/`S`, …). The census
over the labelled corpus shows `0 1 I O` never occur, which removes the
digit/letter homoglyphs too. So — as for EPFO and Udyam — **no ambiguous pair
survives**, and there is no `AMBIGUOUS` entry to retry around.

### The renderer pins geometry, because the font is not identified

The real face is a **thin, narrow** sans: six glyphs span ~116px (aspect ~0.55,
i.e. condensed) at ~31px cap height with stroke/cap ~0.07. None of the bundled
faces is it, and — as with MCA — it could not be identified. So the generator
does what the GST reader does with its font: it **pins the geometry** rather than
trusting the face. Glyphs are laid out (DejaVu Sans Condensed, the narrowest
bundled face) with proportional advance and per-character rotation, then the
whole word's ink bbox is scaled to the measured real target and lightly eroded
toward the measured stroke width. The synthetic ink statistics then match the
real corpus — bbox 116×31, stroke ~2.2px — leaving one residual gap: the
letterform *weight* (DejaVu is denser than the real face). That gap is exactly
what real-label fine-tuning closes, the same division of labour as MCA.

### Endpoint behaviour (measured)

- **No user-agent gate at all.** `curl/8.x`, `python-requests`, an empty UA,
  `Mozilla/5.0` and even `x` every one returns 200 with a valid PNG. That is a
  *third* distinct anti-bot posture in this repo, and the reason the memory note
  says never to reuse a header recipe: Kaveri and Udyam run a UA **denylist**,
  MCA wants most of a real Chrome header set, GST rejects a spoofed browser UA —
  and ITAT checks the UA not at all.
- **Session-bound (inferred, not confirmed).** Within one cookie jar `ci_session`
  is minted on the first fetch and then persists, and this is a CodeIgniter app,
  whose idiomatic captcha stores the expected word in the session. The natural
  consequence is that only the most recently fetched image is submittable — the
  Securimage/Udyam pattern. That is read off the cookie behaviour, not a form
  submission, so `download_itat.py` fetches each captcha on a **fresh session**
  (no jar) to get independent samples, and a live caller should fetch and submit
  through one jar, treating only the last image as valid. How the answer is
  submitted is deliberately not documented, for the same reason as Udyam.

```bash
python3 download_itat.py 400 itat_raw  # no UA gate; fresh session per image
python3 train_itat.py 15               # synthetic-only -> solver/itat_model.pt
python3 finetune_itat.py 30            # augmented fine-tune on hand-labeled reals
python3 eval_itat.py                   # score the held-out split (--all for every label)
```

### Results (held-out real captchas, case-insensitive)

111 captchas were hand-labelled (adjudicated against the model's reads at 8×
zoom, since cold-reading this noise is error-prone) and pinned into a seeded
split — 61 train / 20 dev / 30 **test**. The test 30 are never trained on and
never used to pick the epoch.

| model | test exact | test char |
|---|---|---|
| synthetic-only (no real image used at all) | 0/30 | 20.6% |
| + real fine-tune on 61 labels | 5/30 (16.7%) | 76% |
| **+ affine augmentation of the reals** | **11/30 (36.7%)** | **85.0%** |

Two things drove the numbers, in order of size:

**The synthetic base barely transfers — 20.6% char — because the font is wrong.**
The real face is a thin sans that could not be identified, and DejaVu Condensed
(pinned to the right geometry) is still visibly heavier; the CRNN learns
DejaVu-specific features that do not fire on the real strokes (every glyph
collapses toward `L`/`T`/`F` on real input). This is a larger domain gap than
MCA's (whose synthetic already read reals at ~81% char), and it is why a real
fine-tune is not optional here.

**Augmentation, not more synthetic, was the lever.** Fine-tuning on 61 real
labels overfit hard — 100% on the train images, ~70% on held-out — because
oversampling 61 images verbatim just memorises pixels. Showing each one under a
small random affine jitter each epoch (`finetune_itat.py:augment`) turned 61
images into effectively many and lifted held-out exact from 16.7% to 36.7% and
char from 76% to 85%, with no new labels. The dev-exact rate moved 1/20 → 8/20 —
the memorisation breaking.

**Retry does the rest.** The captcha is length-6, homoglyph-free and free to
re-request, so a wrong read costs a fetch, not a failed submission. Submitting
every read on a fresh captcha each try (single-shot exact 36.7%):

| fetches | cumulative success |
|---|---|
| 1 | 36.7% |
| 2 | 60% |
| 3 | 75% |
| 4 | **84%** |
| 5 | 90% |

Confidence is only a weak gate (mean 0.938, yet 63% of single reads are
imperfect; at a 0.95 cut, 57% of reads pass at 53% exact) — the same "confidence
is not accuracy" caveat as GST — so the reliable knob is the retry count, not a
threshold.

**The ceiling here is labels, as it was for MCA.** 61 training images is thin for
31 classes, the residual errors are genuine shape/rotation confusions
(`D`↔`P`, `Y`↔`V`, `G`↔`6`, `8`↔`9`) that more real data eats, and the
hand-labels themselves carry read noise on this hard captcha, which caps the
measurable char rate below the model's true rate. The lever from here is more
labels, not more renderer archaeology.

### Caveat: case-sensitivity is unverified

The reader folds case because case is partly unreadable and the submit contract
is unverified. If a portal check ever shows ITAT validates case-sensitively, this
reader's folded output would need re-casing — which the glyph often cannot
support (see the MCA section on why case is only partly recoverable). Treat the
case-insensitive number as the honest ceiling for what the image supports, not as
a claim that a case-sensitive submission would match.

---

## Self-improvement loop

Captcha generators change without warning, and the failure is silent: the model
keeps returning confident nonsense and the scraper simply stops finding results.
`selfimprove.py` is built around that risk.

```bash
python3 selfimprove.py check                 # drift + regression, exit 1 on trouble
python3 selfimprove.py harvest --kind gst --n 200
python3 selfimprove.py adapt   --kind gst    # self-train on confident pseudo-labels
python3 selfimprove.py report
```

Three properties matter more than the automation itself:

**`adapt` cannot make things worse.** It self-trains on high-confidence pseudo-
labels but keeps the new weights *only* if the held-out labelled set does not
regress; otherwise it restores the backup and exits non-zero. Pseudo-labelling
drifts into reinforcing its own mistakes, so an ungated loop is worse than none.

**Confidence is never treated as accuracy.** GST reads above 0.99 confidence
were still only 88% correct. Every accuracy claim comes from human labels.

**Accuracy is measured on the held-out split only.** Scoring all labels would
include images the model was fine-tuned on and has memorised — MCA reads 96.5%
over all 395 labels versus 75.0% on its 44 held-out ones. A detector built on
the inflated number stays green while real accuracy falls.

`check` needs no labels at all to catch the common case: it watches served image
size, valid-length rate, charset validity, and confidence against the previous
run, all recorded in `selfimprove_state.json`.

---

## Programmatic use

`solver/api.py` is a CPU, in-process API — no network, only needs
torch/numpy/pillow (scipy is training-only). See `examples/ecourts_securimage.py`.

```python
from solver.api import warmup, solve_bytes, solve_with_retry

warmup(securimage=True, gst=True)            # load models once at startup
text, conf, kind = solve_bytes(image_bytes)  # ('6yyzch', 0.998, 'securimage')

def fetch():                                 # re-request on the SAME session
    return session.get(url, params={"_": time.time()}).content
code, conf, kind, tries = solve_with_retry(fetch, min_conf=0.90, kind="securimage")
```

Routing is by native image size (each captcha has a distinct one): 120×40 →
`gstat`, 150×42 → `itat`, 150×50 → `epfo`, 182×50 → `gst`, 200×60 → `kaveri`,
200×80 → `mca`, 215×80 → `securimage`, 225×80 → `udyam`.

Securimage is **session-bound**: each captcha request rotates the server-side
code, so only the most recently fetched captcha is valid to submit.
`solve_with_retry` returns the last read via the same `fetch` session — submit it
before fetching again. The GST captcha behaves the same way.

---

## Layout

- `solver/` — renderers, models, decode, pipeline, `api.py`
- `train_securimage.py`, `train_gst.py`, `train_mca.py`, `train_epfo.py`,
  `train_itat.py`, `train.py`, `gen_corpus.py`, `finetune_gst.py`,
  `finetune_mca.py`, `finetune_itat.py`
- `mkglyphs_udyam.py` — rebuild the Udyam class library from a corpus
  (Kaveri's equivalent lives in `solver.kaveri.extract_sprites`)
- `selfimprove.py` — drift detection and gated self-training
- `eval_securimage.py`, `eval_gst.py`, `eval_mca.py`, `eval_epfo.py`,
  `eval_kaveri.py`, `eval_udyam.py`, `eval_itat.py`
- `download_gst.py`, `download_mca.py`, `download_epfo.py`,
  `download_securimage.py`, `download_kaveri.py`, `download_udyam.py`,
  `download_itat.py`
- `tests/` — `python3 -m unittest discover -s tests -v`
- `examples/ecourts_securimage.py`
- `fonts/AHGBold.ttf` — Alte Haas Grotesk Bold (via the Securimage project)
- `fonts/DejaVu*.ttf` — DejaVu fonts (Bitstream Vera / Arev licence,
  `fonts/DejaVu-LICENSE`), used by the GST, MCA and ITAT renderers
