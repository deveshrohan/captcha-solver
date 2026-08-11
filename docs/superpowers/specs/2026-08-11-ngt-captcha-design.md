# NGT captcha reader — design

**Target** `https://www.greentribunal.gov.in/sites/all/modules/custom/case_status/captcha.php`
— the National Green Tribunal's case-status captcha.

**Date** 2026-08-11

**Summary** The NGT captcha is fully invertible. It is a fixed bitmap font on a
fixed pixel grid, in pure black, with noise that is painted *underneath* the
text and can therefore never damage it. Reading it is a dictionary lookup, not
an inference problem: exact-black mask → six fixed 9×12 cells → 36-key table.
No model, no renderer archaeology, no fine-tune. This is the third model-free
reader in the repo after Kaveri and Udyam, and the strongest of the three,
because its cover is pixel-exact rather than approximate.

---

## 1. What the artifact is

Measured over 60 harvested images.

| property | measurement |
|---|---|
| size / mode | `120×40`, palette PNG (`imagecreate`, not truecolor) |
| generator fingerprint | `pHYs` = 3780 ppm (96 DPI) — libgd, i.e. PHP-GD |
| text | 6 lowercase alphanumeric characters, full 36-class charset `0-9a-z` |
| ink colour | **exactly** `(0,0,0)` |
| every other non-white pixel | all three channels in `[150,255]`; min luminance 154 |
| noise | isolated single pixels, 41–49 visible per image |
| `PLTE` size | **52 entries in every image**, always `ffffff` then `000000` |
| text grid | origin `x=20`, pitch `9`, rows `13..24` |
| black pixels outside that grid | **0**, across all 60 images |
| glyph cells → distinct bitmaps | 360 cells → **exactly 36**, zero singletons |
| min pairwise Hamming between the 36 | **6** |

### The palette reconstructs the generator

A PNG palette entry survives even when the pixel that caused its allocation is
later overdrawn. `PLTE` is 52 entries in *every* image — white, black, and
exactly 50 noise colours — while only 41–49 noise pixels are ever visible. **The
missing ones are underneath the glyphs.** That fixes the layer order: noise
first, text last.

```php
$im    = imagecreate(120, 40);                  // palette image
imagecolorallocate($im, 255,255,255);           // PLTE[0] background
imagecolorallocate($im, 0,0,0);                 // PLTE[1] text
for ($i = 0; $i < 50; $i++)                     // PLTE[2..51] — always exactly 50
    imagesetpixel($im, rand(0,119), rand(0,39),
        imagecolorallocate($im, rand(150,255), rand(150,255), rand(150,255)));
imagestring($im, 5, 20, 10, $code, $black);     // built-in font 5 (gdFontLarge, 9x15)
imagepng($im);
```

Two consequences follow, and the whole design rests on them:

1. **The ink is never damaged.** Text is drawn last, so no noise pixel can punch
   a hole in a stroke. This is not an empirical hope — it is forced by the layer
   order, and the layer order is proved by the palette. Contrast ITAT, where
   opaque lines drawn *over* the text erase strokes and the renderer has to
   reproduce the damage.
2. **The ink mask is exact.** `pixel == (0,0,0)` is the entire preprocessing
   story. The noise floor is 150, a gulf of 154 levels, so no threshold needs
   tuning and there is no train/test skew to worry about.

Hence: 360 out of 360 glyph cells matched one of exactly 36 pixel-identical
bitmaps, with zero singletons. That is the measurement that says *lookup table*,
not *model*.

### Character separation

All 36 bitmaps are mutually distinct. The classic homoglyphs are wide apart —
`0`/`o` 32, `5`/`s` 35, `9`/`g` 53, `2`/`z` 20 — but three glyphs form a tight
cluster:

```
1 / l = 6      1 / i = 6      i / l = 6
```

Because matching is **exact**, this costs nothing: the three are still distinct
keys. It matters only if a cell is ever damaged, which the layer order forbids.
It does, however, dictate two decisions below (§4 confidence, §5 label check).

---

## 2. Endpoint behaviour (measured by ablation)

Per the repo's standing rule, no header recipe was reused; the gate was
re-measured from scratch. NGT's posture is a **fourth** distinct one.

**User-agent: an allowlist crossed with a denylist.** The UA must contain
`mozilla` *and* must not contain a tool token.

| UA | result |
|---|---|
| `Mozilla`, `Mozilla/4.0`, `mozilla/5.0`, `MyBot Mozilla/5.0` | 200, `image/png` |
| `curl/8.7.1`, `python-requests/2.31`, `Wget/1.21`, `Googlebot/2.1` | 403 |
| empty, `x` | 403 |
| `Mozilla/5.0 curl` | 403 |

**Rate limit: ~10 requests per window, and blocked requests extend the block.**
Harvesting at one request per 8s produced a strikingly regular pattern of
successes between 429s:

```
10, 10, 10, 10, 9, 10, ...
```

An initial burst of ~50 requests in 90s tripped a 429 that did **not** clear
across 7 minutes of backed-off polling; killing the poller and staying silent
for 5 minutes cleared it immediately. So the block window refreshes on every
request made during it. Backoff here is a throughput optimisation, not a
courtesy: polling a block is strictly slower than waiting it out.

A safe steady rate was bounded, not directly measured — 10 requests per ~120s
(one per 12s) sits below the observed trip point.

**Stack: Drupal 7.** Responses set `SSESS<hash>` (`secure`, `HttpOnly`,
`SameSite=Strict`), which persists across fetches on one jar. `captcha.php` is a
raw PHP file inside a custom module directory that bootstraps Drupal. The
idiomatic implementation stores the expected word in `$_SESSION`, whose natural
consequence is that only the most recently fetched image is submittable on a
given session — the Securimage / GST / ITAT pattern.

**That last point is inferred from cookie behaviour, not confirmed.** Confirming
it would mean posting to a live government form. As with Udyam, Kaveri and ITAT,
this design documents the *fetch* contract and deliberately does not document or
probe the *submit* contract.

---

## 3. Architecture

```
PNG ─► ink = (rgb == 0).all()  ─► 6 cells at x=20+9k, rows 13..25 ─► lookup ─► "kqip8x"
       exact; noise floor 150       grid exact: 0 stray px in 60 images    36 keys, no model
```

`solver/ngt.py`, following the Kaveri/Udyam module shape:

| symbol | purpose |
|---|---|
| `CHARSET` | `"0123456789abcdefghijklmnopqrstuvwxyz"` |
| `LENGTH`, `W`, `H` | `6`, `120`, `40` |
| `X0`, `PITCH`, `R0`, `R1` | `20`, `9`, `13`, `25` |
| `CELL_W`, `CELL_H` | `9`, `12` |
| `glyphs()` | lazy-load `solver/ngt_glyphs.json`, bit-packed as in `udyam._pack` |
| `ink_mask(image)` | `(rgb == 0).all(axis=2)` |
| `cells(mask)` | the six fixed slices |
| `solve_image(image)` | → `(text, confidence)` |
| `make_image(label, rng)` | synthetic: 50 noise pixels **then** glyphs |
| `random_label(rng)`, `predict(images)` | corpus generation and batch eval |

Each unit is independently testable: `ink_mask` needs no glyph table, `cells`
needs no images, `make_image` is the inverse of `solve_image` and the tests
exercise that round trip directly.

---

## 4. Routing, confidence, error handling

### 4.1 The 120×40 collision

`(120, 40)` is already the gstat numeric captcha's size — the first size
collision in the repo, so `solve_bytes` needs a content test rather than a name
lookup. `_SIZES` values become "a kind, or a callable returning one":

```python
def _route_120x40(pil):
    """gstat's ink is #333 and it never emits a pure-black pixel; NGT's text IS
    pure black."""
    return "ngt" if (np.asarray(pil) == 0).all(2).sum() >= 20 else "gstat"
```

The threshold is safe by construction, not merely by measurement:

- gstat's palette is `#333` ink, `#ccc` lines and white — it *cannot* produce a
  pure-black pixel. Measured: **0 black pixels across 1024 images** (1000
  synthetic `corpus/`, 24 real `real_test/`).
- NGT's thinnest possible word is six `r` (21 ink px each) = **126** black
  pixels. Measured range over 60 images: 159–228.

### 4.2 Confidence is a proof, not a softmax

Kaveri's contract, for the same reason Kaveri has it:

- **`1.0` iff** all six cells match a template **exactly** *and* no ink lies
  outside the grid. For this generator that is a proof the read is correct.
- **otherwise** `min(0.99, explained_ink / total_ink)`. The explicit cap matters:
  a single wrong glyph is only ~6 mismatched pixels against ~190 ink pixels, so
  the raw ratio would round to 0.97 and read as near-certain. Capping guarantees
  "inexact" is always distinguishable from "exact" no matter how small the
  discrepancy.

Matching is **exact or bust** — a cell that matches nothing reports its nearest
label but can never reach 1.0. This is precisely because `1`, `l` and `i` sit 6
pixels apart: a nearest-neighbour fallback that silently snapped a damaged cell
to the closest bar-glyph would flip `1`→`l` invisibly, and a 3-pixel error is
enough to do it.

The documented retry gate is therefore `min_conf=1.0`. Anything less means the
generator moved, which is exactly what `selfimprove check` should trip on.

### 4.3 Error handling

- wrong image size → does **not** raise; reads what it can (matching
  `test_wrong_size_does_not_raise` in the Kaveri and Udyam suites)
- no ink at all → `("", 0.0)`
- unknown cell → nearest label, confidence < 1.0, never a silent lie
- ink outside the grid → confidence < 1.0; the image is not what this reader
  models

`AMBIGUOUS` gets no `ngt` entry, with a comment explaining why: all 36 bitmaps
are distinct and the cover is exact, so no pair is a coin flip. The `1`/`l`/`i`
proximity is a robustness note, not an ambiguity.

---

## 5. Glyph library and label provenance

`mkglyphs_ngt.py` builds `solver/ngt_glyphs.json` from the corpus, asserting the
structural invariants and failing loudly if any breaks:

1. every cell in the corpus matches a cluster exactly — zero singletons;
2. exactly 36 clusters;
3. min pairwise Hamming ≥ 6 (a label typo that collapsed two classes would
   show up here);
4. all 36 charset members present.

Clustering means the cluster→character map is **36 hand decisions**, read once
off a montage, rather than ~480 hand-labelled glyph slots.

### The circularity risk, and the fix

That efficiency puts the entire correctness of the reader on 36 hand-read
labels — and the tightest cluster is exactly the one a human is most likely to
misread. If `1` and `l` were transposed, the same person hand-labelling the
eval set would most likely transpose them there too, and the held-out score
would come back a confident 100% for a reader that swaps them forever. The
error would be invisible to the evaluation designed to catch it.

**Fix:** `imagestring(..., 5, ...)` is libgd's `gdFontLarge`, whose 9×15 bitmaps
are public in `gdfontl.h`. The shipped bitmaps stay extraction-sourced, and the
canonical font is used **only as a test-time cross-check on the 36 labels**.
That is an external ground truth, and it is the only thing that breaks the
circularity.

### Eval honesty

Glyphs are derived from the **train split only**. `eval_ngt.py` scores the
held-out split against `ngt_labels.json`, with `ngt_split.json` seeded and
pinned. Deriving templates from every label and then scoring those same labels
would reproduce the inflated-number trap the README already documents for MCA
(96.5% over all labels vs 75.0% held-out), and would undo the honest-by-default
property established in `39f7b68`.

Labels: ~40 held-out images, hand-read independently.

---

## 6. Harvesting

`download_ngt.py`, encoding the three measured facts from §2:

- UA must contain `mozilla` and no tool token;
- pace at ~1 request per 12s, below the observed ~10-per-window trip point;
- on a 429, **go silent** and wait the block out rather than polling it, because
  polling refreshes it.

It reuses **one cookie jar** rather than ITAT's fresh-session-per-fetch. This is
Drupal with DB-backed sessions, so a new session per image would litter the
server's session table for no sampling benefit — the images are distinct
regardless, and nothing is ever submitted.

Target ~250 images: ~40 for held-out labels, the rest for the glyph library and
the corpus-wide invariant checks.

---

## 7. Tests

`tests/test_ngt.py`, pinning the claims the reader actually depends on:

**Charset and glyph table**
- 36 classes, length 6, all lowercase alnum
- all 36 bitmaps mutually distinct
- `1`, `l`, `i` specifically remain separated — regression guard on the tightest
  cluster
- the 36 labels match libgd's `gdFontLarge` (§5)

**Ink model**
- exact black is ink; a luminance-150 noise pixel is **not**
- a noise pixel adjacent to a stroke does not join the mask

**Layer order** (the property the cover rests on)
- `make_image` with all 50 noise pixels still round-trips exactly

**Routing**
- an NGT-like 120×40 routes to `ngt`; a gstat-like one (`#333` + `#ccc`) routes
  to `gstat`
- no image in the gstat corpus trips the black-pixel threshold

**Reading**
- round-trip over the whole charset
- `conf == 1.0` on a clean synthetic read
- `conf < 1.0` when one pixel is flipped, and when ink is added outside the grid
- wrong size does not raise
- held-out real accuracy — skipped when the gitignored corpus is absent

---

## 8. Wiring

- `solver/api.py` — `solve_ngt`; no `_load_ngt` / no module-level model cache,
  since there is no model; `warmup(ngt=True)` just parses the glyph table (cheap,
  and the flag exists so callers can enable every kind uniformly, as for Kaveri
  and Udyam); the `_SIZES` callable entry; an `AMBIGUOUS` no-entry comment
- `solve.py` — routing table docstring, noting the 120×40 content split
- `selfimprove.py` — `KINDS` entry; confidence collapses on a generator change,
  so drift detection is genuinely load-bearing here
- `.gitignore` — `ngt_raw/`
- `README.md` — table row, a new section, and the layout list

---

## 9. Expected result

100% exact on the held-out split, by construction rather than by training. This
would be the first reader in the repo whose accuracy claim is a *proof about the
generator* rather than a measurement of a model — closer to Kaveri's exact cover
than to any of the CRNNs, and stronger than Kaveri's because no sprite overlap
resolution is needed at all.

The honest caveats:

- **The submit contract is unverified**, as for Udyam, Kaveri and ITAT. This
  reads the image; whether the portal validates case, or how the answer is
  posted, is not established here.
- **The proof is about today's generator.** It rests on a fixed font, a fixed
  grid and the noise-under-text layer order. If NGT changes any of those the
  reader does not degrade gracefully — it drops to confidence < 1.0 immediately,
  which is the intended behaviour and the reason `selfimprove` is wired in.
- **The rate limit is an operational constraint**, not just a harvesting nuisance:
  a caller retrying against a fresh captcha is spending from a ~10-per-window
  budget. With `min_conf=1.0` and an exact cover, retries should be vanishingly
  rare — but a caller that ignores the gate and hammers will get 429s, and
  polling through them makes it worse.
