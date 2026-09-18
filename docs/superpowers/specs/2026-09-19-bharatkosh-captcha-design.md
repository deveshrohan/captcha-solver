# Bharatkosh captcha reader — design

**Target** `https://bharatkosh.gov.in/NTRPHome/GenerateCaptcha` — the Non-Tax
Receipt Portal's captcha.

**Date** 2026-09-19

**Summary** The Bharatkosh image is the hardest single image in this repo and is
*not* invertible: every glyph gets its own font, size, rotation and gradient
colour, neighbours overlap, and two opaque bars erase 6 of the 40 rows. But the
endpoint hands over something no other portal here does — `?New=0` **re-renders
the same answer with a fresh random draw**. So the unit of reading is not an
image but a *group* of K independent renders of one text. The reader is a
CRNN+CTC over a bar-masked ink map (the ITAT architecture), and its answer is a
vote across renders. The same property gives a label-free accuracy signal
(cross-render agreement) and makes pseudo-labelling sound, which matters because
hand-labelling these is genuinely unreliable.

---

## 1. What the artifact is

Measured over 200 harvested images (40 groups × 5 renders).

| property | measurement |
|---|---|
| size / mode | `150×40` RGBA, alpha always 255 — 200/200 |
| generator fingerprint | chunks `IHDR sRGB gAMA pHYs IDAT IEND`, zlib `78 5e` — GDI+ (`System.Drawing`), consistent with the ASP.NET stack. Not libgd, not Java ImageIO |
| served as | `Content-Type: image/gif` — wrong; the body is PNG. Sniff the magic |
| text | **6 characters** in all 16 groups read so far; mixed case + digits |
| glyphs | each drawn independently: own typeface (sans, serif and rounded faces all appear inside one image), own size, own rotation (visibly up to ~±30°), own **two-colour gradient fill**, antialiased; neighbours overlap |
| bars | **two opaque horizontal bars at rows 14–16 and 24–26, x = 1..149, in 200/200 images.** One flat pastel colour each (2 distinct colours per image, 200/200), 447 px each. Drawn **last** — every bar row is 149/149 bar colour, so no ink or speck survives under them |
| speckle | isolated single pastel pixels, 186–245 per image (mean 214); min channel ≥ 120 in the sample measured |
| everything else non-white | 768–1087 px per image — the ink |
| colours per image | ~500 (gradients × antialiasing), so there is no ink *colour* to key on |

### Why this cannot be a cover

Kaveri, Udyam and NGT are invertible because a glyph is one of N fixed bitmaps.
Here the same character in the same group renders as visibly different shapes
(`A` reads as `A` in three renders and close to `4` in two; group 1). Random
typeface × size × rotation is a continuous family, so there is no library to
match against. ITAT reached the same conclusion from rotation alone.

### What *is* fixed, and is exploited

The bars. They are the only destructive noise, and their position is a
constant, not a random variable. So:

* the synthetic generator paints them at exactly rows 14–16 / 24–26;
* preprocessing **zeroes those six rows** in real and synthetic input alike, so
  the model sees one consistent "blind band" rather than 200 different bar
  colours. No train/test skew, and nothing to learn about bars at all.

The speckle is removed exactly where it is isolated (a non-white pixel with no
4-neighbour), and reproduced in the generator for the specks that touch ink.

## 2. Endpoint behaviour (measured by ablation)

| request | result |
|---|---|
| no UA / empty UA / `curl/8.x` / `python-requests/2.31`, no cookies | 200 PNG — **no request gate** |
| `?New=1` or no param | draws a **new** text into the session |
| `?New=0` on an existing session | **re-renders the same text**: fresh fonts, colours, rotations, specks |
| `?New=0` with no session | 500 (nothing to re-render) |
| 200 requests at 2 s spacing | zero blocks. The trip point was not searched for |

One jar, five fetches (`New=1`, `New=0`×3, no param) read `g2D4aK` four times
and then a different text. All 16 groups read so far are internally consistent.

Stack: ASP.NET MVC, `ASP.NET_SessionId` (secure, HttpOnly, SameSite=Lax) set on
first fetch and stable on the jar.

**Inferred, not measured:** that re-rendering leaves the *submittable* answer
unchanged. It follows from the text staying constant, but only a form
submission proves it, and this repo does not submit forms. If it turned out
false the reader still works per image; only the voting wrapper would go.

## 3. Architecture

```
K renders of one answer  ──►  per render: RGB → ink map → CRNN → CTC log-probs
                              └────────────────────────────────────────────┐
                         sum log-probs of each render's beam candidates  ◄─┘
                              → best 6-char string + agreement-based confidence
```

**`solver/bharatkosh.py`** — the ITAT layout: `CHARSET`, `load_real`,
`_to_input`, `make_input` (generator), `BharatkoshCRNN`, `predict`, plus
`vote(list_of_log_probs)`.

* **Ink map.** `d = 1 − min(r,g,b)/255` (white → 0; a saturated glyph colour →
  high; a pastel speck → ≤ ~0.5), isolated pixels zeroed, bar rows zeroed.
  Single channel, 150×40 upscaled to 64 px tall (the Securimage lesson:
  resolution gates fine distinctions, and rotation makes them finer).
* **Generator.** Per glyph: typeface from a shortlist of Windows core faces
  (Arial, Verdana, Tahoma, Times New Roman, Georgia, Trebuchet, Comic Sans,
  Courier New — all present in `/System/Library/Fonts/Supplemental`, loaded by
  path, never copied into the repo), size, rotation, two-stop gradient fill,
  advance with overlap; then specks, then the two bars. **Every range is
  measured from the real corpus before training, means and spreads both** — the
  MCA lesson. The typeface set is a shortlist, not an identification; it is
  pinned as far as glyph-IoU allows and randomised over the remainder (ITAT).
* **Model.** ITAT's CRNN+CTC, beam decode with `force_len=6`.
* **Voting.** Per render keep the top-B beam strings with log-probs; score a
  candidate by the sum over renders (floor for renders that did not propose
  it); answer = argmax. Summing log-probs rather than majority-voting strings
  lets three renders that are each unsure about a *different* glyph still
  produce the right word.

### Charset and case

62-class, case-sensitive, to start. Two things are decided by data, not now:

1. **Exclusions.** Run the EPFO census on the labelled set; drop classes whose
   absence is significant (`0/O/o`, `1/l/I` are the candidates). 96 characters
   read so far contain no `0`, `1` or `5` — too few to conclude anything.
2. **Case.** With per-glyph random size, relative height — the cue that
   separates `c/C`, `s/S`, `v/V`, `x/X` on MCA — is mostly destroyed *within one
   render*. Across K renders it partly returns. Whether the portal validates
   case is unknown and will not be tested by posting. So `eval` reports **both**
   case-sensitive and case-folded exact, `solve` returns the case-sensitive
   read, and case-homoglyph positions are exposed the way `AMBIGUOUS` is for MCA
   so the scraper can decide.

## 4. Labels — the actual bottleneck

Reading these by eye is unreliable even with five renders side by side: of 16
groups, 14 read confidently and 2 did not (`H`/`N`, `7`/`Z`/`T`, `W`/`N` under
rotation), and case is a guess for `v x c s`. So:

* **Group labels, not image labels.** `bharatkosh_labels.json` maps
  `b0007 → "UXD8FK"`; all K renders inherit it. One read yields K training
  images, and a glyph cut by a bar in one render is whole in the next.
* Groups that do not read confidently go to `bharatkosh_ambiguous.json` and are
  excluded from train *and* eval, never guessed.
* **Pseudo-labels from agreement.** After the synthetic-only model exists: a
  group whose K per-render reads agree unanimously is accepted as a pseudo-label
  for fine-tuning. Renders are independent draws of font/rotation/occlusion, so
  unanimous agreement on a wrong string is far less likely than one wrong read —
  this is what `New=0` buys. Pseudo-labelled groups are **never** used for eval.
* Fine-tune uses ITAT's per-epoch affine jitter, not identical repeats.

## 5. Eval honesty — two legs again

* **Leg 1, label-free:** over a large unlabelled harvest, the rate at which a
  group's renders agree with the group vote (per-render consistency), and the
  unanimous-group rate. Needs no labels, so it scales to thousands and doubles
  as the drift signal for `selfimprove.py`.
* **Leg 2, labelled:** held-out **groups** pinned by group id in
  `bharatkosh_split.json` — split by group, never by image, or renders of a
  test answer leak into training. Hand-read before the model existed. Reported
  as single-render exact, K=3 and K=5 voted exact, each case-sensitive and
  case-folded.

Leg 1 cannot catch a systematic confusion every render shares (rotated `N`→`H`
in all five); leg 2 is small. Same complementary pair as NGT's cover-rate +
labelled score.

## 6. Harvesting

`download_bharatkosh.py GROUPS RENDERS OUTDIR` (written during this analysis).
Fresh jar per group, `New=1` then `New=0`×(K−1), partial groups discarded,
silent back-off on any non-PNG. Files `b0000_r0.png …`. Target for the build:
300 groups × 5 (≈50 min at 2 s).

## 7. Tests

`tests/test_bharatkosh.py`: bar rows are zero after `_to_input` for real and
synthetic; isolated-speck removal leaves ink connected components intact;
generator output is 150×40 with bars at the measured rows; `vote` recovers the
word when each render is wrong at a different position, and is order-invariant;
`solve_bytes` routes 150×40 to `bharatkosh`; a non-150×40 image is rejected.

## 8. Wiring

* `solver/api.py`: `_SIZES[(150, 40)] = "bharatkosh"` (no collision — ITAT is
  150×**42**), `_load_bharatkosh`, `solve_bharatkosh(image)` for one render,
  `warmup(bharatkosh=…)`.
* New: `solve_bharatkosh_group(images)` → voted read. And a fetch helper
  contract for `solve_with_retry`: the caller supplies `fetch(new: bool)`; the
  solver asks for re-renders until the vote margin clears `min_conf` or K hits a
  cap (default 5), and only then asks for a new text.
* `.gitignore`: `bharatkosh_raw/` (done), `!solver/bharatkosh_model.pt`.
* `train_bharatkosh.py`, `finetune_bharatkosh.py`, `eval_bharatkosh.py`,
  `selfimprove.py` kind, README section.

**Done when:** tests pass; eval prints both legs on the pinned held-out groups;
README documents the measured numbers, whatever they are.

## 9. Expected result

No promise of a number. Single-render exact will likely sit well below the
other readers (ITAT, a strictly easier image, is the comparison). The claim this
design makes is narrower and testable: **voted accuracy rises steeply with K**,
because render errors are independent. If it does not — if errors are
correlated across renders — that falsifies the central assumption and the
fallback is more real labels, not more votes.
