# captcha_solver API reference

`captcha_solver` reads captchas from Indian government portals, offline and on
CPU. It identifies which portal a captcha comes from using only the image, reads
it, and can drive the whole fetch → read → submit → retry loop for you.

- [Install](#install)
- [Quick start](#quick-start)
- [Functions](#functions): [`identify`](#identifyimage---str) ·
  [`solve`](#solveimage-kindnone-min_confnone---result) ·
  [`solve_with_retry`](#solve_with_retryfetch-submitnone-kindnone-max_attempts5-min_confnone-max_renders5-raise_on_failuretrue---result) ·
  [`solve_url`](#solve_urlurl-sessionnone-submitnone-timeout30-kwargs---result) ·
  [`warmup`](#warmupkindsnone)
- [`Result`](#result)
- [Exceptions](#exceptions)
- [Supported captchas (`KINDS`)](#supported-captchas-kinds)
- [How retry decides](#how-retry-decides)
- [Recipes](#recipes)
- [CLI](#cli)
- [Threads, memory, latency](#threads-memory-latency)
- [Lower-level engine](#lower-level-engine)

---

## Install

```bash
pip install "git+https://github.com/deveshrohan/captcha-solver.git"           # latest main
pip install "git+https://github.com/deveshrohan/captcha-solver.git@india-readers"   # a branch
pip install /path/to/captcha-solver        # a local checkout
pip install "captcha-solver[http] @ git+https://github.com/deveshrohan/captcha-solver.git"
                                           # + requests, needed only for solve_url()
```

Not on PyPI; install from GitHub or a checkout. Requires Python ≥ 3.9, `numpy`, `pillow` and `torch` (CPU is enough). The
wheel bundles every model and template, about 160 MB. No network access is
needed at runtime.

```python
import captcha_solver as cs
cs.__version__          # '0.1.0'
```

## Quick start

```python
import captcha_solver as cs

# read one image
r = cs.solve(open("captcha.png", "rb").read())
print(r.text, r.kind, r.confidence, r.accepted)     # 078086 madhc 0.999 True

# or let the package fetch, read, submit and retry
r = cs.solve_with_retry(fetch=get_captcha, submit=post_form)
print(r.text, r.attempts)                           # answer the site accepted, fetches used
page = r.response                                   # what post_form returned
```

---

## Functions

### `identify(image) -> str`

Returns the captcha's kind (a key of [`KINDS`](#supported-captchas-kinds)),
worked out from the image alone.

- **image**: `bytes`, a file path (`str`), a binary file object, or a
  `PIL.Image.Image`.
- **Returns**: e.g. `"mca"`, `"gujhc"`.
- **Raises**: [`UnsupportedCaptcha`](#exceptions) if the image's size matches
  no supported captcha or the input is not an image.

It identifies by exact image size. The one size two portals share (120×40,
gstat vs NGT) is split by a content test: NGT's text is pure black, and gstat
never emits a pure-black pixel. It never guesses. An unknown size is an error,
because a guessed reader returns confident nonsense.

```python
cs.identify("madhc_0000.png")      # 'madhc'
cs.identify(b"<html>...")          # raises UnsupportedCaptcha: not an image
```

### `solve(image, kind=None, min_conf=None) -> Result`

Identifies (unless `kind` is given) and reads one captcha.

| parameter | type | meaning |
|---|---|---|
| `image` | bytes / path / file / PIL | the captcha |
| `kind` | `str` or `None` | skip identification and use this reader. Must be a key of `KINDS`, otherwise `ValueError` |
| `min_conf` | `float` or `None` | override the kind's gate. `hrmut` never goes below its 0.97 floor |

Returns a [`Result`](#result). `solve` never retries and never raises on a
bad read; it reports one. A read that fails a check comes back with
`accepted=False` and a `reason`.

```python
r = cs.solve(img)
if r.accepted:
    submit(r.text)
else:
    print("fetch another:", r.reason)    # e.g. 'low-confidence'
```

### `solve_with_retry(fetch, submit=None, kind=None, max_attempts=5, min_conf=None, max_renders=5, raise_on_failure=True) -> Result`

Runs the full loop: fetch a captcha, read it, submit it if it passes the
checks, and start again on anything wrong.

| parameter | type | meaning |
|---|---|---|
| `fetch` | callable | `fetch()` returns the bytes of a **fresh** captcha. It must use the same session and cookies as your form submission. For Bharatkosh it may take `new` (see below) |
| `submit` | callable or `None` | `submit(text)` submits the form. Return something **truthy** if the site accepted it (that value becomes `Result.response`), or `False`/`None` if the site said the captcha was wrong |
| `kind` | `str` or `None` | force a reader instead of identifying each image |
| `max_attempts` | `int` | how many captchas to fetch before giving up |
| `min_conf` | `float` or `None` | override the gate, as in `solve` |
| `max_renders` | `int` | Bharatkosh only: most renders of one answer to vote over |
| `raise_on_failure` | `bool` | `True`: raise `CaptchaUnsolved` when attempts run out. `False`: return the last `Result` |

**Returns** the accepted [`Result`](#result), with `attempts` set to the
number of fetches used and `response` set to what `submit` returned.

**Raises** [`CaptchaUnsolved`](#exceptions) after `max_attempts` (unless
`raise_on_failure=False`). Exceptions raised by your own `fetch`/`submit`
(network errors and so on) propagate unchanged. They are not captcha failures,
and retrying them blindly could hammer the portal.

**Bharatkosh re-renders.** If `fetch` has a parameter named `new`, the loop
calls `fetch(new=True)` to draw a new answer (`GenerateCaptcha?New=1`) and
`fetch(new=False)` to re-render the current one (`New=0`). It votes over up to
`max_renders` renders before spending an attempt on a new answer.

See [How retry decides](#how-retry-decides) for the exact rules.

### `solve_url(url, session=None, submit=None, timeout=30, **kwargs) -> Result`

Shortcut for `solve_with_retry` when the captcha is a plain GET. Needs
`requests` (the `[http]` extra).

| parameter | meaning |
|---|---|
| `url` | the captcha image URL, complete with any portal-specific parameters (Gujarat HC needs `?ct=S`) |
| `session` | the `requests.Session` your form uses. Pass it, or the answer is bound to a throwaway session |
| `submit` | as in `solve_with_retry` |
| `timeout` | per-request timeout in seconds |
| `**kwargs` | passed on to `solve_with_retry` (`max_attempts`, `kind`, …) |

A non-2xx response raises `requests.HTTPError`.

### `warmup(kinds=None)`

Loads models and templates now instead of on first use, so the first live
solve isn't slow. `kinds` is an iterable of `KINDS` keys. `None` loads every
kind, several hundred MB of models. Unknown kinds raise `ValueError`.

```python
cs.warmup(["mca", "gujhc", "calhc"])      # at service start-up
```

---

## `Result`

A frozen dataclass. `str(result)` is the text.

| field | type | meaning |
|---|---|---|
| `text` | `str` | the read. Case follows the portal: Calcutta HC and ITAT are case-insensitive, so they come back lowercase/uppercase |
| `confidence` | `float` | 0–1. What it measures depends on the reader (see below) |
| `kind` | `str` | which captcha this was |
| `accepted` | `bool` | passed every check (and, in `solve_with_retry` with `submit`, the site accepted it) |
| `reason` | `str` or `None` | why it is not accepted (see the table below), `None` when accepted |
| `attempts` | `int` | fetches used. Always 1 from `solve` |
| `response` | any | what `submit` returned for the accepted answer, else `None` |

| `reason` | meaning | what to do |
|---|---|---|
| `low-confidence` | confidence below the kind's gate | fetch a new captcha (the loop does this) |
| `ambiguous-glyph` | the read contains a glyph the generator draws identically to another, e.g. MCA's `I`/`l`, where any guess is a coin flip | fetch a new captcha |
| `wrong-length` | not the answer length this captcha always has | fetch a new captcha |
| `no-text` | nothing readable (blank or changed image) | fetch a new captcha |
| `rejected-by-site` | `submit` returned `False`/`None` | the loop fetches a new captcha |

**What confidence means.** For the model-free readers (`kaveri`, `udyam`,
`ngt`, `hrmut`, `mhigr`, `gujhc`, `madhc`, `karhc`) it measures how much of
the ink the matched glyphs explain, capped by the margin to the next-best
answer. It falls when the generator changes. For the neural readers
(`securimage`, `gst`, `mca`, `epfo`, `itat`, `bharatkosh`, `calhc`) it is the
decoder's per-character probability, and it can be high on a wrong read. Pass
`submit` for those.

## Exceptions

```
CaptchaError                 base class -- catch this to catch everything below
├── UnsupportedCaptcha       not an image, or a size no reader handles
└── CaptchaUnsolved          solve_with_retry ran out of attempts
        .attempts            list[Result], one per fetch, each with its reason
```

```python
try:
    r = cs.solve_with_retry(fetch, submit, max_attempts=4)
except cs.CaptchaUnsolved as e:
    log.warning("gave up: %s", [a.reason for a in e.attempts])
except cs.UnsupportedCaptcha:
    log.error("the portal changed its captcha")      # time to check the reader
```

---

## Supported captchas (`KINDS`)

`cs.KINDS` maps each kind to `{"size": (w, h), "length": n | None, "gate":
float, "what": str}`.

| kind | portal | image size | answer length | default gate | ms/solve (CPU, warm) |
|---|---|---|---|---|---|
| `securimage` | eCourts Securimage | 215×80 | 6 | 0.90 | ~40 |
| `gstat` | gstat numeric | 120×40 | varies | 0.90 | 1 |
| `ngt` | National Green Tribunal | 120×40 | 6 | 1.00 | <1 |
| `gst` | GST portal | 182×50 | 6 | 0.90 | 11 |
| `mca` | MCA portal | 200×80 | 6 | 0.90 | 24 |
| `epfo` | EPFO portal | 150×50 | 5 | 0.90 | 13 |
| `kaveri` | Kaveri (Karnataka registration) | 200×60 | 6 | 0.99 | 4 |
| `udyam` | Udyam registration | 225×80 | 6 | 0.85 | 101 |
| `itat` | ITAT | 150×42 | 6 | 0.90 | 15 |
| `bharatkosh` | Bharatkosh (NTRP) | 150×40 | 6 | 0.90 | 11 |
| `hrmut` | Haryana Jamabandi mutation | 60×20 | 6 | 0.97 | 23 |
| `mhigr` | Maharashtra IGR | 100×30 | 6 | 0.90 | 14 |
| `gujhc` | Gujarat High Court | 200×50 | 5 | 0.90 | 60 |
| `madhc` | Madras High Court | 120×60 | 6 | 0.99 | 54 |
| `calhc` | Calcutta High Court | 120×36 | 6 | 0.90 | 16 |
| `karhc` | Karnataka High Court | 100×20 | 6 | 0.70 | 1 |

Gates were chosen on each reader's validation split, never on its test
split. Held-out accuracy and how each reader works are in the
[README](../README.md).

## How retry decides

For each attempt, `solve_with_retry`:

1. calls `fetch()` and reads the image (identifying it unless `kind` is given);
2. checks the read: right length, no ambiguous glyph, confidence ≥ gate;
3. if any check fails, does **not** submit, and goes back to 1 with a fresh captcha;
4. if `submit` is `None`, returns the read;
5. calls `submit(text)`. Truthy means it returns, with `response` set.
   `False`/`None` marks the attempt `rejected-by-site` and goes back to 1;
6. after `max_attempts`, raises `CaptchaUnsolved` with every attempt.

Two guarantees follow:

- **No wasted submissions.** A read that looks wrong is never sent, so the
  portal sees at most one bad answer per captcha.
- **Session order.** The loop never fetches between reading and submitting,
  and the returned `Result` always belongs to the captcha fetched **last**.
  Most of these portals keep only the most recently served captcha per
  session, so any other order would submit an answer the server has already
  forgotten.

Without `submit` the loop can only catch reads that *look* wrong. A
confidently wrong read is only caught when the site rejects it.

## Recipes

**A `requests` session with a form.** `fetch` and `submit` share `s`:

```python
import requests, captcha_solver as cs

s = requests.Session()
s.get(SEARCH_PAGE)                                  # pick up session cookies

def fetch():
    return s.get(CAPTCHA_URL, timeout=30).content

def submit(text):
    page = s.post(SEARCH_URL, data={"case_no": "123/2024", "captcha": text}, timeout=30)
    return None if "invalid captcha" in page.text.lower() else page

page = cs.solve_with_retry(fetch, submit).response
```

**Gujarat High Court**: the servlet returns an empty body without `ct=S`.

```python
url = "https://gujarathc-casestatus.nic.in/gujarathc/CaptchaServlet?ct=S&tm=%d"
fetch = lambda: s.get(url % (time.time() * 1000), timeout=30).content
```

**Karnataka High Court** speaks TLS 1.3 only. A Python linked against old
LibreSSL (the macOS system `python3`) fails with
`TLSV1_ALERT_PROTOCOL_VERSION`. Use a Python with OpenSSL ≥ 1.1.1, or fetch
with curl. The JPEG arrives labelled `text/html`; that's harmless, because
`solve` sniffs the bytes.

**Bharatkosh**: give `fetch` a `new` parameter to get re-render voting.

```python
def fetch(new):
    return s.get(BHARATKOSH_CAPTCHA, params={"New": int(new)}, timeout=30).content

r = cs.solve_with_retry(fetch, submit)
```

**An async scraper**: the solver is synchronous CPU work (milliseconds), so
run it in a thread:

```python
r = await asyncio.to_thread(cs.solve, image_bytes)
```

**No form, just reading** (for example, an image you already have):

```python
r = cs.solve(img)
answer = r.text if r.accepted else None
```

## CLI

```bash
captcha-solver a.png b.jpg
# a.png   078086  [madhc]  conf 0.999  ok
# b.jpg   int44   [gujhc]  conf 0.223  retry (low-confidence)
python -m captcha_solver a.png
```

The exit status is 1 if any file could not be read as a supported captcha.

## Threads, memory, latency

- **Thread-safe.** Models load once, lazily, behind a lock. Inference holds no
  shared mutable state.
- **Memory.** Each neural reader is about 24 MB of weights. The model-free
  readers are at most a few hundred KB of templates. Only the kinds you
  actually solve, or `warmup`, get loaded.
- **Latency.** See the table above: single-digit milliseconds to about 100 ms
  per image on a laptop CPU. There is no GPU path and none is needed.

## Lower-level engine

`captcha_solver.engine.api` (the repo's `solver.api`) exposes the per-kind
readers (`solve_mca`, `solve_gujhc`, …), `solve_bytes(image, kind=None) ->
(text, confidence, kind)`, `GATES`, `AMBIGUOUS`, and the Bharatkosh helpers
`solve_bharatkosh_group` and `solve_bharatkosh_with_rerender`. Unlike
`identify`, `solve_bytes` falls back to a guess on an unknown size, for
backward compatibility. Prefer the top-level API.
