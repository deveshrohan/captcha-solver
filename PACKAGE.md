# captcha-solver

Offline readers for Indian government-portal captchas. Hand it a captcha image:
it works out which portal's captcha it is from the image itself, reads it, and
says whether the read is safe to submit. Give it a way to fetch a fresh captcha
(and, ideally, to submit the form) and it retries on its own until the site
accepts an answer.

Everything runs locally on CPU. No image leaves the process. Full reference:
[docs/API.md](docs/API.md).

```bash
pip install "git+https://github.com/deveshrohan/captcha-solver.git"
pip install "./captcha-solver[http]"    # from a checkout, + requests for solve_url()
```

## One image

```python
import captcha_solver as cs

cs.identify(image_bytes)          # 'mca'
r = cs.solve(image_bytes)         # bytes, a path, a file object or a PIL image
r.text, r.confidence, r.kind      # ('ab3Xk9', 0.97, 'mca')
r.accepted, r.reason              # (True, None)  or  (False, 'low-confidence')
```

`accepted` is False when the read should not be submitted. `reason` says why:

| reason | meaning |
|---|---|
| `low-confidence` | below the kind's gate (`cs.KINDS[kind]["gate"]`) |
| `ambiguous-glyph` | contains a glyph the generator draws identically to another, e.g. MCA's `I`/`l`: a coin flip however good the reader |
| `wrong-length` / `no-text` | not a valid answer shape for this captcha |
| `rejected-by-site` | (retry only) your `submit` reported the answer wrong |

## Let the package retry

```python
import requests
import captcha_solver as cs

s = requests.Session()                  # the SAME session the form uses

def fetch():
    return s.get(CAPTCHA_URL, timeout=30).content

def submit(text):
    page = s.post(FORM_URL, data={..., "captcha": text}, timeout=30)
    if "invalid captcha" in page.text.lower():
        return False                    # rejected -> the package fetches a new one
    return page                         # accepted -> comes back as r.response

r = cs.solve_with_retry(fetch, submit, max_attempts=5)
results_page = r.response
```

What the loop does on each attempt:

1. `fetch()` a fresh captcha and read it.
2. If the read is not `accepted` (low confidence, ambiguous glyph, wrong
   length), it is **never submitted**. A new captcha is fetched instead.
3. Otherwise `submit(text)`. A falsy return means the site said the captcha
   was wrong, so it goes back to step 1. A truthy return ends the loop.
4. After `max_attempts` fetches it raises `cs.CaptchaUnsolved`, whose
   `.attempts` lists every Result and why it failed. Pass
   `raise_on_failure=False` to get the last Result instead.

`submit` is optional, but it is the only way to catch a read that is
confidently wrong. Confidence can't flag those. On GST, ITAT and Bharatkosh the
README shows wrong reads at confidence ≥ 0.99. Without `submit`, only reads
that *look* wrong are retried, and you get back the first accepted read.

**Sessions matter.** Most of these portals keep only the most recently served
captcha per session. So `fetch` and `submit` must share one session, and the
returned Result always belongs to the captcha fetched last. The loop never
fetches between reading an answer and submitting it.

**Bharatkosh** can re-render the same answer (`GenerateCaptcha?New=0`). If
your `fetch` takes a `new` argument, the loop calls `fetch(new=True)` for a new
answer and `fetch(new=False)` for extra looks at the current one. It votes over
up to `max_renders` renders before spending an attempt on a new answer.

For a plain GET there is a shortcut (needs `requests`):

```python
r = cs.solve_url(CAPTCHA_URL, session=s, submit=submit)
```

## Supported captchas

`cs.KINDS` lists each kind's image size, answer length and gate. Kinds are
told apart by image size, plus a content test for the one shared size
(120×40: gstat vs NGT). An unknown size raises `cs.UnsupportedCaptcha` and is
never guessed.

securimage (eCourts), gstat, ngt, gst, mca, epfo, kaveri, udyam, itat
(before Oct 2026) / itat2 (ITAT since Oct 2026), bharatkosh, hrmut, mhigr, gujhc / madhc / calhc / karhc (Gujarat, Madras,
Calcutta and Karnataka High Courts).

Endpoint notes that bite: Gujarat HC's `CaptchaServlet` needs `?ct=S` (without
it the body is empty). Karnataka HC accepts TLS 1.3 only, which rules out
Python builds linked against old LibreSSL, like the macOS system python3.
Calcutta HC lowercases the answer, so case never matters there.

## Startup and threads

Models load lazily on first use. To pay that cost at startup instead, call
`cs.warmup(["mca", "gujhc"])` (with no argument it loads every kind, several
hundred MB). Inference is thread-safe.

## CLI

```bash
captcha-solver a.png b.jpg      # path, text, kind, confidence, ok / retry (reason)
python -m captcha_solver a.png
```

Use only against services you are authorized to automate.
