#!/usr/bin/env python3
"""Self-improvement loop for the captcha readers.

Captcha generators change without warning — a font swap, a new distortion, a
different length — and the failure mode is silent: the model keeps returning
confident nonsense and the scraper just stops finding results. This script is
built around that risk rather than around chasing accuracy.

    python3 selfimprove.py check                 # drift + regression check, exit 1 on trouble
    python3 selfimprove.py harvest --kind gst --n 200
    python3 selfimprove.py adapt   --kind gst    # self-train on confident pseudo-labels
    python3 selfimprove.py report

Three ideas hold it together:

**A pinned, hand-labelled eval set is the only ground truth.** Confidence is not
accuracy — on GST, reads above 0.99 confidence were still only 88% correct — so
nothing here trusts confidence as a quality signal. Every claim about accuracy
comes from `*_labels.json`, which a human wrote.

**`adapt` can only ever improve.** It self-trains on high-confidence pseudo-
labels, but the new weights are kept *only* if the pinned eval set does not
regress. A loop that can silently make things worse is worse than no loop, and
pseudo-labelling is exactly the kind of process that drifts into reinforcing its
own mistakes.

**`check` needs no labels.** Drift shows up in signals available without ground
truth: the served image size, the fraction of reads with a valid length and
charset, and the confidence distribution versus the last recorded run. Those
catch "the endpoint changed" long before anyone notices missing scrape results.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np

STATE = "selfimprove_state.json"

# kind -> how to fetch, where the corpus lives, expected shape, label file
KINDS = {
    "gst": dict(size=(182, 50), length=6, charset="0123456789",
                corpus="gst_raw", glob="g*.png", labels="gst_labels.json",
                fetch=["python3", "download_gst.py"], model="solver/gst_model.pt"),
    "mca": dict(size=(200, 80), length=6,
                charset="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
                corpus="mca_raw", glob="m*.png", labels="mca_labels.json",
                fetch=["python3", "download_mca.py"], model="solver/mca_model.pt"),
    "epfo": dict(size=(150, 50), length=5,
                 charset="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
                 corpus="epfo_raw", glob="e*.png", labels="epfo_labels.json",
                 fetch=["python3", "download_epfo.py"], model="solver/epfo_model.pt"),
    # No model: Kaveri is read by exact sprite cover, so `adapt` has nothing to
    # train and refuses. `check` still matters — arguably more than elsewhere,
    # because the reader depends on the generator's bitmaps being unchanged, and
    # a font swap turns confidence from 1.0 into something visibly lower rather
    # than into silent nonsense.
    "kaveri": dict(size=(200, 60), length=6,
                   charset="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
                   corpus="kaveri_raw", glob="*.png", labels="kaveri_labels.json",
                   fetch=["python3", "download_kaveri.py"], model=None),
}


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_state():
    return json.load(open(STATE)) if os.path.exists(STATE) else {}


def save_state(s):
    json.dump(s, open(STATE, "w"), indent=1, sort_keys=True)


def _solve_many(paths, kind):
    from solver.api import solve_bytes
    out = []
    for p in paths:
        try:
            t, c, k = solve_bytes(open(p, "rb").read(), kind=kind)
            out.append((p, t, c))
        except Exception as e:                       # a decode/shape change shows up here
            out.append((p, None, 0.0))
    return out


# --------------------------------------------------------------------------
# health: works without any labels
# --------------------------------------------------------------------------
def health(kind, sample=60):
    from PIL import Image
    cfg = KINDS[kind]
    paths = sorted(glob.glob(os.path.join(cfg["corpus"], cfg["glob"])))[-sample:]
    if not paths:
        return dict(kind=kind, n=0, note="no corpus")

    sizes = {}
    for p in paths:
        try:
            sizes[Image.open(p).size] = sizes.get(Image.open(p).size, 0) + 1
        except Exception:
            sizes["unreadable"] = sizes.get("unreadable", 0) + 1
    solved = _solve_many(paths, kind)
    confs = np.array([c for _, _, c in solved])
    texts = [t for _, t, _ in solved]
    ok_len = np.mean([t is not None and len(t) == cfg["length"] for t in texts])
    ok_chr = np.mean([t is not None and all(ch in cfg["charset"] for ch in t) for t in texts])
    return dict(kind=kind, n=len(paths), at=now(),
                size_ok=float(sizes.get(tuple(cfg["size"]), 0)) / len(paths),
                sizes={str(k): v for k, v in sizes.items()},
                conf_mean=float(confs.mean()), conf_p10=float(np.percentile(confs, 10)),
                frac_low_conf=float((confs < 0.90).mean()),
                len_ok=float(ok_len), charset_ok=float(ok_chr))


def labelled_accuracy(kind):
    """Exact-match on the hand-labelled HELD-OUT split — the only real ground truth.

    Deliberately restricted to the pinned test split where one exists. Scoring
    every label instead would include images the model was fine-tuned on, which
    it has memorised: MCA reads 96.5% over all 395 labels versus 75% on its 44
    held-out ones. A regression detector built on the inflated number would stay
    green while real accuracy fell.
    """
    cfg = KINDS[kind]
    if not os.path.exists(cfg["labels"]):
        return None
    labels = json.load(open(cfg["labels"]))
    split_file = cfg["labels"].replace("_labels.json", "_split.json")
    held_out = None
    if os.path.exists(split_file):
        held_out = set(json.load(open(split_file)).get("test", []))
    items = [(p, l) for p, l in sorted(labels.items())
             if os.path.exists(p) and (held_out is None or p in held_out)]
    if not items:
        return None
    got = _solve_many([p for p, _ in items], kind)
    exact = sum(t == l for (_, t, _), (_, l) in zip(got, items))
    chars = sum(sum(a == b for a, b in zip(t or "", l)) for (_, t, _), (_, l) in zip(got, items))
    return dict(n=len(items), exact=exact, exact_frac=exact / len(items),
                chars=chars, char_frac=chars / (len(items) * cfg["length"]))


def cmd_check(args):
    state = load_state()
    problems = []
    for kind in (args.kinds or list(KINDS)):
        h = health(kind, args.sample)
        acc = labelled_accuracy(kind)
        prev = state.get(kind, {})
        print(f"\n[{kind}]  n={h.get('n',0)}")
        if not h.get("n"):
            print("   no corpus — run `harvest` first")
            continue
        print(f"   served size as expected : {h['size_ok']*100:.0f}%   sizes={h['sizes']}")
        print(f"   reads with valid length : {h['len_ok']*100:.0f}%")
        print(f"   reads in charset        : {h['charset_ok']*100:.0f}%")
        print(f"   confidence mean/p10     : {h['conf_mean']:.3f} / {h['conf_p10']:.3f}"
              f"   ({h['frac_low_conf']*100:.0f}% below 0.90)")
        if acc:
            print(f"   labelled eval           : {acc['exact']}/{acc['n']} exact "
                  f"({acc['exact_frac']*100:.1f}%), chars {acc['char_frac']*100:.2f}%")

        # drift rules -- deliberately blunt; these fire on "the endpoint changed"
        if h["size_ok"] < 0.98:
            problems.append(f"{kind}: served image size changed ({h['sizes']})")
        if h["len_ok"] < 0.90:
            problems.append(f"{kind}: {(1-h['len_ok'])*100:.0f}% of reads have the wrong length")
        if h["charset_ok"] < 0.98:
            problems.append(f"{kind}: reads contain out-of-charset characters")
        pc = prev.get("health", {}).get("conf_mean")
        if pc and h["conf_mean"] < pc - 0.05:
            problems.append(f"{kind}: mean confidence fell {pc:.3f} -> {h['conf_mean']:.3f}")
        pa = prev.get("accuracy", {})
        if acc and pa.get("exact_frac") and acc["exact_frac"] < pa["exact_frac"] - 0.05:
            problems.append(f"{kind}: labelled accuracy fell "
                            f"{pa['exact_frac']*100:.1f}% -> {acc['exact_frac']*100:.1f}%")

        state.setdefault(kind, {})["health"] = h
        if acc:
            state[kind]["accuracy"] = acc
        state[kind]["checked_at"] = now()
    save_state(state)

    print()
    if problems:
        print("REGRESSIONS / DRIFT DETECTED:")
        for p in problems:
            print("  ! " + p)
        return 1
    print("no drift detected")
    return 0


def cmd_harvest(args):
    cfg = KINDS[args.kind]
    before = len(glob.glob(os.path.join(cfg["corpus"], cfg["glob"])))
    cmd = cfg["fetch"] + [str(args.n), cfg["corpus"]]
    print("running:", " ".join(cmd), flush=True)
    subprocess.run(cmd)
    after = len(glob.glob(os.path.join(cfg["corpus"], cfg["glob"])))
    print(f"corpus {args.kind}: {before} -> {after} (+{after-before})")
    return 0


def cmd_adapt(args):
    """Self-train on confident pseudo-labels, keeping the result only if the
    hand-labelled eval set does not regress."""
    kind = args.kind
    cfg = KINDS[kind]
    if cfg["model"] is None:
        print(f"{kind}: nothing to train — this reader is an exact sprite cover, "
              f"not a model. If the generator changed, the fix is to re-extract "
              f"the sprite library (solver.kaveri.extract_sprites), not to adapt "
              f"weights. Run `check` to see whether it has.")
        return 1
    script = {"gst": "finetune_gst.py", "mca": "finetune_mca.py",
              "epfo": "finetune_epfo.py"}.get(kind)
    if not script or not os.path.exists(script):
        print(f"no finetune script for {kind}")
        return 1

    before = labelled_accuracy(kind)
    if before is None:
        print(f"{kind}: no labelled eval set — refusing to adapt blind")
        return 1
    print(f"before: {before['exact']}/{before['n']} exact ({before['exact_frac']*100:.1f}%)")

    backup = cfg["model"] + ".bak"
    subprocess.run(["cp", cfg["model"], backup], check=True)
    subprocess.run(["python3", script, str(args.epochs)])

    after = labelled_accuracy(kind)
    print(f"after : {after['exact']}/{after['n']} exact ({after['exact_frac']*100:.1f}%)")
    if after["exact_frac"] + 1e-9 < before["exact_frac"]:
        subprocess.run(["mv", backup, cfg["model"]], check=True)
        print("REVERTED — adaptation made it worse, original weights restored")
        return 1
    os.remove(backup)
    print("kept")
    state = load_state()
    state.setdefault(kind, {})["accuracy"] = after
    state[kind]["adapted_at"] = now()
    save_state(state)
    return 0


def cmd_report(args):
    state = load_state()
    if not state:
        print("no state yet — run `check`")
        return 0
    print("%-6s %-22s %-10s %-10s %s" % ("kind", "last checked", "conf", "len-ok", "labelled exact"))
    for kind, s in sorted(state.items()):
        h = s.get("health", {})
        a = s.get("accuracy", {})
        print("%-6s %-22s %-10s %-10s %s" % (
            kind, s.get("checked_at", "-")[:19],
            f"{h.get('conf_mean', float('nan')):.3f}" if h else "-",
            f"{h.get('len_ok', 0)*100:.0f}%" if h else "-",
            f"{a.get('exact', '?')}/{a.get('n', '?')} ({a.get('exact_frac', 0)*100:.1f}%)" if a else "-"))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="drift + regression check")
    c.add_argument("--kinds", nargs="*", choices=list(KINDS))
    c.add_argument("--sample", type=int, default=60)
    c.set_defaults(fn=cmd_check)
    h = sub.add_parser("harvest", help="download fresh captchas")
    h.add_argument("--kind", required=True, choices=list(KINDS))
    h.add_argument("--n", type=int, default=200)
    h.set_defaults(fn=cmd_harvest)
    a = sub.add_parser("adapt", help="self-train, gated on the labelled eval set")
    a.add_argument("--kind", required=True, choices=list(KINDS))
    a.add_argument("--epochs", type=int, default=12)
    a.set_defaults(fn=cmd_adapt)
    r = sub.add_parser("report", help="summary of recorded runs")
    r.set_defaults(fn=cmd_report)
    args = ap.parse_args()
    sys.exit(args.fn(args))


if __name__ == "__main__":
    main()
