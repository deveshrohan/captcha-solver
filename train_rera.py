#!/usr/bin/env python3
"""Train the glyph CNNs for the segmenting RERA readers.

    python3 train_rera.py ga pretrain      # synthetic Arial Italic only
    python3 train_rera.py ga finetune      # + eye-labelled train glyphs
    python3 train_rera.py up pretrain
    python3 train_rera.py up finetune

Training-time only: synthetic renders use system fonts from
/System/Library/Fonts/Supplemental. The readers themselves never touch a font.
Real glyphs come from the TRAIN split of <name>_split.json only.
"""
import io
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont, ImageFilter

from solver.glyphcnn import GlyphCNN

SUP = "/System/Library/Fonts/Supplemental/"


# ---------------------------------------------------------------- GA
def ga_render(rng, text=None):
    from solver import ga_rera as G
    text = text or "".join(rng.choice(G.CHARSET) for _ in range(G.LENGTH))
    img = Image.new("L", (G.W, G.H), 208)
    x = rng.randint(42, 50)
    size = rng.choice([30, 31, 32])
    font = ImageFont.truetype(SUP + "Arial Italic.ttf", size)
    for ch in text:
        layer = Image.new("L", (60, 60), 0)
        ImageDraw.Draw(layer).text((15, 45), ch, font=font, fill=255, anchor="ls")
        layer = layer.rotate(rng.uniform(-25, 25), resample=Image.NEAREST, center=(25, 38))
        a = np.asarray(layer) > 127
        ys, xs = np.nonzero(a)
        if not len(xs):
            continue
        dy = rng.randint(-3, 3)
        ox = x - xs.min()
        for yy, xx in zip(ys, xs):
            py, px = yy - 45 + 34 + dy, xx + ox
            if 0 <= py < G.H and 0 <= px < G.W:
                img.putpixel((px, py), 0)
        x += xs.max() - xs.min() + 1 + rng.randint(1, 8)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=rng.randint(70, 90))
    return Image.open(io.BytesIO(buf.getvalue())), text


def ga_samples(rng, n):
    from solver import ga_rera as G
    X, Y = [], []
    while len(X) < n:
        im, text = ga_render(rng)
        segs = G.segment(im)
        if len(segs) != len(text):
            continue
        for c, ch in zip(segs, text):
            X.append(G.canvas(c))
            Y.append(G.CHARSET.index(G.FOLD.get(ch, ch)))
    return np.stack(X[:n]), np.array(Y[:n])


def ga_real(split="train"):
    from solver import ga_rera as G
    L = json.load(open("ga_rera_labels.json"))
    S = json.load(open("ga_rera_split.json"))
    X, Y = [], []
    for p in S[split]:
        segs = G.segment(p)
        if len(segs) != len(L[p]):
            continue
        for c, ch in zip(segs, L[p]):
            X.append(G.canvas(c))
            Y.append(G.CHARSET.index(ch))
    return np.stack(X), np.array(Y)


# ---------------------------------------------------------------- UP
UP_FONTS = ["Arial.ttf", "Arial Bold.ttf", "Arial Black.ttf", "Times New Roman.ttf",
            "Times New Roman Bold.ttf", "Georgia.ttf", "Georgia Bold.ttf", "Verdana.ttf",
            "Verdana Bold.ttf", "Tahoma.ttf", "Tahoma Bold.ttf", "Trebuchet MS.ttf",
            "Trebuchet MS Bold.ttf", "Courier New Bold.ttf", "Arial Narrow Bold.ttf"]


def up_glyph(rng, ch):
    """A single MSCaptcha-like glyph: random font and size, thresholded."""
    from solver import up_rera as U
    fonts = [f for f in UP_FONTS if os.path.exists(SUP + f)]
    bold = [f for f in fonts if "Bold" in f or "Black" in f]
    # real glyphs are mostly heavy: small bold faces, blown up by the box scale
    face = rng.choice(bold) if rng.random() < 0.7 else rng.choice(fonts)
    font = ImageFont.truetype(SUP + face, rng.randint(13, 24))
    layer = Image.new("L", (48, 48), 0)
    ImageDraw.Draw(layer).text((24, 24), ch, font=font, fill=255, anchor="mm")
    if rng.random() < 0.5:                    # MSCaptcha squeezes some glyphs
        w = rng.uniform(0.7, 1.0)
        layer = layer.resize((max(1, int(48 * w)), 48))
    if rng.random() < 0.35:
        layer = layer.filter(ImageFilter.MaxFilter(3))
    a = np.asarray(layer) > rng.randint(50, 140)
    # black noise dots punch holes in the green ink
    if rng.random() < 0.8:
        for _ in range(rng.randint(0, 6)):
            a[rng.randrange(a.shape[0]), rng.randrange(a.shape[1])] = False
    return U.canvas_from_mask(a)


def up_samples(rng, n):
    from solver import up_rera as U
    X, Y = [], []
    for _ in range(n):
        ch = rng.choice(U.CHARSET)
        X.append(up_glyph(rng, ch))
        Y.append(U.CHARSET.index(ch))
    return np.stack(X), np.array(Y)


def up_real(split="train"):
    from solver import up_rera as U
    L = json.load(open("up_rera_labels.json"))
    S = json.load(open("up_rera_split.json"))
    X, Y = [], []
    for p in S[split]:
        segs = U.segment(p)
        if len(segs) != len(L[p]):
            continue
        for c, ch in zip(segs, L[p]):
            X.append(U.canvas(c))
            Y.append(U.CHARSET.index(ch))
    return np.stack(X), np.array(Y)


# ---------------------------------------------------------------- training
def augment(X, rng):
    """Shift by up to 2 px; for real-glyph fine-tuning."""
    out = np.zeros_like(X)
    for i, x in enumerate(X):
        dy, dx = rng.randint(-1, 1), rng.randint(-2, 2)
        out[i] = np.roll(np.roll(x, dy, 0), dx, 1)
    return out


def _val(model, Xv, Yv):
    # evaluate a CPU copy: a live model evaluated on MPS has misreported before
    import copy
    m = copy.deepcopy(model).to("cpu").eval()
    with torch.no_grad():
        return (m(torch.from_numpy(Xv)[:, None]).argmax(1).numpy() == Yv).mean()


def fit(model, X, Y, epochs, lr, Xv=None, Yv=None, aug=None, seed=0):
    dev = os.environ.get("DEVICE", "mps" if torch.backends.mps.is_available() else "cpu")
    model.to(dev)
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    lossf = nn.CrossEntropyLoss()
    rng = random.Random(seed)
    for ep in range(epochs):
        model.train()
        Xe = aug(X, rng) if aug else X
        perm = np.random.RandomState(ep + seed).permutation(len(X))
        tot = 0.0
        for i in range(0, len(X), 128):
            b = perm[i:i + 128]
            x = torch.from_numpy(Xe[b])[:, None].to(dev)
            y = torch.from_numpy(Y[b]).long().to(dev)
            opt.zero_grad()
            loss = lossf(model(x), y)
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(b)
        sched.step()
        msg = "ep %d loss %.4f" % (ep, tot / len(X))
        if Xv is not None:
            msg += " val-glyph %.4f" % _val(model, Xv, Yv)
        print(msg, flush=True)
    model.to("cpu").eval()
    return model


def main():
    torch.set_num_threads(int(os.environ.get("THREADS", "4")))
    name, stage = sys.argv[1], sys.argv[2]
    mod = __import__("solver.%s_rera" % name, fromlist=["x"])
    path = "solver/%s_rera_model.pt" % name
    gen = {"ga": ga_samples, "up": up_samples}[name]
    real = {"ga": ga_real, "up": up_real}[name]
    m = GlyphCNN(len(mod.CHARSET), mod.CH, mod.CW)
    rng = random.Random(1)
    if stage == "pretrain":
        X, Y = gen(rng, int(os.environ.get("N", "30000")))
        Xv, Yv = gen(random.Random(99), 3000)
        fit(m, X.astype(np.float32), Y, int(os.environ.get("EPOCHS", "8")), 2e-3, Xv, Yv)
        torch.save(m.state_dict(), path.replace(".pt", "_pre.pt"))
    else:
        m.load_state_dict(torch.load(path.replace(".pt", "_pre.pt")))
        Xr, Yr = real("train")
        Xs, Ys = gen(rng, int(os.environ.get("NSYN", "8000")))
        reps = max(1, len(Xs) // max(1, len(Xr)) // 2)
        X = np.concatenate([np.repeat(Xr, reps, 0), Xs]).astype(np.float32)
        Y = np.concatenate([np.repeat(Yr, reps, 0), Ys])
        print("real train glyphs", len(Xr), "x", reps, "+ synthetic", len(Xs))
        fit(m, X, Y, int(os.environ.get("EPOCHS", "6")), 5e-4, aug=augment)
        torch.save(m.state_dict(), path)
    print("saved", path)


if __name__ == "__main__":
    main()
