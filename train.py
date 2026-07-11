"""Train the digit CNN on the generated captcha corpus.

Each corpus image (filename '<label>_<idx>.png') is run through the real
preprocess+segment pipeline; images that cleanly split into exactly len(label)
digit runs contribute correctly-labeled 28x28 crops. Training uses light
shift augmentation. Every epoch we also evaluate on the real sample images in
the project root (filename == 6-digit label) as a true transfer signal.
"""
import glob
import os
import numpy as np
import torch
import torch.nn as nn

from solver.preprocess import load_ink
from solver.segment import _runs, crop_digit
from solver.model import DigitCNN
from solver.pipeline import image_to_digits, predict

SEED = 1234
CORPUS = "corpus"
EPOCHS = 16
BS = 128
MODEL_PATH = "solver/model.pt"


def load_corpus(outdir, out=28):
    X, y = [], []
    kept = skipped = 0
    for p in sorted(glob.glob(os.path.join(outdir, "*.png"))):
        label = os.path.basename(p).split("_")[0]
        ink = load_ink(p)
        runs = [r for r in _runs(ink.sum(axis=0)) if (r[1] - r[0]) > 1]
        if len(runs) != len(label):        # ambiguous split -> don't risk bad labels
            skipped += 1
            continue
        for (x0, x1), ch in zip(runs, label):
            X.append(crop_digit(ink, x0, x1, out))
            y.append(int(ch))
        kept += 1
    return np.asarray(X, np.float32), np.asarray(y, np.int64), kept, skipped


def shift_aug(imgs, rng, m=2):
    """Zero-fill random shift of each image by up to +-m px."""
    out = np.zeros_like(imgs)
    for i in range(len(imgs)):
        dy, dx = int(rng.integers(-m, m + 1)), int(rng.integers(-m, m + 1))
        ys, ye = max(0, dy), min(imgs.shape[1], imgs.shape[1] + dy)
        xs, xe = max(0, dx), min(imgs.shape[2], imgs.shape[2] + dx)
        out[i, ys:ye, xs:xe] = imgs[i, ys - dy:ye - dy, xs - dx:xe - dx]
    return out


def real_samples():
    out = []
    for p in sorted(glob.glob("*.png")):
        name = os.path.splitext(os.path.basename(p))[0]
        if name.isdigit() and len(name) == 6:
            out.append((p, name))
    return out


def eval_real(model, samples):
    correct = total = exact = 0
    details = []
    for path, label in samples:
        pred, _ = predict(model, image_to_digits(path))
        s = "".join(map(str, pred))
        correct += sum(a == b for a, b in zip(s, label))
        total += len(label)
        exact += (s == label)
        details.append((label, s))
    return correct, total, exact, details


def main():
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    X, y, kept, skipped = load_corpus(CORPUS)
    print(f"corpus: kept {kept} images ({len(X)} labeled digits), skipped {skipped} ambiguous")
    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]
    ntr = int(len(X) * 0.9)
    Xtr, ytr, Xva, yva = X[:ntr], y[:ntr], X[ntr:], y[ntr:]
    Xva_t = torch.tensor(Xva[:, None]); yva_t = torch.tensor(yva)

    model = DigitCNN()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    samples = real_samples()
    print(f"real validation images: {[s[1] for s in samples]}")

    best = (-1, -1.0)
    for ep in range(1, EPOCHS + 1):
        model.train()
        order = rng.permutation(ntr)
        tot = 0.0
        for i in range(0, ntr, BS):
            idx = order[i:i + BS]
            xb = torch.tensor(shift_aug(Xtr[idx], rng)[:, None])
            yb = torch.tensor(ytr[idx])
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            va = (model(Xva_t).argmax(1) == yva_t).float().mean().item()
        rc, rt, exact, details = eval_real(model, samples)
        print(f"epoch {ep:2d}  loss {tot/ntr:.4f}  corpus_val_acc {va:.4f}  "
              f"REAL {rc}/{rt} digits  exact {exact}/{len(samples)}")
        score = (rc, va)
        if score > best:
            best = score
            torch.save(model.state_dict(), MODEL_PATH)
    print(f"\nsaved best -> {MODEL_PATH}  (real {best[0]} digits, corpus_val {best[1]:.4f})")


if __name__ == "__main__":
    main()
