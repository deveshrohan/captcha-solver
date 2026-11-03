"""Shared pieces for readers that segment a captcha into glyphs and classify
each one (`up_rera`, `ga_rera`): connected components without scipy, and a
small CNN over a fixed-size glyph canvas.
"""
import numpy as np
import torch
import torch.nn as nn


def components(mask, min_px=1):
    """8-connected components of a bool mask -> list of dicts
    {x0, x1, y0, y1 (inclusive), n, pix: (ys, xs)} sorted by x0."""
    H, W = mask.shape
    lab = np.zeros((H, W), np.int32)
    out = []
    ys0, xs0 = np.nonzero(mask)
    for y, x in zip(ys0.tolist(), xs0.tolist()):
        if lab[y, x]:
            continue
        k = len(out) + 1
        lab[y, x] = k
        stack, py, px = [(y, x)], [], []
        while stack:
            cy, cx = stack.pop()
            py.append(cy)
            px.append(cx)
            for dy in (-1, 0, 1):
                yy = cy + dy
                if yy < 0 or yy >= H:
                    continue
                for dx in (-1, 0, 1):
                    xx = cx + dx
                    if 0 <= xx < W and mask[yy, xx] and not lab[yy, xx]:
                        lab[yy, xx] = k
                        stack.append((yy, xx))
        if len(py) >= min_px:
            py, px = np.array(py), np.array(px)
            out.append(dict(x0=int(px.min()), x1=int(px.max()), y0=int(py.min()),
                            y1=int(py.max()), n=len(py), pix=(py, px)))
    return sorted(out, key=lambda c: c["x0"])


def merge(a, b):
    py = np.concatenate([a["pix"][0], b["pix"][0]])
    px = np.concatenate([a["pix"][1], b["pix"][1]])
    return dict(x0=min(a["x0"], b["x0"]), x1=max(a["x1"], b["x1"]), y0=min(a["y0"], b["y0"]),
                y1=max(a["y1"], b["y1"]), n=a["n"] + b["n"], pix=(py, px))


class GlyphCNN(nn.Module):
    """1 x H x W glyph canvas -> class logits."""

    def __init__(self, n_classes, h=40, w=40):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3),
                                  nn.Linear(128 * (h // 8) * (w // 8), 256), nn.ReLU(),
                                  nn.Linear(256, n_classes))

    def forward(self, x):
        return self.head(self.f(x))


def predict(model, canvases):
    """canvases (n, H, W) float in [0, 1] -> (argmax indices, softmax probs)."""
    with torch.no_grad():
        x = torch.from_numpy(np.asarray(canvases, np.float32))[:, None]
        p = torch.softmax(model(x), 1).numpy()
    return p.argmax(1), p
