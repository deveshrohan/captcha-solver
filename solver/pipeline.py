"""Inference pipeline shared by solve.py and training-time evaluation."""
import numpy as np
import torch

from .preprocess import load_ink
from .segment import segment
from .model import DigitCNN

N_DIGITS = 6


def image_to_digits(path, n=N_DIGITS, out=28):
    ink = load_ink(path)
    return segment(ink, n, out)  # list of (out x out) uint8 arrays


def predict(model, digit_arrays, device="cpu"):
    X = torch.tensor(np.stack(digit_arrays)[:, None, :, :], dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        logits = model(X)
        conf = torch.softmax(logits, dim=1).max(1).values
        pred = logits.argmax(1)
    return pred.cpu().tolist(), conf.cpu().tolist()


def solve_image(model, path, device="cpu"):
    digs = image_to_digits(path)
    pred, conf = predict(model, digs, device)
    return "".join(map(str, pred)), conf
