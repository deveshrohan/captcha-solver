"""Evaluate a securimage CRNN model on held-out real captchas.

Reports greedy vs length-6 beam decode, exact-match and per-character accuracy,
over the union of hand/eye-labeled real sets:
  captcha_2/real_test/*.png   (filename = label)
  captcha_2/real_eval/*.png   (filename = label, self-labeled from downloads)
"""
import glob
import os
import sys
import numpy as np
import torch

from solver import securimage as S

MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else "solver/securimage_model.pt"


def labeled(dirs):
    items = []
    for d in dirs:
        for p in sorted(glob.glob(os.path.join(d, "*.png"))):
            lab = os.path.splitext(os.path.basename(p))[0].rstrip("_")
            if len(lab) == S.LENGTH and all(c in S.CH2I for c in lab):
                items.append((p, lab))
    return items


def score(preds, labels):
    ch = sum(sum(a == b for a, b in zip(p, l)) for p, l in zip(preds, labels))
    ex = sum(p == l for p, l in zip(preds, labels))
    return ex, ch, len(labels), len(labels) * S.LENGTH


def main():
    dev = "cpu"
    model = S.SecurimageCRNN().to(dev)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=dev))
    model.eval()

    for name, dirs in [("real_test (12)", ["captcha_2/real_test"]),
                       ("real_eval (40)", ["captcha_2/real_eval"]),
                       ("real_hard (15)", ["captcha_2/real_hard"]),
                       ("clean (test+eval)", ["captcha_2/real_test", "captcha_2/real_eval"]),
                       ("ALL (test+eval+hard)", ["captcha_2/real_test", "captcha_2/real_eval", "captcha_2/real_hard"])]:
        items = labeled(dirs)
        if not items:
            continue
        imgs = np.stack([S.load_real(p) for p, _ in items])
        labels = [l for _, l in items]
        gp, _ = S.predict_ctc(model, imgs, dev)
        bp, _ = S.predict_ctc_beam(model, imgs, dev)
        gex, gch, n, tot = score(gp, labels)
        bex, bch, _, _ = score(bp, labels)
        print(f"\n=== {name}  [{MODEL_PATH}] ===")
        print(f"  greedy : exact {gex}/{n} ({gex/n*100:.0f}%)  char {gch}/{tot} ({gch/tot*100:.1f}%)")
        print(f"  beam-6 : exact {bex}/{n} ({bex/n*100:.0f}%)  char {bch}/{tot} ({bch/tot*100:.1f}%)")
        # show the misses (beam) to inspect residual error modes
        miss = [(l, p) for l, p in zip(labels, bp) if l != p]
        if miss:
            print("  beam misses (true -> pred):",
                  ", ".join(f"{l}->{p}" for l, p in miss[:30]))


if __name__ == "__main__":
    main()
