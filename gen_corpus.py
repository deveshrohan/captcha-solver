"""Generate a labeled corpus of synthetic gstat-style captchas.

Usage: python3 gen_corpus.py [N] [OUTDIR]   (defaults: 1000, corpus/)
Each file is named <label>_<index>.png so labels are recoverable and unique.
"""
import os
import sys
import numpy as np

from solver.synth import make_captcha, random_label

SEED = 20240710


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    outdir = sys.argv[2] if len(sys.argv) > 2 else "corpus"
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(SEED)
    for i in range(n):
        label = random_label(rng)
        img = make_captcha(label, rng)
        img.save(os.path.join(outdir, f"{label}_{i:04d}.png"))
    print(f"wrote {n} captchas to {outdir}/")


if __name__ == "__main__":
    main()
