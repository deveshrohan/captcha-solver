"""CLI: solve captcha image(s), auto-routing by captcha type.

  120x40  -> gstat numeric captcha   (threshold + segment + per-digit CNN)
  215x80  -> eCourts securimage      (CRNN + CTC whole-image reader)

Usage: python solve.py IMAGE [IMAGE ...]
"""
import sys
import torch
from PIL import Image

from solver.model import DigitCNN
from solver.pipeline import solve_image
from solver import securimage as S

GSTAT_MODEL = "solver/model.pt"
SECURIMAGE_MODEL = "solver/securimage_model.pt"


def _load_gstat():
    m = DigitCNN(); m.load_state_dict(torch.load(GSTAT_MODEL, map_location="cpu")); m.eval()
    return m


def _load_securimage():
    m = S.SecurimageCRNN(); m.load_state_dict(torch.load(SECURIMAGE_MODEL, map_location="cpu")); m.eval()
    return m


def main(argv):
    if not argv:
        print("usage: python solve.py IMAGE [IMAGE ...]")
        return 1
    gstat = securi = None
    for p in argv:
        w, h = Image.open(p).size
        if w > 180:                                            # securimage (215x80)
            if securi is None:
                securi = _load_securimage()
            text, conf = S.predict_ctc_beam(securi, S.load_real(p)[None], "cpu")  # length-6 beam
            print(f"{p}\t{text[0]}\t[securimage]\t(char conf {conf[0]:.2f})")
        else:                                                   # gstat (120x40)
            if gstat is None:
                gstat = _load_gstat()
            text, conf = solve_image(gstat, p)
            print(f"{p}\t{text}\t[gstat]\t(min digit conf {min(conf):.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
