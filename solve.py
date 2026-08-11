"""CLI: solve captcha image(s), auto-routing by captcha type.

  120x40  -> gstat numeric captcha   (threshold + segment + per-digit CNN)
           OR NGT portal             (exact black mask -> fixed-grid glyph lookup, no model)
           These two share a size — the only collision here — so they are told
           apart by content: gstat's ink is #333 and it never emits a pure-black
           pixel, while NGT's text is pure black. See solver/api._route_120x40.
  150x42  -> ITAT portal             (darkness projection -> CRNN + CTC, case-insensitive)
  150x50  -> EPFO portal             (exact background subtraction -> CRNN + CTC)
  182x50  -> GST portal              (CRNN + CTC, RGB, length-6 beam)
  200x60  -> Kaveri portal           (exact ink mask -> sprite cover, no model)
  200x80  -> MCA portal              (exact ink mask -> CRNN + CTC)
  215x80  -> eCourts securimage      (CRNN + CTC whole-image reader)
  225x80  -> Udyam portal            (luminance ink mask -> template cover, no model)

Usage: python solve.py IMAGE [IMAGE ...]
"""
import sys

from PIL import Image

from solver.api import solve_bytes


def main(argv):
    if not argv:
        print("usage: python solve.py IMAGE [IMAGE ...]")
        return 1
    for p in argv:
        try:
            text, conf, kind = solve_bytes(p)
        except FileNotFoundError:
            print(f"{p}\t<missing file>")
            continue
        w, h = Image.open(p).size
        print(f"{p}\t{text}\t[{kind} {w}x{h}]\t(conf {conf:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
