"""Maharashtra IGR public search captcha, 100x30 GIF, six uppercase alnum."""
from .fixedgreen import solve_image as _solve
CHARSET='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
LENGTH=6
W,H=100,30

def solve_image(image): return _solve(image,'mhigr')
