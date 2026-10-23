"""Jamabandi public mutation captcha, 60x20 GIF, six mixed-case characters."""
from solver.fixedgreen import solve_image as _solve
CHARSET='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
LENGTH=6
W,H=60,20

def solve_image(image): return _solve(image,'hrmut')
