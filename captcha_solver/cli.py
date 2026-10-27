"""captcha-solver IMAGE [IMAGE ...]  -- identify and read captcha images."""
import sys

from .core import CaptchaError, solve


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__.strip())
        return 0 if argv else 1
    status = 0
    for path in argv:
        try:
            r = solve(path)
        except (CaptchaError, OSError) as e:
            print("%s\tERROR\t%s" % (path, e))
            status = 1
            continue
        flag = "ok" if r.accepted else "retry (%s)" % r.reason
        print("%s\t%s\t[%s]\tconf %.3f\t%s" % (path, r.text, r.kind, r.confidence, flag))
    return status


if __name__ == "__main__":
    sys.exit(main())
