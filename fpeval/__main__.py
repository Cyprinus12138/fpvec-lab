"""CLI: python -m fpeval verify plan.json | python -m fpeval eval pred.json gt.json

Run from the repository root: python -m fpeval verify|eval ...
"""

from __future__ import annotations

import argparse
import json
import sys

from .metrics import evaluate
from .model import load_any
from .verify import verify


def _load(path: str):
    with open(path) as f:
        return load_any(json.load(f))


def main() -> int:
    ap = argparse.ArgumentParser(prog="fpeval")
    sub = ap.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("verify", help="validity + watertightness (no GT needed)")
    v.add_argument("plan")

    e = sub.add_parser("eval", help="full pred-vs-GT evaluation")
    e.add_argument("pred")
    e.add_argument("gt")
    e.add_argument("--tol-frac", type=float, default=0.015,
                   help="matching tolerance as fraction of GT frame")

    args = ap.parse_args()
    if args.cmd == "verify":
        res = verify(_load(args.plan))
        print(json.dumps(res.as_dict(), indent=2))
        return 0 if res.r_val > 0 else 1
    else:
        rep = evaluate(_load(args.pred), _load(args.gt), tol_frac=args.tol_frac)
        print(json.dumps(rep.as_dict(), indent=2))
        return 0


if __name__ == "__main__":
    sys.exit(main())
