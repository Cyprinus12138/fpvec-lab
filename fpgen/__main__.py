"""CLI: python -m fpgen generate|preview|overlay"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(prog="fpgen")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="generate a dataset run")
    g.add_argument("-n", type=int, default=10)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--style", default="clean_cad",
                   help="family or comma mix like clean_cad:0.5,marketing:0.5")
    g.add_argument("--typology", default=None, help="pin e.g. 2B2b / studio")
    g.add_argument("--degrade", default=None,
                   choices=["none", "light", "medium", "heavy"],
                   help="pin degradation level; default samples a mix")
    g.add_argument("--out", default="out/fpgen")
    g.add_argument("--run-id", default=None)
    g.add_argument("--workers", type=int, default=1)
    g.add_argument("--open-plan", dest="open_plan", default=None,
                   choices=["on", "off"],
                   help="pin open-plan LDK (wall-less kitchen~living); default samples")

    p = sub.add_parser("preview", help="one sample end-to-end + overlay")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--style", default="clean_cad")
    p.add_argument("--typology", default=None)

    o = sub.add_parser("overlay", help="overlays + contact sheet for a run")
    o.add_argument("run_dir")
    o.add_argument("--take", type=int, default=None)
    o.add_argument("--cols", type=int, default=5)
    o.add_argument("--rows", type=int, default=4)

    args = ap.parse_args()

    if args.cmd == "generate":
        from .pipeline import GenConfig, generate
        mix = {}
        for part in args.style.split(","):
            name, _, weight = part.partition(":")
            mix[name] = float(weight) if weight else 1.0
        cfg = GenConfig(seed=args.seed, n=args.n, out_dir=args.out,
                        run_id=args.run_id, style_mix=mix,
                        typology=args.typology, degrade_level=args.degrade,
                        workers=args.workers,
                        open_plan={"on": True, "off": False}.get(args.open_plan))
        summary = generate(cfg)
        print(json.dumps(summary, indent=1))

    elif args.cmd == "preview":
        from .pipeline import GenConfig, generate
        from .viz import contact_sheet
        cfg = GenConfig(seed=args.seed, n=1, out_dir="out/fpgen",
                        run_id=f"preview_seed{args.seed}",
                        style_mix={args.style: 1.0}, typology=args.typology)
        summary = generate(cfg)
        print(json.dumps(summary, indent=1))
        if summary["generated"]:
            sheet = contact_sheet(Path(summary["out"]), cols=1, rows=1, cell=900)
            print(f"overlay sheet: {sheet}")

    elif args.cmd == "overlay":
        from .viz import contact_sheet, run_overlays
        run_dir = Path(args.run_dir)
        paths = run_overlays(run_dir, take=args.take)
        sheet = contact_sheet(run_dir, cols=args.cols, rows=args.rows)
        print(f"{len(paths)} overlays; sheet: {sheet}")


if __name__ == "__main__":
    main()
