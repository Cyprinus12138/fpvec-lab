"""Fit unit-mix meta distributions from the SG corpus title extractions.

Parses a JSONL of listing-title extractions (29.3k records;
titles like "#1 Loft 1 Bedroom, 2 Bathroom Type A1-P Floor Plan") into a joint
P(n_bed, n_bath) histogram and writes fpgen/assets/meta_stats.json.

Usage (from the repository root; the source corpus is not redistributed, only
the fitted aggregate statistics in fpgen/assets/meta_stats.json are):
    python fpgen/scripts/fit_meta.py --input <jsonl>
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "assets" / "meta_stats.json"

BED_RE = re.compile(r"(\d)\s*[- ]?\s*(?:bedroom|bed(?:room)?s?\b|br\b)", re.I)
BATH_RE = re.compile(r"(\d)\s*[- ]?\s*bath(?:room)?s?\b", re.I)
STUDIO_RE = re.compile(r"\bstudio\b", re.I)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    args = ap.parse_args()

    joint: Counter[tuple[int, int]] = Counter()
    n_total = n_parsed = 0
    with args.input.open() as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            n_total += 1
            if rec.get("classification") != "per_unit_plan":
                continue
            if rec.get("property_type") not in ("condo_apartment", "unknown"):
                continue
            title = rec.get("title") or ""
            beds = BED_RE.search(title)
            baths = BATH_RE.search(title)
            if STUDIO_RE.search(title) and not beds:
                joint[(0, 1)] += 1
                n_parsed += 1
                continue
            if not beds:
                continue
            n_bed = int(beds.group(1))
            # SG-typical bath count when the title omits it
            typical_bath = {0: 1, 1: 1, 2: 2, 3: 2, 4: 3, 5: 4, 6: 5}
            n_bath = int(baths.group(1)) if baths else typical_bath.get(n_bed, 2)
            if not (0 <= n_bed <= 6 and 1 <= n_bath <= 6):
                continue
            # sanity: baths never exceed beds+1
            n_bath = min(n_bath, n_bed + 1) if n_bed else 1
            joint[(n_bed, n_bath)] += 1
            n_parsed += 1

    # drop rare tail combos (<0.3%) — they are mostly title mis-parses
    total = sum(joint.values())
    kept = {k: v for k, v in joint.items() if v / total >= 0.003}
    doc = {
        "source": "SG residential listing titles (not redistributed; see fpgen/scripts/fit_meta.py)",
        "records_scanned": n_total,
        "records_parsed": n_parsed,
        "bed_bath": {f"{b},{ba}": c for (b, ba), c in sorted(kept.items())},
    }
    OUT.write_text(json.dumps(doc, indent=1))
    print(f"parsed {n_parsed}/{n_total}; kept {len(kept)} (bed,bath) combos -> {OUT}")
    for (b, ba), c in sorted(kept.items()):
        print(f"  {b}B{ba}b: {c} ({c / total:.1%})")


if __name__ == "__main__":
    main()
