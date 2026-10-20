"""Partition skv4 converter output by the OFFICIAL CubiCasa5K split lists.

    python -m skv4.split --converted <dir from skv4.convert> \
        --cc5k-root <extracted cubicasa5k dir containing train.txt/val.txt/test.txt> \
        --out <dir>

``skv4.convert`` writes a stride-based train/val for convenience; the released
dataset (and every number in the paper) uses the official CubiCasa5K
train/val/test lists intersected with the plans that convert. Plans missing
from all three lists are dropped and counted in ``stats.json``.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path


def read_list(p: Path) -> set[str]:
    return {l.strip().strip("/").split("/")[-1] for l in p.read_text().splitlines() if l.strip()}


def main():
    import argparse
    ap = argparse.ArgumentParser(prog="python -m skv4.split")
    ap.add_argument("--converted", required=True)
    ap.add_argument("--cc5k-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--copy-images", action="store_true", help="copy images/ into --out")
    a = ap.parse_args()
    src, root, out = Path(a.converted), Path(a.cc5k_root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    official = {s: read_list(root / f"{s}.txt") for s in ("train", "val", "test")}
    rows = [json.loads(l) for f in ("train.jsonl", "val.jsonl") if (src / f).exists()
            for l in open(src / f)]
    rows.sort(key=lambda r: r["id"])
    parts = {s: [] for s in official}; dropped = []
    for r in rows:
        sid = r["id"].split("/")[-1]
        for s, ids in official.items():
            if sid in ids:
                parts[s].append(r); break
        else:
            dropped.append(r["id"])
    for s, rs in parts.items():
        (out / f"{s}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rs))
    if a.copy_images:
        shutil.copytree(src / "images", out / "images", dirs_exist_ok=True)
    stats = {"official": {s: len(v) for s, v in official.items()},
             "converted": len(rows), "counts": {s: len(v) for s, v in parts.items()},
             "dropped_not_in_lists": dropped}
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps({k: v for k, v in stats.items() if k != "dropped_not_in_lists"}, indent=1),
          f"dropped {len(dropped)}")


if __name__ == "__main__":
    main()
