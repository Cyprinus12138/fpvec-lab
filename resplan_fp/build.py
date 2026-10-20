"""Build ResPlan-FP end to end: ResPlan.pkl -> converted rows -> frozen splits -> renders.

    python -m resplan_fp.build --pkl ResPlan.pkl --out resplan_fp_bench [--workers 8]

Output layout (what the released benchmark contains):

    <out>/train.jsonl  val.jsonl  test.jsonl     one row per plan
    <out>/images/resplan_<id>.png                 clean_band_v1 renders (256 px)
    <out>/splits.json                             fingerprints + full test-id list
    <out>/stats.json                              conversion statistics

Split protocol (frozen, do not change): sort the accepted ids as strings,
shuffle them with ``random.Random(42)``, take the first 1000 as test, the next
1000 as val, the rest as train. Each split's fingerprint is
``sha256(",".join(sorted(ids)))[:16]``; the build aborts if the fingerprints
differ from ``resplan_fp/splits.json``, so any later copy can be verified
against the published one.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import random
from pathlib import Path

SEED = 42
N_TEST = 1000
N_VAL = 1000
FROZEN = Path(__file__).resolve().parent / "splits.json"


def fingerprint(ids) -> str:
    return hashlib.sha256(",".join(sorted(ids)).encode()).hexdigest()[:16]


def freeze_splits(ids, seed=SEED, n_test=N_TEST, n_val=N_VAL):
    c = sorted(ids)
    random.Random(seed).shuffle(c)
    return {"test": c[:n_test], "val": c[n_test:n_test + n_val], "train": c[n_test + n_val:]}


def convert_row(plan):
    """One ResPlan plan dict -> benchmark row (without image), or None if rejected.
    Returns (row, reject_reason)."""
    from fpeval import from_vlm_json, verify
    import fpseq
    from fpseq import schema
    from .convert import convert_one
    try:
        r = convert_one(plan)
    except Exception as e:                                   # noqa: BLE001
        return None, f"parse:{type(e).__name__}"
    if r is None:
        return None, "empty"
    gt, roles = r
    fp = from_vlm_json(gt, frame=schema.DEFAULT_FRAME)
    v = verify(fp)
    if not v.valid:
        return None, "invalid"
    try:
        ev = fpseq.plan_to_events(fp, roles=roles)
        t = fpseq.events_to_tensors(ev)
        if len(t["token_type"]) >= schema.SEQ_LEN or len(fp.walls) > schema.MAX_WALLS:
            return None, "over_capacity"
    except Exception:                                        # noqa: BLE001
        return None, "codec"
    n_open = sum(len(w["openings"]) for w in gt["walls"])
    return {"id": f"resplan/{gt['meta']['resplan_id']}", "gt": gt, "roles": roles,
            "frame": schema.DEFAULT_FRAME, "n_walls": len(fp.walls), "n_rooms": len(fp.rooms),
            "n_openings": n_open, "watertight": v.watertight, "r_val": v.r_val,
            "source": "resplan"}, None


def _worker(plan):
    return convert_row(plan)


def main():
    import argparse
    import cv2
    from .render import STYLE, render_row
    ap = argparse.ArgumentParser(prog="python -m resplan_fp.build")
    ap.add_argument("--pkl", required=True, help="ResPlan.pkl from the ResPlan release")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="debug: convert only the first N plans")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--no-freeze-check", action="store_true",
                    help="skip the fingerprint check against the frozen splits (debug/--limit runs)")
    a = ap.parse_args()
    out = Path(a.out); (out / "images").mkdir(parents=True, exist_ok=True)
    with open(a.pkl, "rb") as f:
        plans = pickle.load(f)
    if a.limit:
        plans = plans[:a.limit]
    rows, rej = [], {}
    if a.workers > 1:
        from multiprocessing import Pool
        with Pool(a.workers) as pool:
            results = pool.map(_worker, plans, chunksize=16)
    else:
        results = [convert_row(p) for p in plans]
    for row, why in results:
        if row is None:
            rej[why] = rej.get(why, 0) + 1
        else:
            rows.append(row)
    rows.sort(key=lambda r: r["id"])
    by_id = {r["id"]: r for r in rows}
    splits = freeze_splits(by_id)
    fps = {k: fingerprint(v) for k, v in splits.items()}
    if not a.no_freeze_check:
        frozen = json.loads(FROZEN.read_text())
        if fps != frozen["id_sha256"]:
            raise SystemExit(f"split fingerprints {fps} != frozen {frozen['id_sha256']}; "
                             f"the converter output differs from the released benchmark")
    for name, ids in splits.items():
        with open(out / f"{name}.jsonl", "w") as fh:
            for i in ids:
                r = by_id[i]
                rel = f"images/{i.replace('/', '_')}.png"
                cv2.imwrite(str(out / rel), render_row(r))
                fh.write(json.dumps({**r, "image": rel, "style": STYLE}) + "\n")
    (out / "splits.json").write_text(json.dumps(
        {"seed": SEED, "total": len(rows), "counts": {k: len(v) for k, v in splits.items()},
         "id_sha256": fps, "test_ids": sorted(splits["test"])}, indent=1))
    n = max(1, len(rows))
    stats = {"seen": len(plans), "accepted": len(rows), "rejects": rej,
             "counts": {k: len(v) for k, v in splits.items()},
             "watertight_frac": round(sum(r["watertight"] for r in rows) / n, 3),
             "mean_walls": round(sum(r["n_walls"] for r in rows) / n, 2),
             "mean_rooms": round(sum(r["n_rooms"] for r in rows) / n, 2),
             "mean_openings": round(sum(r["n_openings"] for r in rows) / n, 2)}
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
