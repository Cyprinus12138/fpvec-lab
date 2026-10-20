"""Score a predictions file against a ResPlan-FP (or skv4) split with fpeval.

    python -m baselines.score --bench data/resplan-fp --split test \
        --preds data/resplan-fp/baselines/zeroshot_seq_decode.preds.jsonl [--tol 0.05 0.015]

Predictions file: one JSON object per line, ``{"id": "<row id>", "pred": <plan json or null>}``.
A plan is the same wall-first JSON as the benchmark ``gt`` field (walls with
start/end/thickness/openings, optional rooms), in the 1024 longer-edge frame.

Protocol (the one every number in the paper and README uses): every plan of the
split counts; a missing or ``null`` prediction scores zero on every F1, and the
coverage column reports how many plans had a prediction. ``--ids`` restricts
scoring to a subset (e.g. the 200-plan LLM subset).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fpeval import evaluate, from_vlm_json
from fpeval.metrics import wall_metrics


def load_rows(bench: Path, split: str, ids=None):
    rows = [json.loads(l) for l in open(bench / f"{split}.jsonl")]
    if ids is not None:
        ids = set(ids); rows = [r for r in rows if r["id"] in ids]
    return rows


def load_preds(p: Path):
    return {j["id"]: j.get("pred") for j in (json.loads(l) for l in open(p))}


def score(rows, preds, tols=(0.05, 0.015), full=True):
    out = {"n": len(rows), "covered": 0, "tols": {}}
    per = {t: {"wall_f1": [], "wall_p": [], "wall_r": [], "room_f1": [], "open_f1": [], "edit": []} for t in tols}
    r_val = []
    for r in rows:
        gt = from_vlm_json(r["gt"], frame=float(r.get("frame", 1024.0)))
        p = preds.get(r["id"])
        if p is None:
            for t in tols:
                for k in per[t]:
                    if k != "edit":
                        per[t][k].append(0.0)
            r_val.append(0.0)
            continue
        out["covered"] += 1
        pred = from_vlm_json(p, frame=1024.0)
        for t in tols:
            if full:
                rep = evaluate(pred, gt, tol_frac=t)
                per[t]["wall_f1"].append(rep.walls["f1"]); per[t]["wall_p"].append(rep.walls["precision"]); per[t]["wall_r"].append(rep.walls["recall"])
                per[t]["room_f1"].append(rep.rooms["f1"]); per[t]["open_f1"].append(rep.openings["f1"])
                per[t]["edit"].append(rep.edit["total"])
                if t == tols[0]:
                    r_val.append(rep.r_val)
            else:
                wm = wall_metrics(pred, gt, t * gt.frame)
                per[t]["wall_f1"].append(wm["f1"]); per[t]["wall_p"].append(wm["precision"]); per[t]["wall_r"].append(wm["recall"])
    for t in tols:
        out["tols"][str(t)] = {k: round(float(np.mean(v)), 4) for k, v in per[t].items() if v}
    if r_val:
        out["r_val"] = round(float(np.mean(r_val)), 4)
    return out


def main():
    ap = argparse.ArgumentParser(prog="python -m baselines.score")
    ap.add_argument("--bench", required=True, help="benchmark dir holding <split>.jsonl")
    ap.add_argument("--split", default="test")
    ap.add_argument("--preds", required=True)
    ap.add_argument("--ids", default=None, help="json list of row ids to restrict scoring to")
    ap.add_argument("--tol", type=float, nargs="+", default=[0.05, 0.015])
    ap.add_argument("--walls-only", action="store_true", help="wall F1 only (fast)")
    a = ap.parse_args()
    ids = json.load(open(a.ids)) if a.ids else None
    rows = load_rows(Path(a.bench), a.split, ids)
    res = score(rows, load_preds(Path(a.preds)), tuple(a.tol), full=not a.walls_only)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
