"""Pred-vs-GT metrics + FloorplanVLM-style gated reward.

Metrics (brief §5.6): validity rate, Ext-IoU, Room-IoU, Room-F1@0.5,
Opening-F1 — plus Wall-F1 and mean aligned wall distance (our own
"numbers to beat": <10% walls missing, <5cm aligned).

Reward (brief §4.3): R = 0.1*R_val + 0.5*R_ext + 0.4*alpha*R_int, where
alpha gates internal credit on the external boundary being right.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .geometry import (envelope_polygon, room_faces,
                       room_polygon_from_wall_ids, seg_point_dist)
from .model import FloorPlan, Wall
from .verify import verify

try:
    from scipy.optimize import linear_sum_assignment
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False


# ---------------------------------------------------------------- matching

def _match(cost: list[list[float]], max_cost: float) -> list[tuple[int, int]]:
    """Min-cost bipartite matching; pairs with cost > max_cost are dropped.
    Hungarian when scipy is present, greedy otherwise."""
    if not cost or not cost[0]:
        return []
    if _HAVE_SCIPY:
        import numpy as np
        m = np.array(cost)
        big = max_cost * 1e6 + 1.0
        m[~np.isfinite(m)] = big   # scipy rejects infeasible (all-inf) rows/cols
        rows, cols = linear_sum_assignment(m)
        return [(r, c) for r, c in zip(rows, cols) if cost[r][c] <= max_cost]
    pairs, used_r, used_c = [], set(), set()
    flat = sorted((cost[r][c], r, c)
                  for r in range(len(cost)) for c in range(len(cost[0])))
    for v, r, c in flat:
        if v > max_cost:
            break
        if r in used_r or c in used_c:
            continue
        pairs.append((r, c))
        used_r.add(r)
        used_c.add(c)
    return pairs


def _prf(n_match: int, n_pred: int, n_gt: int) -> tuple[float, float, float]:
    p = n_match / n_pred if n_pred else 0.0
    r = n_match / n_gt if n_gt else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


# ---------------------------------------------------------------- walls

def _wall_cost(a: Wall, b: Wall, tol_dist: float) -> float:
    """Symmetric segment distance if the pair is plausibly the same wall,
    else inf. Criteria: angle < 15 deg, mean sampled distance < tol, and
    projection overlap >= 30% of the shorter wall."""
    va = (a.end[0] - a.start[0], a.end[1] - a.start[1])
    vb = (b.end[0] - b.start[0], b.end[1] - b.start[1])
    la, lb = a.length, b.length
    if la < 1e-9 or lb < 1e-9:
        return math.inf
    cosang = abs(va[0] * vb[0] + va[1] * vb[1]) / (la * lb)
    if cosang < math.cos(math.radians(15)):
        return math.inf
    # mean distance of samples of a onto b, and b onto a
    def mean_d(src: Wall, dst: Wall, n: int = 7) -> float:
        vx, vy = src.end[0] - src.start[0], src.end[1] - src.start[1]
        s = 0.0
        for i in range(n):
            t = i / (n - 1)
            p = (src.start[0] + vx * t, src.start[1] + vy * t)
            s += seg_point_dist(p, dst.start, dst.end)
        return s / n
    d = (mean_d(a, b) + mean_d(b, a)) / 2
    if d > tol_dist:
        return math.inf
    # projection overlap along the longer wall's axis
    lw = a if la >= lb else b
    sw = b if la >= lb else a
    ux, uy = (lw.end[0] - lw.start[0]) / max(la, lb), (lw.end[1] - lw.start[1]) / max(la, lb)
    def proj(p):
        return (p[0] - lw.start[0]) * ux + (p[1] - lw.start[1]) * uy
    lo, hi = sorted((proj(sw.start), proj(sw.end)))
    overlap = min(hi, max(la, lb)) - max(lo, 0.0)
    if overlap < 0.3 * min(la, lb):
        return math.inf
    return d


def wall_metrics(pred: FloorPlan, gt: FloorPlan, tol_dist: float) -> dict:
    cost = [[_wall_cost(pw, gw, tol_dist) for gw in gt.walls] for pw in pred.walls]
    pairs = _match(cost, tol_dist)
    p, r, f = _prf(len(pairs), len(pred.walls), len(gt.walls))
    dists = [cost[i][j] for i, j in pairs]
    return {
        "precision": p, "recall": r, "f1": f,
        "matched": len(pairs), "pred": len(pred.walls), "gt": len(gt.walls),
        "missing_frac": 1.0 - r,
        "mean_dist": sum(dists) / len(dists) if dists else None,
    }


# ---------------------------------------------------------------- openings

def opening_metrics(pred: FloorPlan, gt: FloorPlan, tol_dist: float) -> dict:
    po, go = pred.openings, gt.openings
    cost = []
    for a in po:
        row = []
        for b in go:
            d = math.dist(a.center, b.center)
            wr = min(a.width, b.width) / max(a.width, b.width) if max(a.width, b.width) > 0 else 0
            if a.family != b.family or d > tol_dist or wr < 0.5:
                row.append(math.inf)
            else:
                row.append(d)
        cost.append(row)
    pairs = _match(cost, tol_dist)
    p, r, f = _prf(len(pairs), len(po), len(go))
    return {"precision": p, "recall": r, "f1": f,
            "matched": len(pairs), "pred": len(po), "gt": len(go)}


# ---------------------------------------------------------- human edit cost

# Topology-aware CAD edit distance (research-plan §5): estimated human cost to
# turn a prediction into GT in the editor. Hand-set, NON-GATING weights; the
# ORDERING is the load-bearing claim (calibrate magnitudes against real editor
# ops later). Ordering: move/snap << type-change < create/delete << open<->wall
# conversion. Open<->phantom is most expensive because it is delete-wall PLUS
# re-open the room boundary (or vice versa) — the exact op room-centric models
# force by manufacturing phantom/double walls on shared/open edges.
DEFAULT_EDIT_WEIGHTS = {
    "move": 1.0,          # per tol_dist of endpoint displacement (cheap, continuous)
    "type": 2.0,          # change wall type
    "create": 5.0,        # add a missing wall
    "delete": 5.0,        # remove an extra (phantom/double) wall
    "open_convert": 10.0, # phantom-wall-over-open-edge (or missing wall on a real
                          # boundary): delete/create + fix the open/closed topology
    "open_move": 0.5,     # nudge an existing opening
    "open_create": 4.0,   # add a missing opening
    "open_delete": 4.0,   # remove an extra opening
    "move_radius": 4.0,   # walls within move_radius*tol are "the same wall moved",
                          # not delete+create — the correspondence generosity
}


def _gt_open_edges(gt: FloorPlan, tol: float) -> list[tuple]:
    """GT room-polygon edges not covered by any GT wall = open boundaries."""
    walls = [(w.start, w.end) for w in gt.walls]
    open_edges = []
    for r in gt.rooms:
        poly = r.polygon
        if not poly or len(poly) < 3:
            continue
        for k in range(len(poly)):
            a, b = poly[k], poly[(k + 1) % len(poly)]
            mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
            covered = any(seg_point_dist(mid, ws, we) <= tol for ws, we in walls)
            if not covered:
                open_edges.append((a, b))
    return open_edges


def _wall_on_edges(w: Wall, edges: list[tuple], tol: float) -> bool:
    """Does wall w lie along any of the given (open) edges? (midpoint proximity)"""
    mid = ((w.start[0] + w.end[0]) / 2, (w.start[1] + w.end[1]) / 2)
    return any(seg_point_dist(mid, a, b) <= tol for a, b in edges)


def edit_cost(pred: FloorPlan, gt: FloorPlan, tol_dist: float,
              weights: dict | None = None) -> dict:
    """Estimated human edit cost pred -> GT (topology-aware CAD edit distance).

    Reuses the wall matcher with a GENEROUS radius (a displaced wall is a *move*,
    not delete+create). Returns total cost, per-GT-wall cost (comparable across
    plans), and the op breakdown (create/delete/type/move/open_convert/...).
    """
    W = {**DEFAULT_EDIT_WEIGHTS, **(weights or {})}
    move_tol = W["move_radius"] * tol_dist
    cost = [[_wall_cost(pw, gw, move_tol) for gw in gt.walls] for pw in pred.walls]
    pairs = _match(cost, move_tol)
    mp = {i for i, _ in pairs}
    mg = {j for _, j in pairs}

    ops = {k: 0 for k in ("move", "type", "create", "delete", "open_convert",
                          "open_move", "open_create", "open_delete")}
    total = 0.0

    for i, j in pairs:
        total += W["move"] * (cost[i][j] / tol_dist); ops["move"] += 1
        if getattr(pred.walls[i], "type", None) != getattr(gt.walls[j], "type", None):
            total += W["type"]; ops["type"] += 1

    n_create = len(gt.walls) - len(mg)
    total += W["create"] * n_create; ops["create"] = n_create

    gt_open = _gt_open_edges(gt, tol_dist)
    for i in range(len(pred.walls)):
        if i in mp:
            continue
        if gt_open and _wall_on_edges(pred.walls[i], gt_open, move_tol):
            total += W["open_convert"]; ops["open_convert"] += 1  # phantom over open
        else:
            total += W["delete"]; ops["delete"] += 1

    # openings: generous center match; move / create / delete (reattach folds into
    # delete+create when the host wall is a phantom — a v2 calibration refinement).
    po, go = pred.openings, gt.openings
    ocost = []
    for a in po:
        row = []
        for b in go:
            d = math.dist(a.center, b.center) if a.family == b.family else math.inf
            row.append(d if d <= move_tol else math.inf)
        ocost.append(row)
    opairs = _match(ocost, move_tol)
    omp = {i for i, _ in opairs}; omg = {j for _, j in opairs}
    for i, j in opairs:
        total += W["open_move"] * (ocost[i][j] / tol_dist); ops["open_move"] += 1
    ops["open_create"] = len(go) - len(omg)
    ops["open_delete"] = len(po) - len(omp)
    total += W["open_create"] * ops["open_create"] + W["open_delete"] * ops["open_delete"]

    return {"total": round(total, 3),
            "per_gt_wall": round(total / max(1, len(gt.walls)), 4),
            "ops": ops}


# ---------------------------------------------------------------- rooms

def _resolve_rooms(plan: FloorPlan, weld_tol: float | None = None):
    """Room polygons: explicit boundary > wall-ID cycle > polygonized faces."""
    from shapely.geometry import Polygon
    polys = []
    for r in plan.rooms:
        if r.polygon and len(r.polygon) >= 3:
            poly = Polygon(r.polygon)
            if poly.is_valid and poly.area > 0:
                polys.append((r.label, poly))
                continue
        if len(r.wall_ids) >= 3:
            poly = room_polygon_from_wall_ids(plan, r.wall_ids, weld_tol)
            if poly is not None:
                polys.append((r.label, poly))
    if polys:
        return polys
    return [("", f) for f in room_faces(plan, weld_tol)]


def room_metrics(pred: FloorPlan, gt: FloorPlan, iou_thresh: float = 0.5) -> dict:
    pr, gr = _resolve_rooms(pred), _resolve_rooms(gt)
    cost = []
    for _, pa in pr:
        row = []
        for _, ga in gr:
            inter = pa.intersection(ga).area
            union = pa.union(ga).area
            iou = inter / union if union > 0 else 0.0
            row.append(1.0 - iou if iou >= iou_thresh else math.inf)
        cost.append(row)
    pairs = _match(cost, 1.0 - iou_thresh)
    p, r, f = _prf(len(pairs), len(pr), len(gr))
    ious = [1.0 - cost[i][j] for i, j in pairs]
    labels_ok = sum(
        1 for i, j in pairs
        if pr[i][0] and gr[j][0] and pr[i][0].strip().lower() == gr[j][0].strip().lower())
    labeled_pairs = sum(1 for i, j in pairs if pr[i][0] and gr[j][0])
    return {
        "precision": p, "recall": r, "f1": f,
        "matched": len(pairs), "pred": len(pr), "gt": len(gr),
        "mean_iou": sum(ious) / len(ious) if ious else None,
        "label_acc": labels_ok / labeled_pairs if labeled_pairs else None,
    }


# ---------------------------------------------------------------- composite

def ext_iou(pred: FloorPlan, gt: FloorPlan) -> float:
    pe, ge = envelope_polygon(pred), envelope_polygon(gt)
    if pe.is_empty or ge.is_empty:
        return 0.0
    union = pe.union(ge).area
    return pe.intersection(ge).area / union if union > 0 else 0.0


def alpha_gate(ext: float, lo: float = 0.3, hi: float = 0.7) -> float:
    """FloorplanVLM gating: 0.1 below `lo`, linear to 1.0 at `hi`."""
    if ext <= lo:
        return 0.1
    if ext >= hi:
        return 1.0
    return 0.1 + (ext - lo) / (hi - lo) * 0.9


@dataclass
class EvalReport:
    r_val: float
    ext_iou: float
    alpha: float
    walls: dict
    openings: dict
    rooms: dict
    reward: float
    edit: dict = field(default_factory=dict)
    verify_issues: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "r_val": self.r_val, "ext_iou": round(self.ext_iou, 4),
            "alpha": round(self.alpha, 4), "reward": round(self.reward, 4),
            "walls": self.walls, "openings": self.openings, "rooms": self.rooms,
            "edit": self.edit, "verify_issues": self.verify_issues,
        }


def evaluate(pred: FloorPlan, gt: FloorPlan,
             tol_frac: float = 0.015,
             w_val: float = 0.1, w_ext: float = 0.5, w_int: float = 0.4) -> EvalReport:
    """Full evaluation. `tol_frac` — matching tolerance as a fraction of the
    GT frame (0.015 * 1024 ≈ 15 grid units ≈ 15cm on a 10m plan)."""
    tol = tol_frac * gt.frame
    v = verify(pred)
    ext = ext_iou(pred, gt)
    a = alpha_gate(ext)
    wm = wall_metrics(pred, gt, tol)
    om = opening_metrics(pred, gt, tol * 2)   # opening centers are fuzzier
    rm = room_metrics(pred, gt)
    ec = edit_cost(pred, gt, tol)
    internal = (wm["f1"] + om["f1"] + rm["f1"]) / 3.0
    reward = w_val * v.r_val + w_ext * ext + w_int * a * internal
    return EvalReport(r_val=v.r_val, ext_iou=ext, alpha=a,
                      walls=wm, openings=om, rooms=rm,
                      reward=reward, edit=ec, verify_issues=v.issues)
