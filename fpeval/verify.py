"""Standalone validity + watertightness verification (R_val).

Works WITHOUT ground truth — usable as a training-data QA filter and as the
binary term of the RL reward.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from shapely.geometry import Polygon
from shapely.ops import unary_union

from .geometry import room_faces, room_polygon_from_wall_ids, seg_point_dist
from .model import FloorPlan


@dataclass
class VerifyResult:
    valid: bool                 # schema/reference validity (structural)
    watertight: bool            # all rooms resolve to closed faces
    r_val: float                # binary reward term: 1.0 iff valid AND watertight
    n_faces: int = 0
    issues: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "valid": self.valid, "watertight": self.watertight, "r_val": self.r_val,
            "n_faces": self.n_faces, "issues": self.issues,
        }


def _region_tiling_ok(plan: FloorPlan, tol: float, issues: list[str]) -> bool:
    """Watertight iff the room polygons partition the interior: each a valid,
    non-degenerate ring; no pairwise interior overlap; union is a single
    connected region with no interior holes. Open (non-wall) room edges are
    fine here — walls are not consulted. Slivers below the same threshold
    room_faces uses are tolerated (grid-snap / transform rounding)."""
    min_area = 0.0004 * plan.frame * plan.frame
    polys = []
    for r in plan.rooms:
        p = Polygon(r.polygon)
        if not p.is_valid:
            p = p.buffer(0)
        if p.is_empty or p.area <= min_area:
            issues.append(f"room '{r.label}' polygon degenerate/invalid")
            return False
        polys.append(p)
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if polys[i].intersection(polys[j]).area > min_area:
                issues.append("room polygons overlap (not a partition)")
                return False
    u = unary_union(polys)
    if u.geom_type != "Polygon":
        issues.append("room polygons do not tile a single connected region")
        return False
    if any(Polygon(h).area > min_area for h in u.interiors):
        issues.append("room tiling has interior gaps")
        return False
    return True


def verify(plan: FloorPlan, weld_tol: float | None = None) -> VerifyResult:
    tol = weld_tol if weld_tol is not None else plan.frame * 0.012
    issues: list[str] = []

    # --- structural validity -------------------------------------------------
    if not plan.walls:
        return VerifyResult(valid=False, watertight=False, r_val=0.0,
                            issues=["no walls"])
    seen_ids: set[str] = set()
    by_id = {}
    for w in plan.walls:
        if w.id in seen_ids:
            issues.append(f"duplicate wall id {w.id}")
        seen_ids.add(w.id)
        by_id[w.id] = w
        if w.length < plan.frame * 0.002:
            issues.append(f"degenerate wall {w.id} (len {w.length:.1f})")
        lo, hi = -plan.frame * 0.05, plan.frame * 1.05
        for c in (*w.start, *w.end):
            if not (lo <= c <= hi):
                issues.append(f"wall {w.id} coordinate {c:.0f} out of frame")
                break
        for o in w.openings:
            if o.width <= 0:
                issues.append(f"opening on {w.id} has non-positive width")
            if o.width > w.length + tol:
                issues.append(f"opening on {w.id} wider ({o.width:.0f}) than wall ({w.length:.0f})")
            d = seg_point_dist(o.center, w.start, w.end)
            if d > max(w.thickness, tol):
                issues.append(f"opening on {w.id} center {d:.1f} off its wall")
    for r in plan.rooms:
        for wid in r.wall_ids:
            if wid not in by_id:
                issues.append(f"room '{r.label}' references missing wall {wid}")

    valid = not issues

    # --- watertightness ------------------------------------------------------
    faces = room_faces(plan, tol)
    n_rooms = len(plan.rooms) if plan.rooms else None

    if n_rooms and all(r.polygon and len(r.polygon) >= 3 for r in plan.rooms):
        # region-tiling watertightness: rooms carry explicit polygons, so a room
        # boundary need NOT be fully walled (open-plan edges are allowed). The
        # test is that the room polygons partition the interior — no gaps, no
        # overlaps — not that every room edge chains into a wall cycle.
        watertight = _region_tiling_ok(plan, tol, issues)
    else:
        # wall-graph watertightness: no explicit polygons (predictions), so rooms
        # must resolve to closed wall-ID cycles / polygonized faces.
        watertight = bool(faces)
        if not faces:
            issues.append("wall graph encloses no faces (nothing is closed)")
        if n_rooms:
            if len(faces) < n_rooms:
                watertight = False
                issues.append(f"{n_rooms} rooms declared but only {len(faces)} closed faces")
            for r in plan.rooms:
                if len(r.wall_ids) >= 3:
                    if room_polygon_from_wall_ids(plan, r.wall_ids, tol) is None:
                        watertight = False
                        issues.append(f"room '{r.label}' wall cycle does not close")

    r_val = 1.0 if (valid and watertight) else 0.0
    return VerifyResult(valid=valid, watertight=watertight, r_val=r_val,
                        n_faces=len(faces), issues=issues)
