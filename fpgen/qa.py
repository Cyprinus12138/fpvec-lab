"""Self-QA: structural invariants + the fpeval verify gate.

Every emitted sample must pass BOTH. fpeval.verify is the external honesty
gate (validity + watertightness) — never repaired against, only rejected on.
"""

from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon

from fpeval import from_vlm_json, verify

from .scene import ScenePlan


class QAReject(Exception):
    def __init__(self, cause: str, issues: list[str] | None = None):
        super().__init__(cause)
        self.cause = cause
        self.issues = issues or []


def check_invariants(plan: ScenePlan) -> list[str]:
    """fpgen-side structural invariants (mm space). Returns list of issues."""
    issues: list[str] = []
    walls = plan.wall_by_id()
    rooms = plan.room_by_id()

    for o in plan.openings:
        w = walls.get(o.wall_id)
        if w is None:
            issues.append(f"opening {o.id}: unknown wall {o.wall_id}")
            continue
        if o.offset_mm < 0 or o.offset_mm + o.width_mm > w.length + 1:
            issues.append(f"opening {o.id}: exceeds wall {o.wall_id}")

    for r in plan.rooms:
        for wid in r.wall_ids:
            if wid not in walls:
                issues.append(f"room {r.id}: unknown wall {wid}")
        # wall_ids may be a SUBSET of the boundary (open-plan open edges); the
        # region invariant is on the polygon ring, not the wall count.
        if len(r.polygon) < 3:
            issues.append(f"room {r.id}: polygon has fewer than 3 vertices")

    for f in plan.fixtures:
        room = rooms.get(f.room_id)
        if room is None:
            issues.append(f"fixture {f.id}: unknown room {f.room_id}")
            continue
        rpoly = Polygon(room.polygon)
        fpoly = Polygon(f.corners())
        if not fpoly.within(rpoly.buffer(5)):
            issues.append(f"fixture {f.id} ({f.kind}): outside room {f.room_id}")

    fixtures = plan.fixtures
    for i, a in enumerate(fixtures):
        pa = Polygon(a.corners())
        for b in fixtures[i + 1:]:
            if a.room_id != b.room_id:
                continue
            if a.params.get("in_counter") and b.kind == "counter":
                continue
            if b.params.get("in_counter") and a.kind == "counter":
                continue
            pb = Polygon(b.corners())
            inter = pa.intersection(pb).area
            if inter > 1000:  # >10 cm² overlap
                issues.append(f"fixtures overlap: {a.id}({a.kind})~{b.id}({b.kind})")
    return issues


def gate(plan: ScenePlan, h: np.ndarray, frame: float = 1024.0) -> dict:
    """Run invariants + fpeval verify on the exported GT. Returns the GT doc.

    Raises QAReject on any failure — the caller re-draws the sample.
    """
    issues = check_invariants(plan)
    if issues:
        raise QAReject("invariants", issues)
    gt = plan.to_vlm_json(h)
    result = verify(from_vlm_json(gt, frame=frame))
    if result.r_val < 1.0:
        raise QAReject("fpeval_verify", result.issues)
    gt.setdefault("meta", {})["verify"] = result.as_dict()
    return gt
