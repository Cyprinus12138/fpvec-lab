"""Shared shapely geometry: welding, envelope, room faces."""

from __future__ import annotations

import math

from shapely.geometry import LineString, Polygon, MultiPolygon
from shapely.ops import polygonize, unary_union

from .model import FloorPlan, Point


def weld_points(plan: FloorPlan, tol: float) -> dict[Point, Point]:
    """Cluster wall endpoints within `tol`; map raw endpoint -> cluster centroid.

    Same trick as pascal_adapter's _weld: greedy clustering, order-stable."""
    mapping: dict[Point, Point] = {}
    clusters: list[list[Point]] = []
    for w in plan.walls:
        for p in (w.start, w.end):
            if p in mapping:
                continue
            placed = False
            for cl in clusters:
                cx = sum(q[0] for q in cl) / len(cl)
                cy = sum(q[1] for q in cl) / len(cl)
                if math.dist(p, (cx, cy)) <= tol:
                    cl.append(p)
                    placed = True
                    break
            if not placed:
                clusters.append([p])
    for cl in clusters:
        cx = sum(q[0] for q in cl) / len(cl)
        cy = sum(q[1] for q in cl) / len(cl)
        for p in cl:
            mapping[p] = (cx, cy)
    return mapping


def welded_lines(plan: FloorPlan, tol: float) -> list[LineString]:
    """Wall centerlines (curved walls sampled) with welded endpoints."""
    weld = weld_points(plan, tol)
    lines = []
    for w in plan.walls:
        pts = w.polyline()
        pts[0] = weld.get(w.start, pts[0])
        pts[-1] = weld.get(w.end, pts[-1])
        ls = LineString(pts)
        if ls.length > 1e-9:
            lines.append(ls)
    return lines


def envelope_polygon(plan: FloorPlan, weld_tol: float | None = None) -> Polygon | MultiPolygon:
    """Filled building footprint: union of wall bands, holes filled.

    If walls close a loop, the band union's exterior ring encloses the interior
    -> filling holes yields the true footprint. If walls DON'T close, the result
    stays a thin band and Ext-IoU vs a filled GT footprint drops — non-closure
    penalizes itself, which is exactly the reward shape we want."""
    if not plan.walls:
        return Polygon()
    tol = weld_tol if weld_tol is not None else plan.frame * 0.012
    bands = []
    for w, ls in zip(plan.walls, welded_lines(plan, tol)):
        bands.append(ls.buffer(max(w.thickness, plan.frame * 0.004) / 2.0, cap_style="square"))
    union = unary_union(bands)
    polys = union.geoms if isinstance(union, MultiPolygon) else [union]
    filled = [Polygon(p.exterior) for p in polys if not p.is_empty]
    out = unary_union(filled)
    return out


def room_faces(plan: FloorPlan, weld_tol: float | None = None,
               min_area_frac: float = 0.0004) -> list[Polygon]:
    """Enclosed faces of the welded wall-centerline graph = candidate rooms.

    unary_union noding splits lines at crossings; polygonize returns closed
    faces. Slivers below `min_area_frac` of frame^2 are dropped."""
    tol = weld_tol if weld_tol is not None else plan.frame * 0.012
    lines = welded_lines(plan, tol)
    if not lines:
        return []
    faces = list(polygonize(unary_union(lines)))
    min_area = min_area_frac * plan.frame * plan.frame
    return [f for f in faces if f.area >= min_area]


def room_polygon_from_wall_ids(plan: FloorPlan, wall_ids: list[str],
                               weld_tol: float | None = None) -> Polygon | None:
    """Resolve a room given as a wall-ID cycle into a polygon (None if it
    doesn't chain into a closed loop within the weld tolerance)."""
    tol = weld_tol if weld_tol is not None else plan.frame * 0.012
    by_id = plan.wall_by_id()
    segs = [(by_id[i].start, by_id[i].end) for i in wall_ids if i in by_id]
    if len(segs) < 3:
        return None
    # chain greedily
    chain = [segs.pop(0)]
    while segs:
        tail = chain[-1][1]
        best, best_d, flip = None, None, False
        for k, (a, b) in enumerate(segs):
            for f, p in ((False, a), (True, b)):
                d = math.dist(tail, p)
                if best_d is None or d < best_d:
                    best, best_d, flip = k, d, f
        if best is None or best_d > tol * 2:
            return None
        a, b = segs.pop(best)
        chain.append((b, a) if flip else (a, b))
    if math.dist(chain[-1][1], chain[0][0]) > tol * 2:
        return None
    pts = [c[0] for c in chain] + [chain[-1][1]]
    poly = Polygon(pts)
    return poly if poly.is_valid and poly.area > 0 else None


def seg_point_dist(pt: Point, a: Point, b: Point) -> float:
    """Distance from point to segment ab."""
    ax, ay = a
    bx, by = b
    px, py = pt
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    if L2 < 1e-12:
        return math.dist(pt, a)
    t = min(max(((px - ax) * dx + (py - ay) * dy) / L2, 0.0), 1.0)
    return math.dist(pt, (ax + t * dx, ay + t * dy))
