"""ResPlan pickle -> wall-first floorplan JSON (the ResPlan-FP converter).

ResPlan (17k plans) stores each plan as a dict of shapely geometries: 'wall'
(MultiPolygon wall bodies, rectilinear unions of wall_depth-thick rectangles),
'door'/'window'/'front_door' opening bodies that FILL the gaps in the wall band
(wall ∪ openings is one connected region), and semantic room layers
('living','bedroom','bathroom','kitchen','balcony'; one room instance per
connected component). Coordinates span ~0-260 units; 'wall_depth' is the wall
thickness in the same units.

Conversion mirrors skv4.convert and REUSES its `to_vlm_json` wholesale
(scale long edge -> 1024 frame, junction weld, Manhattan snap, degenerate drop,
exterior/interior labeling, FACES-based rooms via fpeval.room_faces with
max-overlap labels, opening projection onto nearest wall):

  wall+opening band -> rasterize -> skeletonize -> sknw -> Douglas-Peucker
  (centerlines; the ResPlan analog of cc5k's walls_from_skeleton, which is
  SVG-path-bound and can't be called directly)  ->  to_vlm_json(...)

Openings: door/window/front_door polygon -> minAreaRect -> the two short-side
midpoints are the opening span (center + width); front_door counts as "door".
Faces with no semantic-layer majority get label "room" (cc5k's 'undefined').

QA per plan: fpeval.verify(plan).valid + the fpseq codec round-trip gate
(plan_to_events/events_to_tensors under SEQ_LEN/MAX_WALLS), as in skv4.convert.

NOTE: resplan_utils.py (the authors' helpers) imports geopandas at module level
which this venv lacks; its two relevant one-liners (get_geometries,
normalize_keys) are inlined below with attribution.
"""

from __future__ import annotations

import math

import numpy as np
from shapely import minimum_rotated_rectangle
from shapely.geometry import GeometryCollection, LineString, MultiLineString, \
    MultiPolygon, Point, Polygon
from shapely.ops import unary_union

from skv4.convert import FRAME, to_vlm_json

# our five target semantic layers -> fpseq ROOM_ROLES vocabulary
ROOM_LAYERS = (("living", "living"), ("bedroom", "bedroom"),
               ("bathroom", "bath"), ("kitchen", "kitchen"),
               ("balcony", "balcony"))
OPENING_LAYERS = (("door", "door"), ("window", "window"),
                  ("front_door", "door"))          # front_door counts as door


def _geoms(geom_data):
    """Individual non-empty geometries from single/multi/collections
    (inlined from resplan_utils.get_geometries; geopandas unavailable)."""
    if geom_data is None:
        return []
    if isinstance(geom_data, (Polygon, LineString, Point)):
        return [] if geom_data.is_empty else [geom_data]
    if isinstance(geom_data, (MultiPolygon, MultiLineString, GeometryCollection)):
        return [g for g in geom_data.geoms if g is not None and not g.is_empty]
    return []


def _normalize_keys(plan: dict) -> dict:
    """balacony -> balcony (inlined from resplan_utils.normalize_keys)."""
    if "balacony" in plan and "balcony" not in plan:
        plan["balcony"] = plan.pop("balacony")
    return plan


def walls_from_band(band_parts, wall_depth, ox, oy, H, W, res=4.0):
    """ResPlan analog of skv4.convert.walls_from_skeleton (same machinery, but
    that function is bound to the CubiCasa5K SVG parse). Rasterize the wall band
    polygons (walls + door/window bodies, which fill the opening gaps so the
    band is continuous) -> morphological close -> skeletonize -> sknw graph
    (nodes = junctions, edges = wall runs) -> Douglas-Peucker each run into
    straight segments. Returns [(p0, p1, thickness, "interior")] in translated
    plan units (origin at band bbox min)."""
    import cv2
    from skimage.morphology import skeletonize
    import sknw
    h, w = int(math.ceil(H * res)) + 3, int(math.ceil(W * res)) + 3
    mask = np.zeros((h, w), np.uint8)
    for poly in band_parts:
        ext = np.round(np.array([[(x - ox) * res, (y - oy) * res]
                                 for x, y in poly.exterior.coords])).astype(np.int32)
        if len(ext) >= 3:
            cv2.fillPoly(mask, [ext], 1)
        for ring in poly.interiors:
            hole = np.round(np.array([[(x - ox) * res, (y - oy) * res]
                                      for x, y in ring.coords])).astype(np.int32)
            if len(hole) >= 3:
                cv2.fillPoly(mask, [hole], 0)
    if mask.sum() == 0:
        return []
    k = max(3, int(round(wall_depth * res * 0.5)) | 1)   # close gaps ~½ thickness
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    graph = sknw.build_sknw(skeletonize(mask > 0))
    eps = max(h, w) * 0.006
    segs = []
    for u, v in graph.edges():
        pts = graph[u][v]["pts"]                          # Nx2 (row, col)
        if len(pts) < 2:
            continue
        xy = np.array([[p[1], p[0]] for p in pts], np.float32)
        # pin the run's ends to the exact junction node coords (shared endpoints)
        xy[0] = graph.nodes[u]["o"][::-1]; xy[-1] = graph.nodes[v]["o"][::-1]
        appr = cv2.approxPolyDP(xy.reshape(-1, 1, 2), eps, False).reshape(-1, 2)
        for a, b in zip(appr[:-1], appr[1:]):
            if math.dist(a, b) > wall_depth * res * 0.6:
                segs.append(((float(a[0]) / res, float(a[1]) / res),
                             (float(b[0]) / res, float(b[1]) / res),
                             wall_depth, "interior"))
    return segs


def _manhattan_snap(segs, tol):
    """Snap endpoint coordinates to shared axis values (1-D greedy clustering
    of all endpoint x's, then y's; each cluster -> its mean). ResPlan is
    strictly Manhattan, but skeleton junctions carry ~1-raster-px jitter, so
    near-collinear runs come out slightly sloped; downstream that jitter makes
    face polygons touch at points instead of sharing edges, which breaks
    fpeval's region-tiling watertightness. Snapping BEFORE to_vlm_json keeps
    its weld/axis-clean no-ops on already-consistent coordinates."""
    def snap_axis(vals):
        m = {}
        cur = []
        for v in sorted(vals):
            if cur and v - cur[-1] > tol:
                c = sum(cur) / len(cur)
                for x in cur:
                    m[x] = c
                cur = []
            cur.append(v)
        if cur:
            c = sum(cur) / len(cur)
            for x in cur:
                m[x] = c
        return m
    mx = snap_axis([p[0] for a, b, *_ in segs for p in (a, b)])
    my = snap_axis([p[1] for a, b, *_ in segs for p in (a, b)])
    return [((mx[a[0]], my[a[1]]), (mx[b[0]], my[b[1]]), th, ty)
            for a, b, th, ty in segs]


def _opening_span(poly):
    """Opening body polygon -> (p0, p1) endpoints of its long axis: minAreaRect,
    midpoints of the two SHORT sides. Width = dist(p0, p1) = rect long side."""
    r = minimum_rotated_rectangle(poly)
    if r.geom_type != "Polygon":
        return None
    c = list(r.exterior.coords)[:4]
    e1, e2 = math.dist(c[0], c[1]), math.dist(c[1], c[2])
    if e1 >= e2:      # long axis along (c0,c1): endpoints = short-side midpoints
        p0 = ((c[3][0] + c[0][0]) / 2, (c[3][1] + c[0][1]) / 2)
        p1 = ((c[1][0] + c[2][0]) / 2, (c[1][1] + c[2][1]) / 2)
    else:
        p0 = ((c[0][0] + c[1][0]) / 2, (c[0][1] + c[1][1]) / 2)
        p1 = ((c[2][0] + c[3][0]) / 2, (c[2][1] + c[3][1]) / 2)
    return p0, p1


def convert_one(plan: dict):
    """ResPlan plan dict -> (vlm_json, roles) or None if degenerate."""
    plan = _normalize_keys(plan)
    wall_parts = [g for g in _geoms(plan.get("wall")) if g.geom_type == "Polygon"]
    opening_parts = []                                    # [(Polygon, kind)]
    for layer, kind in OPENING_LAYERS:
        for g in _geoms(plan.get(layer)):
            if g.geom_type == "Polygon":
                opening_parts.append((g, kind))
    if not wall_parts:
        return None

    band_parts = wall_parts + [g for g, _k in opening_parts]
    ox, oy, mx, my = unary_union(band_parts).bounds       # content origin -> (0,0)
    W, H = mx - ox, my - oy
    if max(W, H) < 1e-6:
        return None
    wd = float(plan.get("wall_depth") or 4.0)

    walls = walls_from_band(band_parts, wd, ox, oy, H, W)
    walls = _manhattan_snap(walls, max(1.0, wd * 0.35))
    if len(walls) < 3:
        return None

    rooms = []                                            # [(role, polygon_pts)]
    for layer, role in ROOM_LAYERS:
        for g in _geoms(plan.get(layer)):
            if g.geom_type != "Polygon":
                continue
            pts = [(x - ox, y - oy) for x, y in list(g.exterior.coords)[:-1]]
            if len(pts) >= 3:
                rooms.append((role, pts))
    if not rooms:
        return None

    openings = []                                         # [(p0, p1, kind)]
    for g, kind in opening_parts:
        span = _opening_span(g)
        if span is None:
            continue
        (x0, y0), (x1, y1) = span
        openings.append(((x0 - ox, y0 - oy), (x1 - ox, y1 - oy), kind))

    gt = to_vlm_json(walls, rooms, openings, H, W)
    for r in gt["rooms"]:                                 # unlabeled faces -> "room"
        if r["_role"] == "undefined":
            r["_role"], r["label"] = "room", "ROOM"
    gt["meta"] = {"source": "resplan", "license": "MIT", "frame": FRAME,
                  "resplan_id": int(plan["id"])}
    roles = [r.pop("_role") for r in gt["rooms"]]
    return gt, roles


def main():
    import argparse, json, pickle
    from pathlib import Path
    from fpeval import from_vlm_json, verify
    import fpseq
    from fpseq import schema
    ap = argparse.ArgumentParser(prog="python -m resplan_fp.convert")
    ap.add_argument("--pkl", required=True, help="path to ResPlan.pkl (from the ResPlan release)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.1)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    with open(a.pkl, "rb") as f:
        plans = pickle.load(f)
    if a.limit:
        plans = plans[:a.limit]
    rows, rej = [], {}
    for p in plans:
        try:
            r = convert_one(p)
        except Exception as e:
            rej[f"parse:{type(e).__name__}"] = rej.get(f"parse:{type(e).__name__}", 0) + 1; continue
        if r is None:
            rej["empty"] = rej.get("empty", 0) + 1; continue
        gt, roles = r
        plan = from_vlm_json(gt, frame=schema.DEFAULT_FRAME)
        v = verify(plan)
        if not v.valid:                       # QA: structural validity (NOT watertight)
            rej["invalid"] = rej.get("invalid", 0) + 1; continue
        try:
            ev = fpseq.plan_to_events(plan, roles=roles)
            t = fpseq.events_to_tensors(ev)
            if len(t["token_type"]) >= schema.SEQ_LEN or len(plan.walls) > schema.MAX_WALLS:
                rej["over_capacity"] = rej.get("over_capacity", 0) + 1; continue
        except Exception:
            rej["codec"] = rej.get("codec", 0) + 1; continue
        n_open = sum(len(w["openings"]) for w in gt["walls"])
        rows.append({"id": f"resplan/{gt['meta']['resplan_id']}", "gt": gt,
                     "roles": roles, "frame": schema.DEFAULT_FRAME,
                     "n_walls": len(plan.walls), "n_rooms": len(plan.rooms),
                     "n_openings": n_open, "watertight": v.watertight,
                     "r_val": v.r_val, "source": "resplan"})
    rows.sort(key=lambda r: r["id"])
    stride = max(2, round(1 / a.val_frac))
    tr = [r for i, r in enumerate(rows) if i % stride]; va = [r for i, r in enumerate(rows) if not i % stride]
    for name, rs in (("train.jsonl", tr), ("val.jsonl", va)):
        (out / name).write_text("\n".join(json.dumps(r) for r in rs) + "\n")
    n = max(1, len(rows))
    stats = {"seen": len(plans), "accepted": len(rows), "train": len(tr), "val": len(va),
             "watertight_frac": round(sum(r["watertight"] for r in rows) / n, 3),
             "r_val_mean": round(sum(r["r_val"] for r in rows) / n, 3),
             "mean_walls": round(sum(r["n_walls"] for r in rows) / n, 2),
             "mean_rooms": round(sum(r["n_rooms"] for r in rows) / n, 2),
             "mean_openings": round(sum(r["n_openings"] for r in rows) / n, 2),
             "rejects": rej}
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
