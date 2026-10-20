"""CubiCasa5k SVG -> wall-first floorplan JSON (the skv4 converter; research/NC-only).

Parses cc5k `model.svg` (reusing the vendored svg_utils.PolygonWall for wall
centerlines + width; own lean parse for rooms + openings, skipping the icon path
that crashes on new numpy), then assembles our schema:

  walls (centerline + thickness + type) → rooms (wall-ID cycles + polygon) →
  openings (on-wall offset + width + class),

welds wall endpoints into shared junctions, assigns each room polygon to its
bounding walls, projects doors/windows onto walls, and rescales to the 1024
fpeval frame. Output is QA'd through fpeval.verify + the fpseq codec round-trip.

cc5k is CC-BY-NC-SA -> research/paper only, NEVER ship-path.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from xml.dom import minidom

import numpy as np

from .svg_utils import PolygonWall, get_points, get_direction  # vendored, see header there

FRAME = 1024.0


def walls_from_skeleton(svg_path, H, W):
    """Clean vector walls from the wall SEGMENTATION mask (the accurate cc5k GT),
    not from room polygons. Rasterize the Wall/Railing polygons -> morphological
    close -> skeletonize -> sknw graph (nodes = junctions, edges = wall runs) ->
    Douglas-Peucker simplify each run into straight segments. Yields SINGLE
    centerlines with correct extent: no room-derived double walls, no raw-polygon
    overshoots. Returns [(p0, p1, thickness, type)] in SVG coords."""
    import cv2
    from skimage.morphology import skeletonize
    import sknw
    d = minidom.parse(svg_path)
    mask = np.zeros((H, W), np.uint8)
    widths, wid = [], 0
    for e in d.getElementsByTagName("g"):
        if e.getAttribute("id") in ("Wall", "Railing"):
            try:
                w = PolygonWall(e, wid, (H, W)); wid += 1
            except ValueError:
                continue
            poly = np.array([[int(x), int(y)] for x, y in zip(w.X, w.Y)], np.int32)
            if len(poly) >= 3:
                cv2.fillPoly(mask, [poly], 1); widths.append(float(w.max_width))
    if mask.sum() == 0:
        return []
    thickness = float(np.median(widths)) if widths else 10.0
    k = max(3, int(round(thickness * 0.5)) | 1)          # close gaps up to ~½ thickness
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    graph = sknw.build_sknw(skeletonize(mask > 0))
    eps = max(H, W) * 0.006
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
            if math.dist(a, b) > thickness * 0.6:
                segs.append(((float(a[0]), float(a[1])),
                             (float(b[0]), float(b[1])), thickness, "interior"))
    return segs


def _label_ext_int(walls, rooms, s, thickness):
    """Label each wall exterior (perimeter: one side is background) vs interior
    (between two rooms), by testing perpendicular offsets against the room union."""
    from shapely.geometry import Polygon as SPoly, Point
    from shapely.ops import unary_union
    polys = [SPoly([(x * s, y * s) for x, y in poly]) for _r, poly in rooms if len(poly) >= 3]
    polys = [p for p in polys if p.is_valid and not p.is_empty]
    if not polys:
        return walls
    union = unary_union(polys).buffer(thickness * 0.5)
    off = thickness * 1.3
    out = []
    for a, b, w, _t in walls:
        mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        dx, dy = b[0] - a[0], b[1] - a[1]; L = math.hypot(dx, dy) or 1
        nx, ny = -dy / L, dx / L
        in1 = union.contains(Point(mx + nx * off, my + ny * off))
        in2 = union.contains(Point(mx - nx * off, my - ny * off))
        out.append((a, b, w, "exterior" if in1 != in2 else "interior"))
    return out


def _room_role(cls: str) -> str:
    """'Space Kitchen' / 'Space LivingRoom' -> our role vocabulary (best-effort)."""
    t = cls.replace("Space", "").strip().lower()
    m = {"livingroom": "living", "diningroom": "dining", "kitchen": "kitchen",
         "bedroom": "bedroom", "bath": "bath", "bathroom": "bath", "toilet": "wc",
         "entry": "foyer", "lobby": "foyer", "hall": "corridor", "corridor": "corridor",
         "storage": "store", "closet": "wic", "outdoor": "balcony", "balcony": "balcony",
         "garage": "utility", "utility": "utility", "undefined": "corridor"}
    for k, v in m.items():
        if k in t:
            return v
    return "corridor"


def _svg_dims(svg) -> tuple[int, int]:
    """(H, W) in the SVG's OWN coordinate space (viewBox), NOT the PNG pixels."""
    el = svg.getElementsByTagName("svg")[0]
    vb = el.getAttribute("viewBox")
    if vb:
        _, _, w, h = (float(x) for x in vb.split())
    else:
        w = float(el.getAttribute("width")); h = float(el.getAttribute("height"))
    return int(math.ceil(h)), int(math.ceil(w))


def parse_svg(svg_path: str):
    """-> (walls, rooms, openings, (H, W)) in SVG coordinate space.
    walls: [(p0, p1, width, wtype)]; rooms: [(role, polygon)]; openings: [(p0,p1,kind)]"""
    svg = minidom.parse(svg_path)
    H, W = _svg_dims(svg)
    walls, rooms, openings = [], [], []
    wid = 0
    for e in svg.getElementsByTagName("g"):
        eid, cls = e.getAttribute("id"), e.getAttribute("class")
        if eid in ("Wall", "Railing"):
            try:
                w = PolygonWall(e, wid, (H, W)); wid += 1
            except ValueError:
                continue
            (x0, y0), (x1, y1) = w.end_points
            wtype = "exterior" if "External" in cls else "interior"
            walls.append(((float(x0), float(y0)), (float(x1), float(y1)),
                          float(w.max_width), wtype))
        elif eid in ("Window", "Door"):
            X, Y = get_points(e)
            locs = np.column_stack((X, Y))
            d = get_direction(X, Y)
            axis = 0 if d == "H" else 1
            order = locs[locs[:, axis].argsort()]
            p0 = order[:2].mean(0); p1 = order[2:].mean(0)   # two ends of the opening
            openings.append(((float(p0[0]), float(p0[1])), (float(p1[0]), float(p1[1])),
                             "window" if eid == "Window" else "door"))
        elif cls.startswith("Space "):
            for c in e.childNodes:
                if getattr(c, "tagName", None) == "polygon":
                    pts = [tuple(map(float, p.split(","))) for p in c.getAttribute("points").split()]
                    if len(pts) >= 3:
                        rooms.append((_room_role(cls), pts))
                    break
    return walls, rooms, openings, (H, W)


def _weld(walls, tol):
    """Snap nearby wall endpoints to shared junction centroids (in place-ish)."""
    pts = []
    for p0, p1, *_ in walls:
        pts += [p0, p1]
    cl: list[list] = []
    for p in pts:
        for c in cl:
            if math.dist(c[0], p) <= tol:
                c.append(p); break
        else:
            cl.append([p])
    cent = {}
    for c in cl:
        cx = sum(p[0] for p in c) / len(c); cy = sum(p[1] for p in c) / len(c)
        for p in c:
            cent[p] = (cx, cy)
    return [(cent[p0], cent[p1], wdt, wt) for p0, p1, wdt, wt in walls]


def _seg_overlap_on_edge(wp0, wp1, a, b, tol):
    """True if wall segment (wp0,wp1) lies along polygon edge (a,b): collinear +
    projection overlap."""
    ex, ey = b[0] - a[0], b[1] - a[1]
    L = math.hypot(ex, ey)
    if L < 1e-6:
        return False
    ux, uy = ex / L, ey / L

    def foot(p):
        return (p[0] - a[0]) * ux + (p[1] - a[1]) * uy

    def perp(p):
        return abs(-(p[0] - a[0]) * uy + (p[1] - a[1]) * ux)

    if perp(wp0) > tol or perp(wp1) > tol:
        return False
    t0, t1 = sorted((foot(wp0), foot(wp1)))
    return min(t1, L) - max(t0, 0.0) > 0.25 * min(L, math.dist(wp0, wp1) + 1e-6)


def to_vlm_json(walls, rooms, openings, H, W, weld_frac=0.02):
    s = FRAME / max(H, W)
    tol = FRAME * weld_frac
    # `walls` are SKELETON walls (from walls_from_skeleton): single centerlines
    # traced from the wall segmentation mask -> correct extent, no room-derived
    # double walls, no raw-polygon overshoots. Just scale to frame, weld
    # junctions, Manhattan-snap, drop degenerate stubs.
    thickness = (sum(w[2] for w in walls) / max(1, len(walls))) * s if walls else 10.0
    walls = [((a[0] * s, a[1] * s), (b[0] * s, b[1] * s), w * s, t) for a, b, w, t in walls]
    walls = _axis_clean(walls)
    walls = _weld(walls, max(tol, thickness * 1.2))
    walls = [w for w in walls if math.dist(w[0], w[1]) > thickness * 0.5]  # degenerate
    walls = _label_ext_int(walls, rooms, s, thickness)   # perimeter vs interior

    wall_json = [{"id": f"wall_{i+1}", "start": [round(w[0][0], 2), round(w[0][1], 2)],
                  "end": [round(w[1][0], 2), round(w[1][1], 2)],
                  "thickness": round(w[2], 2), "type": w[3], "openings": []}
                 for i, w in enumerate(walls)]

    # rooms: FACES of the welded wall graph (shapely polygonize via
    # fpeval.room_faces) -> wall-ID cycles close BY CONSTRUCTION and polygons
    # live on wall centerlines, so fpeval's region-tiling watertightness holds.
    # (The previous nearest-room partition kept inner-face polygons whose union
    # is disconnected -> GT r_val=0 on 99.4% of plans, and proximity-picked
    # wall_ids chained closed only 44% -> incoherent room supervision. Both
    # measured 2026-07-08; that room collapse was a GT artifact, not the model.)
    # Labels: max-overlap against the original cc5k room polygons; faces no
    # cc5k room claims (shafts/closets) stay 'undefined' so the tiling has no
    # holes.
    from shapely.geometry import LineString, Polygon as SPoly
    from fpeval.model import from_vlm_json as _fp_from
    from fpeval.geometry import room_faces, room_polygon_from_wall_ids

    plan = _fp_from({"walls": wall_json, "rooms": []}, frame=FRAME)
    faces = room_faces(plan)
    rooms_frame = []
    for role, poly in rooms:
        p = SPoly([(x * s, y * s) for x, y in poly])
        if not p.is_valid:
            p = p.buffer(0)
        if not p.is_empty and p.area > (FRAME * 0.01) ** 2:
            rooms_frame.append((role, p))
    rooms_json = []
    for f in faces:
        # a wall belongs to this face's cycle iff most of its length lies on
        # the face boundary (midpoint tests misfire on stubs/T-junctions)
        bufb = f.exterior.buffer(max(1.0, tol * 0.25))
        wids = []
        for wj, w in zip(wall_json, walls):
            ls = LineString([w[0], w[1]])
            if ls.length > 0 and bufb.intersection(ls).length >= 0.6 * ls.length:
                wids.append(wj["id"])
        if len(wids) < 3 or room_polygon_from_wall_ids(plan, wids) is None:
            continue                      # must chain into a closed cycle
        best_role, best_ov = None, 0.0
        for role, p in rooms_frame:
            ov = f.intersection(p).area
            if ov > best_ov:
                best_ov, best_role = ov, role
        role = best_role if best_ov > 0.3 * f.area else "undefined"
        poly_pts = list(f.simplify(0.5).exterior.coords)[:-1]
        rooms_json.append({"label": role.upper(), "walls": wids,
                           "polygon": [[round(x, 2), round(y, 2)] for x, y in poly_pts],
                           "_role": role})

    # openings -> nearest wall + along-wall offset + width
    for p0, p1, kind in openings:
        c = ((p0[0] + p1[0]) / 2 * s, (p0[1] + p1[1]) / 2 * s)
        width = math.dist(p0, p1) * s
        best, bd = None, 1e18
        for i, w in enumerate(walls):
            d = _point_seg_dist(c, w[0], w[1])
            if d < bd:
                bd, best = d, i
        if best is not None and bd <= tol * 2:
            # project the center onto the host wall's centerline (the SVG mark
            # sits on the wall band; the skeleton centerline can be ~half a
            # thickness away, which fails verify's on-wall check)
            (ax, ay), (bx, by) = walls[best][0], walls[best][1]
            ex, ey = bx - ax, by - ay
            L2 = ex * ex + ey * ey
            t = max(0.0, min(1.0, ((c[0] - ax) * ex + (c[1] - ay) * ey) / L2)) if L2 > 1e-9 else 0.0
            c = (ax + t * ex, ay + t * ey)
            wall_len = math.sqrt(L2)
            wall_json[best]["openings"].append(
                {"type": kind, "center": [round(c[0], 2), round(c[1], 2)],
                 "width": round(min(width, wall_len), 2)})

    return {"walls": wall_json, "rooms": rooms_json,
            "meta": {"source": "cubicasa5k", "license": "CC-BY-NC-SA", "frame": FRAME}}


def _axis_clean(walls, snap_deg=12.0, drop_deg=30.0):
    """Reduce walls-from-rooms noise: snap near-axis walls to exact H/V (cc5k is
    ~Manhattan; also re-centers the room-inner-face offset onto the centerline),
    and drop steep diagonals (>drop_deg from the nearest axis) — those are almost
    always nearest-room-partition / degenerate-polygon artifacts (radiating
    'star' segments around fixtures), not real walls. Walls between snap_deg and
    drop_deg are kept as-is (genuine mild angles)."""
    st = math.tan(math.radians(snap_deg))
    dt = math.tan(math.radians(drop_deg))
    out = []
    for (a, b, th, typ) in walls:
        adx, ady = abs(b[0] - a[0]), abs(b[1] - a[1])
        if adx >= ady:                       # closer to horizontal
            if ady <= st * adx:              # within snap_deg -> exact horizontal
                ym = (a[1] + b[1]) / 2
                out.append(((a[0], ym), (b[0], ym), th, typ)); continue
        else:                                # closer to vertical
            if adx <= st * ady:              # within snap_deg -> exact vertical
                xm = (a[0] + b[0]) / 2
                out.append(((xm, a[1]), (xm, b[1]), th, typ)); continue
        if min(adx, ady) > dt * max(adx, ady):   # steeper than drop_deg -> artifact
            continue
        out.append((a, b, th, typ))          # mild diagonal -> keep
    return out


def _norm(vx, vy):
    L = math.hypot(vx, vy) or 1.0
    return vx / L, vy / L


def _facing(a, b, c, d, gap):
    """Edges (a,b) and (c,d) face each other across a wall gap: near-parallel,
    perpendicular separation in (small, gap], projection overlap."""
    ux, uy = _norm(b[0] - a[0], b[1] - a[1])
    vx, vy = _norm(d[0] - c[0], d[1] - c[1])
    if abs(ux * vy - uy * vx) > 0.26:            # > ~15deg from parallel
        return False
    perp = abs(-(c[0] - a[0]) * uy + (c[1] - a[1]) * ux)
    if not (1.5 < perp <= gap):                   # across a gap, not coincident
        return False
    L = math.hypot(b[0] - a[0], b[1] - a[1])
    tc = (c[0] - a[0]) * ux + (c[1] - a[1]) * uy
    td = (d[0] - a[0]) * ux + (d[1] - a[1]) * uy
    lo, hi = max(0.0, min(tc, td)), min(L, max(tc, td))
    return hi - lo > 0.3 * min(L, math.dist(c, d))


def _midline(a, b, c, d):
    """Centerline between two facing edges (endpoint-matched)."""
    if math.dist(a, c) + math.dist(b, d) <= math.dist(a, d) + math.dist(b, c):
        return ((a[0] + c[0]) / 2, (a[1] + c[1]) / 2), ((b[0] + d[0]) / 2, (b[1] + d[1]) / 2)
    return ((a[0] + d[0]) / 2, (a[1] + d[1]) / 2), ((b[0] + c[0]) / 2, (b[1] + c[1]) / 2)


def derive_walls_from_rooms(rooms_frame, gap, thickness):
    """Clean room polygons (frame coords) -> wall segments. Interior walls =
    centerline between two facing room edges; exterior walls = unpaired edges."""
    edges = []
    for ri, (_role, poly) in enumerate(rooms_frame):
        for k in range(len(poly)):
            edges.append([ri, poly[k], poly[(k + 1) % len(poly)]])
    walls = []
    used = [False] * len(edges)
    for i, (ri, a, b) in enumerate(edges):
        if used[i]:
            continue
        pair = None
        for j in range(i + 1, len(edges)):
            rj, c, d = edges[j]
            if used[j] or rj == ri:
                continue
            if _facing(a, b, c, d, gap):
                pair = j; break
        if pair is not None:
            _, c, d = edges[pair]
            used[i] = used[pair] = True
            p0, p1 = _midline(a, b, c, d)
            walls.append((p0, p1, thickness, "interior"))
        else:
            used[i] = True
            walls.append((a, b, thickness, "exterior"))
    return walls


def _partition_rooms(rooms_frame, thickness, res: int = 2):
    """Nearest-room partition: assign every point of the building footprint to the
    nearest room polygon -> watertight tiling (boundaries at wall centerlines).
    Rasterizes at `res` px per frame unit, argmin over per-room distance
    transforms, then vectorizes + simplifies each region back to a polygon."""
    import cv2
    import numpy as np
    from scipy.ndimage import distance_transform_edt

    xs = [c[0] for _, p in rooms_frame for c in p.exterior.coords]
    ys = [c[1] for _, p in rooms_frame for c in p.exterior.coords]
    pad = thickness * 1.5
    x0, y0 = min(xs) - pad, min(ys) - pad
    W = int((max(xs) + pad - x0) * res) + 1
    H = int((max(ys) + pad - y0) * res) + 1

    def raster(poly):
        pts = np.array([[(cx - x0) * res, (cy - y0) * res] for cx, cy in poly.exterior.coords],
                       np.int32)
        m = np.zeros((H, W), np.uint8); cv2.fillPoly(m, [pts], 1); return m

    masks = [raster(p) for _, p in rooms_frame]
    footprint = np.zeros((H, W), np.uint8)
    for m in masks:
        footprint |= m
    # close the wall-band gaps so the footprint is one solid region
    k = int(thickness * res * 2) | 1
    footprint = cv2.morphologyEx(footprint, cv2.MORPH_CLOSE,
                                 cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    # per-room distance (0 inside the room, growing outward) -> argmin = nearest
    dists = np.stack([distance_transform_edt(1 - m) for m in masks])
    label = dists.argmin(0)
    out = []
    for i, (role, _p) in enumerate(rooms_frame):
        reg = ((label == i) & (footprint > 0)).astype(np.uint8)
        cnts, _ = cv2.findContours(reg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        eps = thickness * res * 0.5
        approx = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
        poly = [(float(px) / res + x0, float(py) / res + y0) for px, py in approx]
        out.append((role, poly))
    return out


def _point_seg_dist(p, a, b):
    ex, ey = b[0] - a[0], b[1] - a[1]
    L2 = ex * ex + ey * ey
    if L2 < 1e-9:
        return math.dist(p, a)
    t = max(0, min(1, ((p[0] - a[0]) * ex + (p[1] - a[1]) * ey) / L2))
    return math.dist(p, (a[0] + t * ex, a[1] + t * ey))


def _n_floors(svg) -> int:
    """Count distinct level containers (`Floorplan <Level>`: Floor-N, Basement-N,
    Attic-N, ...). cc5k multi-floor SVGs lay levels side by side; our render shows
    only one but parse_svg reads Spaces from ALL levels -> phantom walls in blank
    image regions. NOTE: must catch Basement/Attic too, not just `Floor-*`."""
    levels = set()
    for g in svg.getElementsByTagName("g"):
        toks = g.getAttribute("class").split()
        if "Floorplan" in toks:
            for t in toks:
                if t != "Floorplan":
                    levels.add(t)
    return len(levels) or 1


def _ink_and_spur_filter(walls, img, H, W, openings):
    """skv4: drop GT walls the INPUT IMAGE cannot support.
    (a) ink gate: <10% dark pixels along the centerline, sampled OUTSIDE declared
        opening spans (openings legitimately blank the band) -> phantom (invisible
        Railing polygons / render mismatches; measured 2.1% of skv3 walls, was
        teaching the model to hallucinate walls on blank canvas);
    (b) spur prune: free-ended stubs shorter than 2x thickness (skeleton spurs)."""
    import cv2
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    h2, w2 = gray.shape
    sc = max(h2, w2) / max(H, W)
    spans = []
    for a, b, _k in openings:
        spans.append((np.array(a, float), np.array(b, float)))
    kept = []
    for (a, b, th, ty) in walls:
        A = np.array(a, float); B = np.array(b, float)
        ts = np.linspace(0.02, 0.98, 25)
        pts = A[None, :] * (1 - ts[:, None]) + B[None, :] * ts[:, None]
        keep_mask = np.ones(len(pts), bool)
        for oa, ob in spans:                      # skip samples inside opening spans
            om = (oa + ob) / 2; orad = np.linalg.norm(ob - oa) / 2 + th
            keep_mask &= (np.linalg.norm(pts - om[None, :], axis=1) > orad)
        pp = pts[keep_mask]
        if len(pp) >= 5:
            ij = np.clip((pp * sc).astype(int), 0, [w2 - 1, h2 - 1])
            ink = (gray[ij[:, 1], ij[:, 0]] < 128).mean()
            if ink < 0.10:
                continue                           # (a) no image evidence
        kept.append((a, b, th, ty))
    # (b) spur prune on the survivors
    from collections import Counter
    endc = Counter()
    def key(p): return (round(p[0], 1), round(p[1], 1))
    for (a, b, _t, _y) in kept:
        endc[key(a)] += 1; endc[key(b)] += 1
    out = []
    for (a, b, th, ty) in kept:
        free = (endc[key(a)] == 1) or (endc[key(b)] == 1)
        if free and math.dist(a, b) < th * 2.0:
            continue
        out.append((a, b, th, ty))
    return out


def convert_one(svg_path: str, img=None):
    """svg -> (vlm_json, roles, img) or None if parse/convert fails."""
    from xml.dom import minidom
    if _n_floors(minidom.parse(svg_path)) > 1:
        raise ValueError("multifloor")          # phantom-wall risk; skip (~20% of cc5k)
    _rawwalls, rooms, openings, (H, W) = parse_svg(svg_path)
    walls = walls_from_skeleton(svg_path, H, W)   # clean centerlines from wall mask
    if img is None:
        img = _render_svg(svg_path)
    walls = _ink_and_spur_filter(walls, img, H, W, openings)   # skv4 gates
    if len(walls) < 3 or len(rooms) < 1:
        return None
    gt = to_vlm_json(walls, rooms, openings, H, W)
    roles = [r.pop("_role") for r in gt["rooms"]]
    return gt, roles, (H, W)


def _render_svg(svg_path, long_edge=256):
    import cairosvg, cv2, numpy as np
    png = cairosvg.svg2png(url=svg_path, output_width=long_edge)  # width-scaled
    arr = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_UNCHANGED)
    if arr.shape[2] == 4:  # composite alpha over white
        a = arr[:, :, 3:4].astype(np.float32) / 255
        arr = (arr[:, :, :3].astype(np.float32) * a + 255 * (1 - a)).astype(np.uint8)
    else:
        arr = arr[:, :, :3]
    return arr


def main():
    import argparse, json, cv2
    from pathlib import Path
    from fpeval import from_vlm_json, verify, evaluate
    import fpseq
    from fpseq import schema
    ap = argparse.ArgumentParser(prog="python -m skv4.convert")
    ap.add_argument("--svg-dir", required=True, help="dir of extracted cc5k sample dirs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--img-long", type=int, default=256)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--shard", type=int, default=0, help="this shard index (0-based)")
    ap.add_argument("--nshards", type=int, default=1, help="total shards for parallel convert")
    a = ap.parse_args()
    out = Path(a.out); (out / "images").mkdir(parents=True, exist_ok=True)
    svgs = sorted(Path(a.svg_dir).glob("**/model.svg"))
    if a.limit:
        svgs = svgs[:a.limit]
    if a.nshards > 1:                              # round-robin shard for parallelism
        svgs = svgs[a.shard::a.nshards]
    rows, rej = [], {}
    for sp in svgs:
        sid = sp.parent.name
        try:
            r = convert_one(str(sp))
        except Exception as e:
            rej[f"parse:{type(e).__name__}"] = rej.get(f"parse:{type(e).__name__}", 0) + 1; continue
        if r is None:
            rej["empty"] = rej.get("empty", 0) + 1; continue
        gt, roles, _ = r
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
        try:
            img = _render_svg(str(sp), a.img_long)
        except Exception:
            rej["render"] = rej.get("render", 0) + 1; continue
        rel = f"images/cc5k_{sid}.png"; cv2.imwrite(str(out / rel), img)
        rows.append({"id": f"cc5k/{sid}", "image": rel, "gt": gt, "roles": roles,
                     "frame": schema.DEFAULT_FRAME, "n_walls": len(plan.walls),
                     "watertight": v.watertight, "source": "cubicasa5k"})
    rows.sort(key=lambda r: r["id"])
    stride = max(2, round(1 / a.val_frac))
    tr = [r for i, r in enumerate(rows) if i % stride]; va = [r for i, r in enumerate(rows) if not i % stride]
    for name, rs in (("train.jsonl", tr), ("val.jsonl", va)):
        (out / name).write_text("\n".join(json.dumps(r) for r in rs) + "\n")
    wt = sum(r["watertight"] for r in rows)
    stats = {"seen": len(svgs), "accepted": len(rows), "train": len(tr), "val": len(va),
             "watertight_frac": round(wt / max(1, len(rows)), 3), "rejects": rej}
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
