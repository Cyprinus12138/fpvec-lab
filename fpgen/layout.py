"""Spatial realization: program tree -> room polygons -> wall graph.

Recursive proportional guillotine splits over a rectangular envelope
(rectangular, Manhattan layouts only). Rooms tile the envelope exactly, so shared cell edges ARE wall
centerlines. Every room cycle closes by construction EXCEPT open-plan LDK
boundaries, which are deliberately left wall-less (an open edge on the cycle);
watertightness there is region-based (the polygons still tile), not wall-based.

Rejection philosophy: any constraint violation raises LayoutReject; the
caller re-draws with a fresh sub-seed, bounded, and logs the cause.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from shapely.geometry import LineString, Polygon, Point as ShPoint
from shapely.ops import unary_union
from shapely.prepared import prep

from .meta import UnitMeta
from .programs import Carve, Program, Row, RoomSpec, all_leaves
from .scene import GRID_MM, Room, ScenePlan, Wall, snap

MIN_SEG_MM = 150.0
SIDE_EPS_MM = 30.0

# thickness sampling ranges (mm) per wall class; one draw per plan
THICKNESS_RANGES = {
    "exterior": (200, 300),
    "interior": (100, 150),
    "partition": (75, 100),
    "shelter": (250, 400),
}
PARTITION_ROLES = {"wic", "store", "aircon_ledge", "wardrobe"}

# open-plan: shared boundaries among these roles may be left WALL-LESS (open
# LDK), so the room cycle carries an open (non-wall) edge. The room polygons
# still tile the envelope exactly — watertightness stays region-based, not
# "every room edge is a wall" (fpeval verify polygon path). Strictly OPT-IN via
# meta.flags["open_plan"] (default off): a probabilistic mix is a caller/config
# concern, so the default path draws no rng and stays byte-identical to before
# this feature — the reserved-seed frozen eval tiers reproduce exactly.
OPEN_PLAN_ROLES = {"living_dining", "kitchen"}


class LayoutReject(Exception):
    """Sampled layout violates a constraint; caller re-draws."""

    def __init__(self, cause: str):
        super().__init__(cause)
        self.cause = cause


@dataclass
class LayoutResult:
    plan: ScenePlan
    # wall_id -> (room key left of direction, room key right); None = outside
    adjacency: dict[str, tuple[str | None, str | None]]
    specs: dict[str, RoomSpec]
    program: Program
    room_key_to_id: dict[str, str]
    thicknesses: dict[str, float]
    # unordered room-key pairs whose shared boundary is OPEN (no wall) —
    # open-plan LDK. Passable-by-construction for reachability.
    open_adjacency: set = field(default_factory=set)


# ---------------------------------------------------------------------------
# splitting
# ---------------------------------------------------------------------------

Rect = tuple[float, float, float, float]  # x0, y0, x1, y1


def _rect_dims(r: Rect) -> tuple[float, float]:
    return r[2] - r[0], r[3] - r[1]


def _weight(node: Row | RoomSpec) -> float:
    if isinstance(node, RoomSpec):
        return node.nominal_sqm + sum(c.spec.nominal_sqm for c in node.carves)
    w = sum(_weight(c) for c in node.children)
    if node.corridor is not None:
        w += node.corridor.nominal_sqm
    return w


def _min_need(node: Row | RoomSpec) -> float:
    """Conservative minimum strip width a node needs (mm).

    A Row's children each span the full cross dimension of its strip, so the
    Row needs at least the largest child requirement; a corridor row
    additionally spends corridor depth before splitting the rest.
    """
    if isinstance(node, RoomSpec):
        return node.min_dim_mm
    need = max((_min_need(c) for c in node.children), default=900.0)
    if node.corridor is not None:
        need += 1350.0
    return need


def _split_along(rect: Rect, axis: str, children: list[Row | RoomSpec],
                 rng: np.random.Generator) -> list[Rect]:
    """Cut rect into strips along axis, one per child, proportional to child
    area weights but clamped so no strip is narrower than the child's minimum
    need (proportional-only splits give 0.7 m slivers for small rooms)."""
    x0, y0, x1, y1 = rect
    span = (x1 - x0) if axis == "x" else (y1 - y0)
    weights = np.array([_weight(c) for c in children], dtype=float)
    weights *= rng.uniform(0.92, 1.08, size=len(weights))
    widths = weights / weights.sum() * span
    mins = np.array([_min_need(c) + 80.0 for c in children])
    if mins.sum() > span:
        raise LayoutReject("span_too_small")
    for _ in range(6):
        deficit = np.clip(mins - widths, 0, None).sum()
        if deficit <= 0:
            break
        widths = np.maximum(widths, mins)
        flexible = widths > mins
        slack = (widths - mins)[flexible].sum()
        if slack <= 0:
            break
        widths[flexible] -= (widths - mins)[flexible] / slack * deficit
    widths = np.maximum(widths, mins)
    widths *= span / widths.sum()

    out: list[Rect] = []
    cur = x0 if axis == "x" else y0
    end = x1 if axis == "x" else y1
    for i, w in enumerate(widths):
        nxt = end if i == len(widths) - 1 else snap(cur + w)
        if axis == "x":
            out.append((cur, y0, nxt, y1))
        else:
            out.append((x0, cur, x1, nxt))
        cur = nxt
    return out


def _check_leaf_cell(spec: RoomSpec, rect: Rect) -> None:
    w, h = _rect_dims(rect)
    if w < spec.min_dim_mm or h < spec.min_dim_mm:
        raise LayoutReject(f"min_dim:{spec.key}")
    if max(w, h) / max(min(w, h), 1.0) > spec.aspect_max:
        raise LayoutReject(f"aspect:{spec.key}")


def _assign(node: Row | RoomSpec, rect: Rect, facing: str | None,
            cells: dict[str, Rect], rng: np.random.Generator,
            parent_axis: str | None = None) -> None:
    """Recursively assign cells to leaves. facing = side of `rect` toward the
    previous sibling ('left'/'right'/'top'/'bottom')."""
    if isinstance(node, RoomSpec):
        _check_leaf_cell(node, rect)
        cells[node.key] = rect
        return

    x0, y0, x1, y1 = rect
    if node.corridor is not None:
        if facing is None:
            facing = "left" if (x1 - x0) >= (y1 - y0) else "top"
        depth = snap(rng.uniform(950, 1350))
        if facing == "left":
            corr = (x0, y0, snap(x0 + depth), y1)
            rest: Rect = (corr[2], y0, x1, y1)
            row_axis = "y"
        elif facing == "right":
            corr = (snap(x1 - depth), y0, x1, y1)
            rest = (x0, y0, corr[0], y1)
            row_axis = "y"
        elif facing == "top":
            corr = (x0, y0, x1, snap(y0 + depth))
            rest = (x0, corr[3], x1, y1)
            row_axis = "x"
        else:
            corr = (x0, snap(y1 - depth), x1, y1)
            rest = (x0, y0, x1, corr[1])
            row_axis = "x"
        _check_leaf_cell(node.corridor, corr)
        cells[node.corridor.key] = corr
        strips = _split_along(rest, row_axis, node.children, rng)
        for child, strip in zip(node.children, strips):
            # children of a corridor row: nested rows split along the same
            # axis as the row so every leaf keeps corridor contact
            _assign(child, strip, facing, cells, rng, parent_axis=row_axis)
        return

    w, h = _rect_dims(rect)
    preferred = parent_axis or ("x" if w >= h else "y")
    axes = [preferred, "y" if preferred == "x" else "x"]
    last_err: LayoutReject | None = None
    for axis in axes:
        trial: dict[str, Rect] = {}
        try:
            strips = _split_along(rect, axis, node.children, rng)
            # order children from the facing end so first child touches the
            # previous sibling (e.g. kitchen stays against living)
            if facing in ("right", "bottom") and (
                    (axis == "x" and facing == "right") or (axis == "y" and facing == "bottom")):
                strips = strips[::-1]
            for i, (child, strip) in enumerate(zip(node.children, strips)):
                child_facing = facing
                if i > 0:
                    child_facing = {"x": "left", "y": "top"}[axis]
                _assign(child, strip, child_facing, trial, rng, parent_axis=None)
            cells.update(trial)
            return
        except LayoutReject as e:
            last_err = e
            continue
    raise last_err or LayoutReject("split")


# ---------------------------------------------------------------------------
# carves (balcony / AC ledge / WIC): cell -> host polygon + carve polygon
# ---------------------------------------------------------------------------


def _exterior_sides(rect: Rect, env: Rect) -> list[str]:
    x0, y0, x1, y1 = rect
    ex0, ey0, ex1, ey1 = env
    sides = []
    if abs(x0 - ex0) < 1:
        sides.append("left")
    if abs(x1 - ex1) < 1:
        sides.append("right")
    if abs(y0 - ey0) < 1:
        sides.append("top")
    if abs(y1 - ey1) < 1:
        sides.append("bottom")
    return sides


def _carve_rect(host: Rect, side: str, depth: float, frac: float,
                rng: np.random.Generator) -> Rect:
    x0, y0, x1, y1 = host
    w, h = _rect_dims(host)
    if side in ("left", "right"):
        span = h * frac
        off = snap(rng.uniform(0, h - span))
        # keep carve corners off the host corners or exactly on them
        if off < MIN_SEG_MM:
            off = 0.0
        if (h - span - off) < MIN_SEG_MM:
            off = h - span
        cy0, cy1 = y0 + off, y0 + snap(off + span)
        if side == "left":
            return (x0, cy0, snap(x0 + depth), cy1)
        return (snap(x1 - depth), cy0, x1, cy1)
    span = w * frac
    off = snap(rng.uniform(0, w - span))
    if off < MIN_SEG_MM:
        off = 0.0
    if (w - span - off) < MIN_SEG_MM:
        off = w - span
    cx0, cx1 = x0 + off, x0 + snap(off + span)
    if side == "top":
        return (cx0, y0, cx1, snap(y0 + depth))
    return (cx0, snap(y1 - depth), cx1, y1)


def _apply_carves(spec: RoomSpec, rect: Rect, env: Rect,
                  polygons: dict[str, Polygon], rng: np.random.Generator) -> None:
    host_poly = Polygon([(rect[0], rect[1]), (rect[2], rect[1]),
                         (rect[2], rect[3]), (rect[0], rect[3])])
    used_sides: set[str] = set()
    placed_carves: list[Polygon] = []
    for carve in spec.carves:
        if carve.edge == "exterior":
            sides = _exterior_sides(rect, env)
            if not sides:
                raise LayoutReject(f"carve_no_exterior:{carve.spec.key}")
        else:
            sides = ["left", "right", "top", "bottom"]
        free_sides = [s for s in sides if s not in used_sides]
        if not free_sides:
            raise LayoutReject(f"carve_side_clash:{carve.spec.key}")
        cpoly = None
        for _ in range(8):  # carves from the same host must not overlap
            side = free_sides[rng.integers(len(free_sides))]
            depth = snap(rng.uniform(*carve.depth_mm))
            frac = float(rng.uniform(*carve.width_frac))
            crect = _carve_rect(rect, side, depth, frac, rng)
            cw, ch = _rect_dims(crect)
            if cw < carve.spec.min_dim_mm or ch < carve.spec.min_dim_mm:
                continue
            cand = Polygon([(crect[0], crect[1]), (crect[2], crect[1]),
                            (crect[2], crect[3]), (crect[0], crect[3])])
            if any(cand.intersection(prev).area > 1.0 for prev in placed_carves):
                continue
            cpoly = cand
            used_sides.add(side)
            break
        if cpoly is None:
            raise LayoutReject(f"carve_overlap:{carve.spec.key}")
        placed_carves.append(cpoly)
        host_poly = host_poly.difference(cpoly)
        if host_poly.geom_type != "Polygon" or host_poly.is_empty:
            raise LayoutReject(f"carve_split_host:{spec.key}")
        hw = host_poly.bounds
        if (hw[2] - hw[0]) < spec.min_dim_mm or (hw[3] - hw[1]) < spec.min_dim_mm:
            raise LayoutReject(f"carve_host_min_dim:{spec.key}")
        polygons[carve.spec.key] = cpoly
    polygons[spec.key] = host_poly


def _corner_owner(polygons: dict[str, Polygon], corner: tuple) -> str | None:
    owners = [k for k, p in polygons.items()
              if any(math.dist(c, corner) <= 1.0 for c in p.exterior.coords)]
    return owners[0] if len(owners) == 1 else None


def _notch_one_corner(polygons: dict[str, Polygon], env: Rect,
                      specs: dict, rng: np.random.Generator) -> None:
    """Cut a rect from one corner room -> L-shaped unit footprint."""
    corners = [(env[0], env[1]), (env[2], env[1]), (env[0], env[3]),
               (env[2], env[3])]
    order = rng.permutation(len(corners))
    for ci in order:
        corner = corners[int(ci)]
        key = _corner_owner(polygons, corner)
        if key is None:
            continue
        poly = polygons[key]
        x0, y0, x1, y1 = poly.bounds
        w, h = x1 - x0, y1 - y0
        nw = snap(rng.uniform(0.15, 0.4) * w)
        nh = snap(rng.uniform(0.15, 0.4) * h)
        if min(nw, nh) < 600:
            continue
        cx = corner[0] if corner[0] == x0 else corner[0] - nw
        cy = corner[1] if corner[1] == y0 else corner[1] - nh
        notch = Polygon([(cx, cy), (cx + nw, cy), (cx + nw, cy + nh),
                         (cx, cy + nh)])
        cand = poly.difference(notch)
        if cand.geom_type != "Polygon" or cand.is_empty:
            continue
        bx0, by0, bx1, by1 = cand.bounds
        spec = specs.get(key)
        min_dim = spec.min_dim_mm if spec else 1200.0
        if (bx1 - bx0) < min_dim or (by1 - by0) < min_dim:
            continue
        # remaining L-arm widths must also stay usable
        if (w - nw) < max(900.0, min_dim * 0.5) or (h - nh) < max(900.0, min_dim * 0.5):
            continue
        polygons[key] = cand
        return


def _chamfer_one_corner(polygons: dict[str, Polygon], env: Rect,
                        rng: np.random.Generator) -> None:
    """Cut a diagonal across one envelope corner of whichever room owns it."""
    corners = [(env[0], env[1]), (env[2], env[1]), (env[0], env[3]),
               (env[2], env[3])]
    order = rng.permutation(len(corners))
    for ci in order:
        corner = corners[int(ci)]
        owners = [k for k, p in polygons.items()
                  if any(math.dist(c, corner) <= 1.0
                         for c in p.exterior.coords)]
        if len(owners) != 1:
            continue  # shared corners (carve edge cases) can't chamfer cleanly
        for key, poly in ((owners[0], polygons[owners[0]]),):
            coords = list(poly.exterior.coords)[:-1]
            for i, (x, y) in enumerate(coords):
                if math.dist((x, y), corner) > 1.0:
                    continue
                prev = coords[i - 1]
                nxt = coords[(i + 1) % len(coords)]
                lp, ln = math.dist(prev, (x, y)), math.dist(nxt, (x, y))
                cmax = min(1500.0, 0.4 * lp, 0.4 * ln)
                if cmax < 500.0:
                    continue
                c = snap(rng.uniform(500.0, cmax))
                p1 = (x + (prev[0] - x) / lp * c, y + (prev[1] - y) / lp * c)
                p2 = (x + (nxt[0] - x) / ln * c, y + (nxt[1] - y) / ln * c)
                new = coords[:i] + [p1, p2] + coords[i + 1:]
                cand = Polygon(new)
                if cand.is_valid and cand.area > 0:
                    polygons[key] = cand
                    return


# ---------------------------------------------------------------------------
# wall graph extraction
# ---------------------------------------------------------------------------


def _segments_from_rings(polygons: dict[str, Polygon]) -> list[tuple[tuple, tuple]]:
    """Node all room boundaries against each other, explode to 2-pt segments."""
    rings = [p.exterior for p in polygons.values()]
    noded = unary_union(rings)
    geoms = getattr(noded, "geoms", [noded])
    segs = []
    for line in geoms:
        coords = list(line.coords)
        for a, b in zip(coords, coords[1:]):
            if math.dist(a, b) > 1e-6:
                segs.append((tuple(a), tuple(b)))
    return segs


def _side_rooms(seg: tuple[tuple, tuple], prepared: list[tuple[str, object]],
                ) -> tuple[str | None, str | None]:
    (ax, ay), (bx, by) = seg
    mx, my = (ax + bx) / 2, (ay + by) / 2
    length = math.dist((ax, ay), (bx, by))
    nx, ny = -(by - ay) / length, (bx - ax) / length
    left = ShPoint(mx - nx * SIDE_EPS_MM, my - ny * SIDE_EPS_MM)
    right = ShPoint(mx + nx * SIDE_EPS_MM, my + ny * SIDE_EPS_MM)
    lkey = rkey = None
    for key, pp in prepared:
        if lkey is None and pp.contains(left):
            lkey = key
        if rkey is None and pp.contains(right):
            rkey = key
        if lkey and rkey:
            break
    return lkey, rkey


def _wall_class(lkey: str | None, rkey: str | None,
                specs: dict[str, RoomSpec]) -> str:
    roles = {specs[k].role for k in (lkey, rkey) if k is not None}
    # shelter walls are structurally thick on ALL sides, envelope included
    if "household_shelter" in roles:
        return "shelter"
    if lkey is None or rkey is None:
        return "exterior"
    if roles & PARTITION_ROLES:
        return "partition"
    return "interior"


def _merge_collinear(
    segs: list[tuple[tuple, tuple, tuple]],
) -> list[tuple[tuple, tuple, tuple]]:
    """Merge touching collinear segments with identical (adjacency, class).

    Each item: (a, b, key) where key = (lkey, rkey, cls). Manhattan-only.
    """
    out = []
    by_line: dict[tuple, list[tuple[tuple, tuple, tuple]]] = {}
    for a, b, key in segs:
        if abs(a[0] - b[0]) < 1e-6:  # vertical
            lo, hi = sorted([a, b], key=lambda p: p[1])
        elif abs(a[1] - b[1]) < 1e-6:  # horizontal
            lo, hi = sorted([a, b], key=lambda p: p[0])
        else:  # non-Manhattan: pass through unmerged
            out.append((a, b, key))
            continue
        if (lo, hi) != (a, b):  # direction flipped -> left/right rooms swap
            key = (key[1], key[0], key[2])
        line_id = ("v" if abs(a[0] - b[0]) < 1e-6 else "h",
                   round(lo[0] if abs(a[0] - b[0]) < 1e-6 else lo[1], 3), key)
        by_line.setdefault(line_id, []).append((lo, hi, key))
    for (orient, _, key), items in by_line.items():
        idx = 1 if orient == "v" else 0
        items.sort(key=lambda it: it[0][idx])
        cur_lo, cur_hi = items[0][0], items[0][1]
        for lo, hi, _ in items[1:]:
            if abs(lo[idx] - cur_hi[idx]) < 1e-6:
                cur_hi = hi
            else:
                out.append((cur_lo, cur_hi, key))
                cur_lo, cur_hi = lo, hi
        out.append((cur_lo, cur_hi, key))
    return out


# ---------------------------------------------------------------------------
# main entry
# ---------------------------------------------------------------------------


def realize(program: Program, meta: UnitMeta,
            rng: np.random.Generator) -> LayoutResult:
    """One attempt: program -> ScenePlan (walls + rooms, no openings yet)."""
    leaves = all_leaves(program.root)
    specs = {s.key: s for s in leaves}

    env_area_mm2 = meta.target_area_sqm * 1e6
    aspect = float(rng.uniform(1.0, 2.2))
    env_w = snap(math.sqrt(env_area_mm2 * aspect))
    env_h = snap(env_area_mm2 / env_w)
    env: Rect = (0.0, 0.0, env_w, env_h)

    cells: dict[str, Rect] = {}
    _assign(program.root, env, None, cells, rng)

    polygons: dict[str, Polygon] = {}
    for key, rect in cells.items():
        spec = specs[key]
        if spec.carves:
            _apply_carves(spec, rect, env, polygons, rng)
        else:
            polygons[key] = Polygon([(rect[0], rect[1]), (rect[2], rect[1]),
                                     (rect[2], rect[3]), (rect[0], rect[3])])

    # envelope variety passes:
    # - notch (p~0.2): cut a rect out of one corner room -> L-shaped plan
    # - chamfer (p~0.22): diagonal across one envelope corner (non-Manhattan —
    #   segments pass through merge unmerged, renderer buffers any angle)
    r = rng.random()
    if r < 0.20:
        _notch_one_corner(polygons, env, specs, rng)
    elif r < 0.42:
        _chamfer_one_corner(polygons, env, rng)

    # hard adjacency requirements (>=600 mm shared boundary)
    for a, b in program.hard_adjacent:
        alts = b.split("|")
        pa = polygons.get(a)
        if pa is None:
            continue
        ok = False
        for alt in alts:
            pb = polygons.get(alt)
            if pb is None:
                continue
            shared = pa.boundary.intersection(pb.boundary)
            if shared.length >= 600:
                ok = True
                break
        if not ok:
            raise LayoutReject(f"adjacency:{a}~{b}")

    # walls
    thicknesses = {cls: float(snap(rng.uniform(*rr), 5))
                   for cls, rr in THICKNESS_RANGES.items()}
    prepared = [(key, prep(poly)) for key, poly in polygons.items()]
    raw = []
    for seg in _segments_from_rings(polygons):
        lkey, rkey = _side_rooms(seg, prepared)
        if lkey is None and rkey is None:
            raise LayoutReject("orphan_segment")
        cls = _wall_class(lkey, rkey, specs)
        raw.append((seg[0], seg[1], (lkey, rkey, cls)))
    merged = _merge_collinear(raw)

    merged.sort(key=lambda it: (min(it[0][1], it[1][1]), min(it[0][0], it[1][0]),
                                it[0], it[1]))

    # open-plan: leave the LDK shared boundaries wall-less. The room polygons
    # (built below from the tiling) are untouched, so those room cycles simply
    # carry an open edge — no phantom wall is minted.
    open_plan = bool(meta.flags.get("open_plan", False))  # opt-in; no default draw
    open_adjacency: set = set()

    def _is_open(lkey, rkey) -> bool:
        return (open_plan and lkey is not None and rkey is not None
                and specs[lkey].role in OPEN_PLAN_ROLES
                and specs[rkey].role in OPEN_PLAN_ROLES)

    walls: list[Wall] = []
    adjacency: dict[str, tuple[str | None, str | None]] = {}
    wi = 0
    for a, b, (lkey, rkey, cls) in merged:
        if math.dist(a, b) < MIN_SEG_MM:
            raise LayoutReject("short_segment")
        if _is_open(lkey, rkey):
            open_adjacency.add(frozenset((lkey, rkey)))
            continue
        wi += 1
        wid = f"wall_{wi}"
        walls.append(Wall(id=wid, start=a, end=b,
                          thickness_mm=thicknesses[cls], role=cls))  # type: ignore[arg-type]
        adjacency[wid] = (lkey, rkey)

    # rooms with ordered wall cycles
    rooms: list[Room] = []
    room_key_to_id: dict[str, str] = {}
    wall_mid = {w.id: ((w.start[0] + w.end[0]) / 2, (w.start[1] + w.end[1]) / 2)
                for w in walls}
    for j, (key, poly) in enumerate(sorted(polygons.items()), start=1):
        ring = poly.exterior
        my_walls = [w.id for w in walls if key in adjacency[w.id]]
        my_walls.sort(key=lambda wid: ring.project(ShPoint(wall_mid[wid])))
        rid = f"room_{j}"
        room_key_to_id[key] = rid
        rooms.append(Room(
            id=rid, role=specs[key].role, label="",  # labels assigned later
            polygon=[(x, y) for x, y in list(ring.coords)[:-1]],
            wall_ids=my_walls,
            area_sqm=round(poly.area / 1e6, 1),
        ))

    plan = ScenePlan(
        bbox_mm=(env_w, env_h),
        walls=walls,
        rooms=rooms,
        meta={"unit_meta": {"typology": meta.typology,
                            "n_bed": meta.n_bed, "n_bath": meta.n_bath,
                            "target_area_sqm": meta.target_area_sqm,
                            "flags": meta.flags}},
    )
    return LayoutResult(plan=plan, adjacency=adjacency, specs=specs,
                        program=program, room_key_to_id=room_key_to_id,
                        thicknesses=thicknesses, open_adjacency=open_adjacency)


def generate_layout(program: Program, meta: UnitMeta,
                    seed_seq: np.random.SeedSequence,
                    max_tries: int = 80) -> tuple[LayoutResult, dict[str, int]]:
    """Bounded rejection-sampling wrapper. Returns (result, reject counts)."""
    rejects: dict[str, int] = {}
    for child_seq in seed_seq.spawn(max_tries):
        rng = np.random.default_rng(child_seq)
        try:
            return realize(program, meta, rng), rejects
        except LayoutReject as e:
            rejects[e.cause] = rejects.get(e.cause, 0) + 1
    raise LayoutReject(f"exhausted:{max_tries}:{rejects}")
