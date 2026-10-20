"""Room-role-conditioned fixture placement solver.

Interprets the declarative rules table (rules.py): rejection sampling of
candidate poses with collision, clearance, and door-swing checks — solvers
stay dumb, constraints stay data. Placement failures of soft rules drop the
fixture (logged); hard-rule failures veto the sample.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from shapely.geometry import Point as ShPoint, Polygon

from .layout import LayoutResult
from .rules import RULES, FixtureRule, ac_unit_count
from .scene import Fixture, Opening, Room, Wall

WALL_GAP_MM = 10.0  # gap between wall inner face and fixture back
POSE_TRIES = 64


class SemanticsReject(Exception):
    def __init__(self, cause: str):
        super().__init__(cause)
        self.cause = cause


@dataclass
class FurnishStats:
    placed: int = 0
    dropped: list[str] = field(default_factory=list)
    unfurnished_rooms: list[str] = field(default_factory=list)


def _quad_to_fixture(fid: str, kind: str, room_id: str,
                     center: tuple[float, float], w: float, d: float,
                     rot_deg: float, params: dict | None = None) -> Fixture:
    return Fixture(id=fid, kind=kind, room_id=room_id,
                   rect_mm=(center[0] - w / 2, center[1] - d / 2, w, d),
                   rotation_deg=rot_deg, params=params or {})


def _fixture_poly(f: Fixture) -> Polygon:
    return Polygon(f.corners())


def _inward_normal(layout: LayoutResult, wall: Wall, room_key: str) -> tuple[float, float]:
    """Unit normal pointing from the wall centerline into the given room.

    Convention shared with layout._side_rooms: the "left" adjacency room lies
    on the -normal side of Wall.normal().
    """
    n = wall.normal()
    l, _ = layout.adjacency[wall.id]
    return (-n[0], -n[1]) if l == room_key else n


def _rot_for_inward(n: tuple[float, float]) -> float:
    """Rotation (deg) that points the fixture's local +y (front) along n."""
    return math.degrees(math.atan2(n[1], n[0])) - 90.0


def door_obstacles(layout: LayoutResult, room_key: str) -> list[Polygon]:
    """Swing sectors + approach rectangles of openings on this room's walls."""
    plan = layout.plan
    walls = plan.wall_by_id()
    rid = layout.room_key_to_id[room_key]
    room = plan.room_by_id()[rid]
    out: list[Polygon] = []
    for o in plan.openings:
        if o.wall_id not in room.wall_ids:
            continue
        if o.kind == "window":
            continue
        wall = walls[o.wall_id]
        n = _inward_normal(layout, wall, room_key)
        p1 = wall.point_at(o.offset_mm)
        p2 = wall.point_at(o.offset_mm + o.width_mm)
        # approach rectangle (both rooms need to walk through)
        depth = 600.0
        out.append(Polygon([
            p1, p2,
            (p2[0] + n[0] * depth, p2[1] + n[1] * depth),
            (p1[0] + n[0] * depth, p1[1] + n[1] * depth),
        ]))
        if o.kind != "door" or o.swing is None:
            continue
        l, _ = layout.adjacency[o.wall_id]
        swings_here = (o.swing == "left") == (l == room_key)
        if not swings_here:
            continue
        hinge = p1 if o.hinge == "start" else p2
        other = p2 if o.hinge == "start" else p1
        r = o.width_mm
        a0 = math.atan2(other[1] - hinge[1], other[0] - hinge[0])
        a1 = math.atan2(n[1], n[0])
        # sweep from the closed-door direction to the inward normal
        da = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi
        pts = [hinge]
        for k in range(9):
            a = a0 + da * k / 8
            pts.append((hinge[0] + r * math.cos(a), hinge[1] + r * math.sin(a)))
        out.append(Polygon(pts))
    return out


class RoomSolver:
    def __init__(self, layout: LayoutResult, room_key: str,
                 rng: np.random.Generator, fid_counter: list[int]):
        self.layout = layout
        self.key = room_key
        self.rng = rng
        self.fid = fid_counter
        self.plan = layout.plan
        self.room: Room = self.plan.room_by_id()[layout.room_key_to_id[room_key]]
        self.poly = Polygon(self.room.polygon)
        walls = self.plan.wall_by_id()
        self.walls = [walls[wid] for wid in self.room.wall_ids]
        max_t = max((w.thickness_mm for w in self.walls), default=100.0)
        self.inset = max_t / 2 + WALL_GAP_MM
        # free region = room minus each wall's own half-slab (a single
        # max-thickness inset would wrongly ban fixtures on thinner walls)
        from shapely.geometry import LineString
        from shapely.ops import unary_union
        slabs = unary_union([
            LineString([w.start, w.end]).buffer(w.thickness_mm / 2 + 2,
                                                cap_style="square")
            for w in self.walls])
        self.free = self.poly.difference(slabs)
        self.obstacles = door_obstacles(layout, room_key)
        self.solids: list[Polygon] = []
        self.placed: list[Fixture] = []

    # -- helpers -------------------------------------------------------------
    def _next_id(self) -> str:
        self.fid[0] += 1
        return f"fixture_{self.fid[0]}"

    def _ok(self, fpoly: Polygon, clearance: Polygon | None) -> bool:
        if not fpoly.within(self.free.buffer(WALL_GAP_MM + 5)):
            return False
        for ob in self.obstacles:
            if fpoly.intersects(ob):
                return False
        for s in self.solids:
            if fpoly.intersects(s):
                return False
        if clearance is not None:
            for s in self.solids:
                if clearance.intersects(s):
                    return False
        return True

    def _commit(self, f: Fixture) -> Fixture:
        self.solids.append(_fixture_poly(f))
        self.placed.append(f)
        self.plan.fixtures.append(f)
        return f

    def _eligible_walls(self, pref: str, min_len: float) -> list[Wall]:
        cands = []
        for w in self.walls:
            if w.length < min_len:
                continue
            l, r = self.layout.adjacency[w.id]
            is_ext = l is None or r is None
            if pref == "exterior" and not is_ext:
                continue
            if pref == "interior" and is_ext:
                continue
            cands.append(w)
        if pref == "longest":
            cands.sort(key=lambda w: -w.length)
        else:
            self.rng.shuffle(cands)
        return cands

    def _clearance_poly(self, center, w, d, rot_deg, front, sides) -> Polygon | None:
        if front <= 0 and sides <= 0:
            return None
        cw, cd = w + 2 * sides, d + front
        a = math.radians(rot_deg)
        # clearance box extends `front` beyond the front (+y local) face
        cy_off = front / 2
        cx = center[0] - math.sin(a) * cy_off
        cy = center[1] + math.cos(a) * cy_off
        f = Fixture(id="", kind="", room_id="", rect_mm=(cx - cw / 2, cy - cd / 2, cw, cd),
                    rotation_deg=rot_deg)
        return _fixture_poly(f)

    # -- placement strategies --------------------------------------------
    def place_on_wall(self, rule: FixtureRule, w: float, d: float,
                      params: dict | None = None) -> Fixture | None:
        for wall in self._eligible_walls(rule.wall_pref, w + 300):
            n = _inward_normal(self.layout, wall, self.key)
            rot = _rot_for_inward(n)
            back_off = wall.thickness_mm / 2 + WALL_GAP_MM
            for _ in range(POSE_TRIES // 4):
                t = float(self.rng.uniform(150, wall.length - w - 150))
                mid = wall.point_at(t + w / 2)
                center = (mid[0] + n[0] * (back_off + d / 2),
                          mid[1] + n[1] * (back_off + d / 2))
                f = _quad_to_fixture(self._next_id(), rule.kind, self.room.id,
                                     center, w, d, rot, params)
                fpoly = _fixture_poly(f)
                clr = self._clearance_poly(center, w, d, rot,
                                           rule.clearance_front_mm,
                                           rule.clearance_sides_mm)
                if self._ok(fpoly, clr):
                    return self._commit(f)
        return None

    def place_in_corner(self, rule: FixtureRule, w: float, d: float) -> Fixture | None:
        coords = self.room.polygon
        order = self.rng.permutation(len(coords))
        for i in order:
            cx, cy = coords[int(i)]
            centroid = self.poly.centroid
            sx = 1 if centroid.x > cx else -1
            sy = 1 if centroid.y > cy else -1
            center = (cx + sx * (self.inset + w / 2), cy + sy * (self.inset + d / 2))
            f = _quad_to_fixture(self._next_id(), rule.kind, self.room.id,
                                 center, w, d, 0.0)
            if self._ok(_fixture_poly(f), None):
                return self._commit(f)
        return None

    def place_center(self, rule: FixtureRule, w: float, d: float,
                     params: dict | None = None) -> Fixture | None:
        minx, miny, maxx, maxy = self.free.bounds
        for _ in range(POSE_TRIES):
            cx = float(self.rng.uniform(minx, maxx))
            cy = float(self.rng.uniform(miny, maxy))
            rot = float(self.rng.choice([0.0, 90.0]))
            f = _quad_to_fixture(self._next_id(), rule.kind, self.room.id,
                                 (cx, cy), w, d, rot, params)
            fpoly = _fixture_poly(f)
            clr = self._clearance_poly((cx, cy), w, d, rot,
                                       rule.clearance_front_mm,
                                       rule.clearance_sides_mm)
            if fpoly.within(self.free) and self._ok(fpoly, clr):
                return self._commit(f)
        return None

    def place_near(self, rule: FixtureRule, w: float, d: float) -> Fixture | None:
        anchor = next((f for f in self.placed if f.kind == rule.near), None)
        if anchor is None:
            return None
        mode = rule.params.get("mode", "front")
        ax, ay, aw, ah = anchor.rect_mm
        acx, acy = ax + aw / 2, ay + ah / 2
        a = math.radians(anchor.rotation_deg)
        fx, fy = -math.sin(a), math.cos(a)  # anchor front direction
        if mode == "front":
            gap = float(self.rng.uniform(350, 550))
            dist = ah / 2 + gap + d / 2
            center = (acx + fx * dist, acy + fy * dist)
            f = _quad_to_fixture(self._next_id(), rule.kind, self.room.id,
                                 center, w, d, anchor.rotation_deg)
            if self._ok(_fixture_poly(f), None):
                return self._commit(f)
            return None
        if mode == "beside":
            sx, sy = math.cos(a), math.sin(a)  # along-wall direction
            for sign in (1, -1):
                dist = aw / 2 + 50 + w / 2
                center = (acx + sx * dist * sign, acy + sy * dist * sign)
                f = _quad_to_fixture(self._next_id(), rule.kind, self.room.id,
                                     center, w, d, anchor.rotation_deg)
                if self._ok(_fixture_poly(f), None):
                    return self._commit(f)
            return None
        if mode == "opposite":
            # wall placement on the wall facing the anchor's front
            best: tuple[float, Wall] | None = None
            for wall in self._eligible_walls("any", w + 300):
                n = _inward_normal(self.layout, wall, self.key)
                dot = n[0] * fx + n[1] * fy
                if dot < -0.7:  # wall faces back toward the anchor
                    mid = wall.point_at(wall.length / 2)
                    dd = math.dist((acx, acy), mid)
                    if best is None or dd > best[0]:
                        best = (dd, wall)
            if best is None:
                return None
            wall = best[1]
            n = _inward_normal(self.layout, wall, self.key)
            rot = _rot_for_inward(n)
            back_off = wall.thickness_mm / 2 + WALL_GAP_MM
            # center it in front of the anchor if possible
            proj = wall.length / 2
            for t in (proj - w / 2, 150.0, wall.length - w - 150.0):
                if t < 150 or t + w > wall.length - 150:
                    continue
                mid = wall.point_at(t + w / 2)
                center = (mid[0] + n[0] * (back_off + d / 2),
                          mid[1] + n[1] * (back_off + d / 2))
                f = _quad_to_fixture(self._next_id(), rule.kind, self.room.id,
                                     center, w, d, rot)
                if self._ok(_fixture_poly(f), None):
                    return self._commit(f)
            return None
        return None

    def place_counter_run(self, rule: FixtureRule) -> list[Fixture]:
        """Kitchen counter: strip(s) of `depth` along 1-2 door-free walls."""
        depth = float(self.rng.uniform(*rule.size_mm[1]))
        openings_by_wall: dict[str, list[Opening]] = {}
        for o in self.plan.openings:
            openings_by_wall.setdefault(o.wall_id, []).append(o)
        free_walls = [w for w in self.walls
                      if not any(o.kind != "window"
                                 for o in openings_by_wall.get(w.id, []))
                      and w.length >= 1200]
        free_walls.sort(key=lambda w: -w.length)
        out: list[Fixture] = []
        for wall in free_walls[: (2 if self.rng.random() < 0.5 else 1)]:
            n = _inward_normal(self.layout, wall, self.key)
            rot = _rot_for_inward(n)
            back_off = wall.thickness_mm / 2 + WALL_GAP_MM
            margin = 150.0
            length = wall.length - 2 * margin
            mid = wall.point_at(margin + length / 2)
            center = (mid[0] + n[0] * (back_off + depth / 2),
                      mid[1] + n[1] * (back_off + depth / 2))
            f = _quad_to_fixture(self._next_id(), "counter", self.room.id,
                                 center, length, depth, rot)
            fpoly = _fixture_poly(f)
            ok = fpoly.within(self.free.buffer(WALL_GAP_MM + 5)) and not any(
                fpoly.intersects(ob) for ob in self.obstacles)
            if ok and not any(fpoly.intersects(s) for s in self.solids):
                out.append(self._commit(f))
        return out

    def place_in_counter(self, rule: FixtureRule, w: float, d: float) -> Fixture | None:
        counters = [f for f in self.placed if f.kind == "counter"]
        self.rng.shuffle(counters)
        min_from = rule.params.get("min_from", {})
        for counter in counters:
            cx, cy, cw, ch = counter.rect_mm
            ccx, ccy = cx + cw / 2, cy + ch / 2
            a = math.radians(counter.rotation_deg)
            ux, uy = math.cos(a), math.sin(a)  # along-counter direction
            for _ in range(POSE_TRIES // 4):
                t = float(self.rng.uniform(-(cw / 2 - w / 2 - 50), cw / 2 - w / 2 - 50))
                center = (ccx + ux * t, ccy + uy * t)
                ok = True
                for other_kind, dmin in min_from.items():
                    for f in self.placed:
                        if f.kind == other_kind:
                            ox, oy, ow, oh = f.rect_mm
                            if math.dist(center, (ox + ow / 2, oy + oh / 2)) < dmin:
                                ok = False
                if not ok:
                    continue
                # in-counter fixtures may overlap their counter, nothing else
                f = _quad_to_fixture(self._next_id(), rule.kind, self.room.id,
                                     center, w, d, counter.rotation_deg,
                                     {**rule.params, "in_counter": True})
                fpoly = _fixture_poly(f)
                clash = any(
                    fpoly.intersects(s) for s, pf in zip(self.solids, self.placed)
                    if pf.kind != "counter")
                if not clash:
                    return self._commit(f)
        return None


def furnish(layout: LayoutResult, rng: np.random.Generator) -> FurnishStats:
    """Furnish every room per RULES. Mutates layout.plan.fixtures."""
    stats = FurnishStats()
    fid = [0]
    n_bed = layout.program.meta.n_bed if layout.program.meta else 2

    for key, spec in layout.specs.items():
        room_rules = RULES.get(spec.role)
        if room_rules is None or not room_rules.rules:
            continue
        solver = RoomSolver(layout, key, rng, fid)
        if solver.free.is_empty:
            stats.unfurnished_rooms.append(key)
            continue
        room_placed = 0
        required_dropped = 0
        for rule in room_rules.rules:
            if rule.prob < 1.0 and rng.random() > rule.prob:
                continue
            count = rule.count
            if spec.role == "aircon_ledge" and rule.kind == "ac_unit":
                count = ac_unit_count(n_bed)
            n = int(rng.integers(count[0], count[1] + 1))
            for _ in range(n):
                w = float(rng.uniform(*rule.size_mm[0]))
                d = float(rng.uniform(*rule.size_mm[1]))
                params = dict(rule.params)
                if rule.kind == "bed":
                    wr = (rule.params["master_width"]
                          if spec.label_role == "bedroom_master"
                          else rule.params["common_width"])
                    w = float(rng.uniform(*wr))
                    params = {}
                placed: Fixture | list[Fixture] | None
                if rule.placement == "wall":
                    placed = solver.place_on_wall(rule, w, d, params)
                elif rule.placement == "corner":
                    placed = solver.place_in_corner(rule, w, d)
                elif rule.placement == "center":
                    placed = solver.place_center(rule, w, d, params)
                elif rule.placement == "near":
                    placed = solver.place_near(rule, w, d)
                elif rule.placement == "counter_run":
                    placed = solver.place_counter_run(rule) or None
                elif rule.placement == "in_counter":
                    placed = solver.place_in_counter(rule, w, d)
                else:
                    placed = None
                if placed is None:
                    if rule.hard:
                        raise SemanticsReject(f"hard_rule:{rule.kind}@{key}")
                    stats.dropped.append(f"{rule.kind}@{key}")
                    if rule.prob >= 1.0:
                        required_dropped += 1
                else:
                    room_placed += len(placed) if isinstance(placed, list) else 1
        # a room is "unfurnished" only if required rules failed AND nothing
        # landed — optional rules whose dice didn't fire are not a failure
        if room_placed == 0 and required_dropped > 0:
            stats.unfurnished_rooms.append(key)
        stats.placed += room_placed

    if len(stats.unfurnished_rooms) > max(1, len(layout.specs) // 8):
        raise SemanticsReject(
            f"too_many_unfurnished:{','.join(stats.unfurnished_rooms)}")
    return stats
