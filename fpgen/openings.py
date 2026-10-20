"""Opening placement: doors along the program's access tree, windows on
exterior walls, then a BFS reachability check over the resulting room graph.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from .layout import LayoutResult
from .scene import Opening

END_MARGIN_MM = 150.0
GAP_MM = 100.0

DOOR_WIDTH = {"entrance": (900, 1100), "door": (750, 950), "shelter": (700, 800)}
SLIDING_WIDTH = (1400, 2600)
CASED_OPENING_WIDTH = (1000, 2400)
WINDOW_WIDTH = (900, 2400)

# fallback door-routing when the declared access_parent shares no usable wall
ALT_PARENTS = {
    "bedroom": ["corridor", "living", "foyer"],
    "study": ["corridor", "living", "foyer"],
    "household_shelter": ["kitchen", "yard", "corridor", "foyer", "living"],
    "wc": ["yard", "kitchen", "corridor", "living"],
    "store": ["kitchen", "corridor", "foyer", "living", "yard"],
    "corridor": ["living", "foyer", "kitchen"],
    "bath": ["corridor", "living", "foyer", "kitchen"],
    "yard": ["kitchen", "corridor"],
    "utility": ["kitchen", "corridor", "yard"],
}
# rooms that are legitimately unreachable by door (accessed for maintenance)
NO_ACCESS_ROLES = {"aircon_ledge", "planter", "void", "db"}


class OpeningsReject(Exception):
    def __init__(self, cause: str):
        super().__init__(cause)
        self.cause = cause


def _walls_between(layout: LayoutResult, a: str, b: str | None) -> list[str]:
    """Wall ids whose two sides are rooms a and b (b=None -> exterior)."""
    out = []
    for wid, (l, r) in layout.adjacency.items():
        if {l, r} == {a, b}:
            out.append(wid)
    return out


def _place_on_wall(layout: LayoutResult, wall_id: str, width: float,
                   taken: dict[str, list[tuple[float, float]]],
                   rng: np.random.Generator) -> float | None:
    """Find a free offset for an opening of `width` on the wall, else None."""
    wall = layout.plan.wall_by_id()[wall_id]
    usable = wall.length - 2 * END_MARGIN_MM - width
    if usable < 0:
        return None
    for _ in range(12):
        off = END_MARGIN_MM + float(rng.uniform(0, usable))
        span = (off - GAP_MM, off + width + GAP_MM)
        if all(span[1] < lo or span[0] > hi for lo, hi in taken.get(wall_id, [])):
            taken.setdefault(wall_id, []).append((off, off + width))
            return off
    return None


def _add_door(layout: LayoutResult, room_key: str, parent: str | None,
              kind: str, width_range: tuple[float, float],
              taken: dict, rng: np.random.Generator,
              params: dict | None = None) -> Opening | None:
    walls = _walls_between(layout, room_key, parent)
    rng.shuffle(walls)
    walls.sort(key=lambda w: -layout.plan.wall_by_id()[w].length)
    for wid in walls:
        wall_len = layout.plan.wall_by_id()[wid].length
        wmax = min(width_range[1], wall_len - 2 * END_MARGIN_MM)
        if wmax < width_range[0]:
            continue
        width = float(rng.uniform(width_range[0], wmax))
        off = _place_on_wall(layout, wid, width, taken, rng)
        if off is not None:
            # doors swing into the room being entered (the child room).
            # Convention shared with semantics/render: "left" = the side room
            # returned first by layout adjacency (sampled at mid - normal*eps).
            swing = None
            if kind == "door":
                swing = "left" if layout.adjacency[wid][0] == room_key else "right"
            return Opening(
                id="", wall_id=wid, offset_mm=off, width_mm=width, kind=kind,  # type: ignore[arg-type]
                hinge=("start" if rng.random() < 0.5 else "end") if kind == "door" else None,
                swing=swing,
                params=params or {},
            )
    return None


def place_openings(layout: LayoutResult, rng: np.random.Generator) -> None:
    """Populate layout.plan.openings in place. Raises OpeningsReject."""
    plan = layout.plan
    taken: dict[str, list[tuple[float, float]]] = {}
    openings: list[Opening] = []

    # --- entrance ----------------------------------------------------------
    entrance = None
    entrance_rooms = [layout.program.entrance_key] + [
        k for k in ("foyer", "living", "corridor", "kitchen")
        if k in layout.specs and k != layout.program.entrance_key]
    for room_key in entrance_rooms:
        entrance = _add_door(layout, room_key, None, "door",
                             DOOR_WIDTH["entrance"], taken, rng,
                             params={"entrance": True})
        if entrance is not None:
            layout.program.entrance_key = room_key
            break
    if entrance is None:
        raise OpeningsReject("entrance")
    openings.append(entrance)

    # --- interior doors along the access tree ------------------------------
    for key, spec in layout.specs.items():
        if spec.access_parent in (None, "ENTRANCE"):
            continue
        role = spec.role
        if spec.door_kind == "sliding_door":
            width_range, kind, params = SLIDING_WIDTH, "sliding_door", {}
        elif spec.door_kind == "opening":
            width_range, kind, params = CASED_OPENING_WIDTH, "opening", {}
        elif role == "household_shelter":
            width_range, kind, params = DOOR_WIDTH["shelter"], "door", {"heavy": True}
        else:
            width_range, kind, params = DOOR_WIDTH["door"], "door", {}

        parents = [spec.access_parent] + [
            p for p in ALT_PARENTS.get(role, []) if p != spec.access_parent]
        # already open-connected to a parent (open-plan LDK) — no door needed
        if any(frozenset((key, p)) in layout.open_adjacency
               for p in parents if p in layout.specs):
            continue
        door = None
        for parent in parents:
            if parent not in layout.specs:
                continue
            door = _add_door(layout, key, parent, kind, width_range, taken, rng,
                             params=params)
            if door is not None:
                break
        if door is None:
            raise OpeningsReject(f"door:{key}")
        openings.append(door)

    # --- windows ------------------------------------------------------------
    for key, spec in layout.specs.items():
        if not spec.window or spec.role == "household_shelter":
            continue
        ext_walls = _walls_between(layout, key, None)
        ext_walls.sort(key=lambda w: -plan.wall_by_id()[w].length)
        placed = 0
        want = 2 if (ext_walls and plan.wall_by_id()[ext_walls[0]].length > 5000
                     and rng.random() < 0.5) else 1
        for wid in ext_walls:
            if placed >= want:
                break
            wall_len = plan.wall_by_id()[wid].length
            wmax = min(WINDOW_WIDTH[1], wall_len - 2 * END_MARGIN_MM)
            if wmax < WINDOW_WIDTH[0]:
                continue
            width = float(rng.uniform(WINDOW_WIDTH[0], wmax))
            off = _place_on_wall(layout, wid, width, taken, rng)
            if off is not None:
                openings.append(Opening(id="", wall_id=wid, offset_mm=off,
                                        width_mm=width, kind="window"))
                placed += 1
        if placed == 0 and spec.role == "bedroom":
            raise OpeningsReject(f"window:{key}")

    for i, o in enumerate(openings, start=1):
        o.id = f"opening_{i}"
    plan.openings = openings

    _check_reachability(layout)


def _check_reachability(layout: LayoutResult) -> None:
    plan = layout.plan
    passable = {"door", "sliding_door", "opening"}
    graph: dict[str | None, set[str | None]] = {}
    # open-plan boundaries are passable by construction (no wall to hold a door)
    for pair in layout.open_adjacency:
        a, b = tuple(pair)
        graph.setdefault(a, set()).add(b)
        graph.setdefault(b, set()).add(a)
    for o in plan.openings:
        if o.kind not in passable:
            continue
        l, r = layout.adjacency[o.wall_id]
        graph.setdefault(l, set()).add(r)
        graph.setdefault(r, set()).add(l)

    start = layout.program.entrance_key
    seen: set[str | None] = {start}
    queue = deque([start])
    while queue:
        cur = queue.popleft()
        for nxt in graph.get(cur, ()):
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)

    for key, spec in layout.specs.items():
        if spec.role in NO_ACCESS_ROLES:
            continue
        if key not in seen:
            raise OpeningsReject(f"unreachable:{key}")
