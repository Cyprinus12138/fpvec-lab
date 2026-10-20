"""Canonical ordering + augmentation transform, operating on fpeval FloorPlan.

Canonical form (recomputed *after* any augmentation, so ordering is a pure
function of geometry):

* each wall's endpoints are oriented **upper-left first** — the endpoint with the
  smaller ``(y, x)`` is ``start`` (curvature sign negates on a swap, since the
  sagitta is signed relative to direction);
* walls are sorted by ``(start_y, start_x, end_y, end_x)``; a wall's canonical
  index is its position in that order (this is the WREF index);
* a room cycle is rotated to begin at its minimum member wall-index.

`transform` applies one of the 8 flip×rot90 symmetries about the frame centre —
an isometry that keeps coordinates in-frame — to a whole FloorPlan, for the
aug-equivariance path.
"""

from __future__ import annotations

from fpeval.model import FloorPlan, Opening, Room, Wall

Point = tuple[float, float]


# ------------------------------------------------------------- wall ordering

def _oriented(w: Wall) -> Wall:
    """Return a copy of `w` with endpoints upper-left first."""
    if (w.start[1], w.start[0]) <= (w.end[1], w.end[0]):
        return w
    return Wall(id=w.id, start=w.end, end=w.start,
                thickness=w.thickness, curvature=-w.curvature,
                type=w.type,
                openings=[Opening(kind=o.kind, center=o.center,
                                  width=o.width, wall_id=o.wall_id)
                          for o in w.openings])


def canonical_walls(plan: FloorPlan) -> list[Wall]:
    """Walls oriented upper-left-first and sorted canonically."""
    walls = [_oriented(w) for w in plan.walls]
    walls.sort(key=lambda w: (w.start[1], w.start[0], w.end[1], w.end[0]))
    return walls


def canonical_index(walls: list[Wall]) -> dict[str, int]:
    """Map original wall id -> canonical index."""
    return {w.id: i for i, w in enumerate(walls)}


def rotate_cycle_to_min(members: list[tuple]) -> list[tuple]:
    """Rotate a room member cycle to begin at its smallest wall index. OPEN
    markers ride along; rotation starts at the lowest ``("W", idx)`` member."""
    wref_positions = [(m[1], i) for i, m in enumerate(members) if m[0] == "W"]
    if not wref_positions:
        return members
    _, start = min(wref_positions)
    return members[start:] + members[:start]


# ------------------------------------------------------------- augmentation

def transform(plan: FloorPlan, flip: bool, k: int) -> FloorPlan:
    """Apply flip (reflect x about centre) then `k` quarter-turns about the
    frame centre. Returns a new FloorPlan; ids/labels/types preserved."""
    c = plan.frame / 2.0
    det = -1.0 if flip else 1.0  # curvature sign follows orientation-reversal

    def pt(p: Point) -> Point:
        x, y = p
        if flip:
            x = 2 * c - x
        for _ in range(k % 4):
            x, y = 2 * c - y, x   # rot90: (x,y) -> (2c - y, x), det = +1
        return (x, y)

    def xform_wall(w: Wall) -> Wall:
        return Wall(
            id=w.id, start=pt(w.start), end=pt(w.end),
            thickness=w.thickness, curvature=w.curvature * det, type=w.type,
            openings=[Opening(kind=o.kind, center=pt(o.center),
                              width=o.width, wall_id=o.wall_id)
                      for o in w.openings])

    def xform_room(r: Room) -> Room:
        return Room(label=r.label, wall_ids=list(r.wall_ids),
                    polygon=[pt(p) for p in r.polygon] if r.polygon else None)

    return FloorPlan(walls=[xform_wall(w) for w in plan.walls],
                     rooms=[xform_room(r) for r in plan.rooms],
                     frame=plan.frame)
