"""Wall-cycle geometry: ordering walls around a room ring, locating open
(non-wall) edges, and reconstructing the room polygon.

Two directions, kept consistent so the codec round-trips:

* **encode** (`ring_order_from_polygon`): given the GT room polygon plus the
  room's wall segments, order the walls by their projection along the polygon
  exterior and mark where consecutive walls leave a gap → an open edge. The
  polygon is authoritative here (fpeval GT carries it, E2f).
* **decode** (`polygon_from_cycle`): given walls already in canonical ring order
  plus the positions of OPEN markers (no coordinates), chain wall endpoints —
  orienting each wall by shared-endpoint coincidence — and bridge each open span
  with a straight edge between the two loose endpoints. Reproduces the ring
  (with possibly extra collinear vertices — geometrically identical).

All coordinates are FRAME units. Chaining tolerance scales with the frame.
"""

from __future__ import annotations

import math

Point = tuple[float, float]
Seg = tuple[Point, Point]


def _dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


# ------------------------------------------------------------------ encode

def ring_order_from_polygon(polygon: list[Point], segs: list[Seg],
                            tol: float) -> tuple[list[int], list[bool]]:
    """Order `segs` (a room's wall chords) around `polygon` and flag open gaps.

    Returns ``(order, open_before)`` where ``order`` indexes into `segs` in
    ring-traversal order and ``open_before[i]`` is True iff an open (non-wall)
    edge precedes ``segs[order[i]]`` in the cycle. A segment lies on the ring;
    its along-ring position is the projection of its midpoint. A gap between the
    exit vertex of one wall and the entry vertex of the next (> tol) is an open
    edge.
    """
    from shapely.geometry import LineString, Point as ShPoint

    ring = LineString(list(polygon) + [polygon[0]])
    order = sorted(range(len(segs)),
                   key=lambda k: ring.project(ShPoint(
                       ((segs[k][0][0] + segs[k][1][0]) / 2,
                        (segs[k][0][1] + segs[k][1][1]) / 2))))

    # An open (non-wall) edge sits between two walls that are consecutive along
    # the ring but do NOT touch — i.e. share no endpoint. This is orientation-
    # free (no fragile entry/exit projection, which mis-fires at the ring seam):
    # collinear-split walls and corner-adjacent walls always share a vertex, so
    # only genuine gaps flag as open.
    n = len(order)
    open_before = [False] * n
    for i in range(n):
        prev = segs[order[(i - 1) % n]]
        cur = segs[order[i]]
        touches = min(_dist(prev[0], cur[0]), _dist(prev[0], cur[1]),
                      _dist(prev[1], cur[0]), _dist(prev[1], cur[1]))
        if touches > tol:
            open_before[i] = True
    return order, open_before


# ------------------------------------------------------------------ decode

def polygon_from_cycle(segs: list[Seg], open_before: list[bool],
                       tol: float) -> list[Point]:
    """Reconstruct a room ring from walls in canonical order + OPEN markers.

    `segs[i]` are wall chords (endpoints in arbitrary order); `open_before[i]`
    marks an open edge preceding wall i (cyclically). Walls that are NOT
    open-separated from their neighbour share an endpoint; the grammar forbids
    two consecutive OPEN, so every wall has at least one direct neighbour and
    each wall's orientation is fixed locally by coincidence with that neighbour.
    Returns the ring vertices (first not repeated).
    """
    n = len(segs)
    if n == 0:
        return []
    if n == 1:
        # single wall + one open edge closing back to it
        return [segs[0][0], segs[0][1]]

    def shares(p: Point, seg: Seg) -> bool:
        return _dist(p, seg[0]) <= tol or _dist(p, seg[1]) <= tol

    # Orient each wall as (entry, exit). For a wall directly connected to its
    # NEXT neighbour, exit is the endpoint shared with that neighbour. For a wall
    # whose next edge is OPEN, exit is the endpoint shared with the PREVIOUS
    # neighbour's chain instead — resolved by falling through to the direct link
    # on whichever side is not open.
    exit_pt: list[Point] = [None] * n  # type: ignore[list-item]
    entry_pt: list[Point] = [None] * n  # type: ignore[list-item]
    for i in range(n):
        a, b = segs[i]
        nxt = segs[(i + 1) % n]
        open_next = open_before[(i + 1) % n]
        if not open_next and shares(b, nxt) and not shares(a, nxt):
            entry_pt[i], exit_pt[i] = a, b
        elif not open_next and shares(a, nxt) and not shares(b, nxt):
            entry_pt[i], exit_pt[i] = b, a
        else:
            # next edge is open (or ambiguous): orient by the PREVIOUS neighbour
            prv = segs[(i - 1) % n]
            open_cur = open_before[i]
            if not open_cur and shares(a, prv) and not shares(b, prv):
                entry_pt[i], exit_pt[i] = a, b
            elif not open_cur and shares(b, prv) and not shares(a, prv):
                entry_pt[i], exit_pt[i] = b, a
            else:
                # isolated (both sides open, or a junction touched on both ends):
                # fall back to ring continuity — pick the orientation whose entry
                # is closest to the previous wall's exit if known, else identity.
                entry_pt[i], exit_pt[i] = a, b

    # Assemble the ring: entry, exit of each wall in order; an open edge inserts
    # only the straight bridge (prev exit -> this entry), which appears naturally
    # because we always append entry then exit.
    verts: list[Point] = []
    for i in range(n):
        e, x = entry_pt[i], exit_pt[i]
        if not verts or _dist(verts[-1], e) > tol:
            verts.append(e)
        verts.append(x)
    # drop a duplicated closing vertex
    if len(verts) >= 2 and _dist(verts[0], verts[-1]) <= tol:
        verts.pop()
    return verts


def chain_closed(segs: list[Seg], tol: float) -> list[Point] | None:
    """Chain a fully-closed set of wall chords into a ring by pure endpoint
    adjacency (no ordering supplied). Returns None if it does not close.
    Used as a decode fallback for rooms with no open edges."""
    remaining = list(segs)
    chain = [remaining.pop(0)]
    while remaining:
        tail = chain[-1][1]
        best, best_d, flip = None, None, False
        for k, (a, b) in enumerate(remaining):
            for f, p in ((False, a), (True, b)):
                d = _dist(tail, p)
                if best_d is None or d < best_d:
                    best, best_d, flip = k, d, f
        if best is None or best_d > tol:
            return None
        a, b = remaining.pop(best)
        chain.append((b, a) if flip else (a, b))
    if _dist(chain[-1][1], chain[0][0]) > tol:
        return None
    return [c[0] for c in chain]
