"""Canonical in-memory floorplan model + loader.

Source format is the wall-first plan JSON used throughout this repository:
walls as ``{id, start:[x,y], end:[x,y], thickness, curvature,
openings:[{type,center,width}]}``, rooms as ``{label, walls:[wallId]}`` with an
optional explicit ``polygon``. Coordinates are in an image-derived frame
(longer edge 1024 by convention).

Everything loads into ``FloorPlan``; all metrics operate on that. Curved walls
are approximated by sampled polylines (sagitta interpretation of ``curvature``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

Point = tuple[float, float]

DOOR_KINDS = {"door", "sliding_door", "opening"}
WINDOW_KINDS = {"window"}


@dataclass
class Opening:
    kind: str                 # door | sliding_door | opening | window
    center: Point
    width: float
    wall_id: str | None = None

    @property
    def family(self) -> str:
        return "door" if self.kind in DOOR_KINDS else "window"


@dataclass
class Wall:
    id: str
    start: Point
    end: Point
    thickness: float = 10.0
    curvature: float = 0.0    # sagitta: max offset of arc midpoint from chord (signed)
    type: str = "interior"    # exterior | interior | partition | shelter
    openings: list[Opening] = field(default_factory=list)

    @property
    def length(self) -> float:
        return math.dist(self.start, self.end)

    def polyline(self, n: int = 8) -> list[Point]:
        """Sampled polyline; straight walls return [start, end]."""
        if abs(self.curvature) < 1e-9 or self.length < 1e-9:
            return [self.start, self.end]
        (x0, y0), (x1, y1) = self.start, self.end
        dx, dy = x1 - x0, y1 - y0
        L = self.length
        # unit normal (left of direction); sagitta bulge follows curvature sign
        nx, ny = -dy / L, dx / L
        pts: list[Point] = []
        for i in range(n + 1):
            t = i / n
            # circular-ish bulge: sagitta * sin-profile is a good chord approximation
            off = self.curvature * math.sin(math.pi * t)
            pts.append((x0 + dx * t + nx * off, y0 + dy * t + ny * off))
        return pts


@dataclass
class Room:
    label: str = ""
    wall_ids: list[str] = field(default_factory=list)
    polygon: list[Point] | None = None   # resolved boundary, if the source had one


@dataclass
class FloorPlan:
    walls: list[Wall]
    rooms: list[Room]
    frame: float = 1024.0   # nominal coordinate frame (longer-edge grid size)

    @property
    def openings(self) -> list[Opening]:
        return [o for w in self.walls for o in w.openings]

    def wall_by_id(self) -> dict[str, Wall]:
        return {w.id: w for w in self.walls}

    def bbox(self) -> tuple[float, float, float, float]:
        xs = [c for w in self.walls for c in (w.start[0], w.end[0])]
        ys = [c for w in self.walls for c in (w.start[1], w.end[1])]
        if not xs:
            return (0.0, 0.0, 0.0, 0.0)
        return (min(xs), min(ys), max(xs), max(ys))


# ---------------------------------------------------------------- loaders

def from_vlm_json(doc: dict, frame: float = 1024.0) -> FloorPlan:
    """Load the Gemini/fixture format: {walls:[...], rooms:[...]}."""
    walls: list[Wall] = []
    for i, w in enumerate(doc.get("walls", [])):
        wid = str(w.get("id", f"wall_{i + 1}"))
        openings = [
            Opening(
                kind=str(o.get("type", o.get("kind", "opening"))),
                center=(float(o["center"][0]), float(o["center"][1])),
                width=float(o.get("width", 0.0)),
                wall_id=wid,
            )
            for o in w.get("openings", [])
            if o.get("center") is not None
        ]
        walls.append(Wall(
            id=wid,
            start=(float(w["start"][0]), float(w["start"][1])),
            end=(float(w["end"][0]), float(w["end"][1])),
            thickness=float(w.get("thickness", 10.0)),
            curvature=float(w.get("curvature", 0.0)),
            type=str(w.get("type", "interior")),
            openings=openings,
        ))
    rooms = []
    for r in doc.get("rooms", []):
        poly = r.get("polygon")
        rooms.append(Room(
            label=str(r.get("label", r.get("name", ""))),
            wall_ids=[str(x) for x in r.get("walls", [])],
            polygon=([(float(p[0]), float(p[1])) for p in poly]
                     if poly and len(poly) >= 3 else None),
        ))
    return FloorPlan(walls=walls, rooms=rooms, frame=frame)


def load_any(doc: dict) -> FloorPlan:
    """Load a plan document (VLM JSON is the only supported format)."""
    return from_vlm_json(doc)
