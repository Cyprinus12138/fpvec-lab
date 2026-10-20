"""Canonical in-memory floorplan model + loaders.

Two source formats exist in this repo:

* **VLM JSON** (Gemini / fixtures / FloorplanVLM-style): walls as
  ``{id, start:[x,y], end:[x,y], thickness, curvature, openings:[{type,center,width}]}``,
  rooms as ``{label, walls:[wallId]}``. Coordinates are in an image-derived grid
  (typically 0..1024 per axis).
* **PropWise DSL** (mm, walls as polyline/arc segment chains, openings by
  offset along the wall path, rooms as boundary polygons).

Both load into ``FloorPlan``; all metrics operate on that. Curved walls are
approximated by sampled polylines (sagitta interpretation of ``curvature``).
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
    type: str = "interior"    # exterior | interior | partition | shelter (§1 tuple)
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


def _dsl_path_points(start: Point, segments: list[dict], arc_n: int = 8) -> list[Point]:
    pts = [start]
    for seg in segments:
        to = (float(seg["to"][0]), float(seg["to"][1]))
        if seg.get("type") == "arc":
            # approximate the arc with a sagitta bulge over the chord
            a = pts[-1]
            chord = math.dist(a, to)
            r = float(seg.get("radius_mm", chord))
            if chord > 1e-9 and r > chord / 2:
                sag = r - math.sqrt(max(r * r - (chord / 2) ** 2, 0.0))
                if seg.get("large_arc"):
                    sag = 2 * r - sag
                sign = 1.0 if seg.get("sweep") else -1.0
                dx, dy = to[0] - a[0], to[1] - a[1]
                nx, ny = -dy / chord, dx / chord
                for i in range(1, arc_n):
                    t = i / arc_n
                    off = sign * sag * math.sin(math.pi * t)
                    pts.append((a[0] + dx * t + nx * off, a[1] + dy * t + ny * off))
        pts.append(to)
    return pts


def from_propwise_dsl(doc: dict, frame: float | None = None) -> FloorPlan:
    """Load the PropWise mm DSL. Polyline walls are split into straight Walls
    (ids get a ``#k`` suffix); openings are resolved from path offsets to
    absolute centers; rooms keep their resolved boundary polygons."""
    walls: list[Wall] = []
    # wall id -> list of (cum_len_start, seg_start, seg_end, Wall)
    paths: dict[str, list[tuple[float, Point, Point, Wall]]] = {}
    for w in doc.get("walls", []):
        pts = _dsl_path_points(
            (float(w["start"][0]), float(w["start"][1])), w.get("segments", []))
        th = float(w.get("thickness_mm", 150))
        cum = 0.0
        segs: list[tuple[float, Point, Point, Wall]] = []
        for k in range(len(pts) - 1):
            a, b = pts[k], pts[k + 1]
            wid = w["id"] if len(pts) == 2 else f"{w['id']}#{k}"
            wall = Wall(id=wid, start=a, end=b, thickness=th)
            walls.append(wall)
            segs.append((cum, a, b, wall))
            cum += math.dist(a, b)
        paths[w["id"]] = segs

    for o in doc.get("openings", []):
        segs = paths.get(o.get("wall", ""))
        if not segs:
            continue
        mid = float(o["offset_mm"]) + float(o["length_mm"]) / 2.0
        # find the sub-segment containing the opening midpoint
        host = segs[-1]
        for s in segs:
            cum, a, b, wall = s
            if cum <= mid <= cum + math.dist(a, b) + 1e-6:
                host = s
                break
        cum, a, b, wall = host
        seg_len = math.dist(a, b)
        t = 0.0 if seg_len < 1e-9 else min(max((mid - cum) / seg_len, 0.0), 1.0)
        center = (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
        wall.openings.append(Opening(
            kind=str(o.get("kind", "opening")), center=center,
            width=float(o["length_mm"]), wall_id=wall.id))

    rooms = []
    for r in doc.get("rooms", []):
        poly = None
        if r.get("boundary_start") is not None and r.get("boundary"):
            poly = _dsl_path_points(
                (float(r["boundary_start"][0]), float(r["boundary_start"][1])),
                r["boundary"])
        rooms.append(Room(label=str(r.get("label", r.get("role", ""))), polygon=poly))

    if frame is None:
        bbox = doc.get("bbox_mm")
        frame = float(max(bbox[2] - bbox[0], bbox[3] - bbox[1])) if bbox else 1024.0
    return FloorPlan(walls=walls, rooms=rooms, frame=frame)


def load_any(doc: dict) -> FloorPlan:
    """Dispatch on document shape."""
    if "bbox_mm" in doc or (doc.get("walls") and "segments" in doc["walls"][0]):
        return from_propwise_dsl(doc)
    return from_vlm_json(doc)
