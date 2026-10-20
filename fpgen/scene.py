"""Canonical mm-space scene representation for fpgen.

This is fpgen's own schema (deliberately NOT the PropWise DSL): built to export
the dependency-ordered VLM JSON that fpeval scores and the Pascal adapter
consumes. Conventions: origin top-left, +x right, +y down, integer-ish mm,
coordinates snapped to a 10 mm grid at layout time.

The only two serialized formats are:
  - ScenePlan JSON (`scene.json`, via to_json/from_json) — the mm canonical form,
    kept so any sample can be re-rendered under a new style without re-running
    layout;
  - VLM JSON GT (via to_vlm_json(H)) — exactly the contract of
    fpeval.model.from_vlm_json, plus auxiliary keys it ignores (fixtures, texts,
    meta).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Literal

import numpy as np

SCHEMA_VERSION = 1
GRID_MM = 10

Point = tuple[float, float]

WallRole = Literal["exterior", "interior", "partition", "shelter"]
OpeningKind = Literal["door", "sliding_door", "opening", "window"]


def snap(v: float, grid: float = GRID_MM) -> float:
    return round(v / grid) * grid


def apply_h(h: np.ndarray, pt: Point) -> Point:
    x, y, w = h @ (pt[0], pt[1], 1.0)
    return (x / w, y / w)


def h_scale(h: np.ndarray) -> float:
    """Isotropic scale factor of a similarity transform (used for lengths)."""
    return float(math.sqrt(abs(np.linalg.det(h[:2, :2]))))


@dataclass
class Wall:
    id: str
    start: Point
    end: Point
    thickness_mm: float
    role: WallRole = "interior"
    curvature_mm: float = 0.0  # sagitta, same sign convention as fpeval

    @property
    def length(self) -> float:
        return math.dist(self.start, self.end)

    def point_at(self, offset_mm: float) -> Point:
        """Point along the wall chord at distance offset_mm from start."""
        t = offset_mm / self.length if self.length else 0.0
        return (
            self.start[0] + (self.end[0] - self.start[0]) * t,
            self.start[1] + (self.end[1] - self.start[1]) * t,
        )

    def direction(self) -> Point:
        l = self.length or 1.0
        return ((self.end[0] - self.start[0]) / l, (self.end[1] - self.start[1]) / l)

    def normal(self) -> Point:
        dx, dy = self.direction()
        return (-dy, dx)


@dataclass
class Opening:
    id: str
    wall_id: str
    offset_mm: float  # distance from wall.start to the opening's leading edge
    width_mm: float
    kind: OpeningKind
    # render-only door parameters (None for windows / plain openings)
    hinge: Literal["start", "end"] | None = None
    swing: Literal["left", "right"] | None = None
    params: dict = field(default_factory=dict)  # e.g. {"heavy": True} for shelter doors

    @property
    def center_offset_mm(self) -> float:
        return self.offset_mm + self.width_mm / 2.0


@dataclass
class Room:
    id: str
    role: str  # SG role vocabulary (living, bedroom, bath, household_shelter, ...)
    label: str  # printed label, e.g. "MASTER BEDROOM"
    polygon: list[Point]  # closed ring (first point NOT repeated), CCW in y-down
    wall_ids: list[str]  # ordered centerline cycle around the polygon
    area_sqm: float = 0.0


@dataclass
class Fixture:
    id: str
    kind: str
    room_id: str
    rect_mm: tuple[float, float, float, float]  # x, y, w, h (pre-rotation AABB)
    rotation_deg: float = 0.0  # about rect center
    params: dict = field(default_factory=dict)

    def corners(self) -> list[Point]:
        x, y, w, h = self.rect_mm
        cx, cy = x + w / 2, y + h / 2
        a = math.radians(self.rotation_deg)
        ca, sa = math.cos(a), math.sin(a)
        out = []
        for px, py in ((x, y), (x + w, y), (x + w, y + h), (x, y + h)):
            dx, dy = px - cx, py - cy
            out.append((cx + dx * ca - dy * sa, cy + dx * sa + dy * ca))
        return out


@dataclass
class TextItem:
    id: str
    kind: Literal["room_label", "area", "dimension", "logo", "legend", "scalebar_text"]
    text: str
    anchor_mm: Point  # text center anchor
    angle_deg: float = 0.0
    font_size_mm: float = 250.0
    room_id: str | None = None
    bbox_mm: tuple[float, float, float, float] | None = None  # backfilled by render


@dataclass
class ScenePlan:
    bbox_mm: tuple[float, float]  # (width, height); plan occupies [0,w]x[0,h]
    walls: list[Wall] = field(default_factory=list)
    openings: list[Opening] = field(default_factory=list)
    rooms: list[Room] = field(default_factory=list)
    fixtures: list[Fixture] = field(default_factory=list)
    texts: list[TextItem] = field(default_factory=list)
    meta: dict = field(default_factory=dict)
    seed_path: tuple[int, ...] = ()
    schema_version: int = SCHEMA_VERSION

    # -- lookups ------------------------------------------------------------
    def wall_by_id(self) -> dict[str, Wall]:
        return {w.id: w for w in self.walls}

    def openings_on(self, wall_id: str) -> list[Opening]:
        return [o for o in self.openings if o.wall_id == wall_id]

    def room_by_id(self) -> dict[str, Room]:
        return {r.id: r for r in self.rooms}

    # -- serialization ------------------------------------------------------
    def to_json(self) -> str:
        doc = asdict(self)
        doc["seed_path"] = list(self.seed_path)
        return json.dumps(doc, indent=1)

    @classmethod
    def from_json(cls, text: str) -> "ScenePlan":
        doc = json.loads(text)
        plan = cls(
            bbox_mm=tuple(doc["bbox_mm"]),
            walls=[Wall(**{**w, "start": tuple(w["start"]), "end": tuple(w["end"])})
                   for w in doc["walls"]],
            openings=[Opening(**o) for o in doc["openings"]],
            rooms=[Room(**{**r, "polygon": [tuple(p) for p in r["polygon"]]})
                   for r in doc["rooms"]],
            fixtures=[Fixture(**{**f, "rect_mm": tuple(f["rect_mm"])})
                      for f in doc["fixtures"]],
            texts=[TextItem(**{**t, "anchor_mm": tuple(t["anchor_mm"]),
                               "bbox_mm": tuple(t["bbox_mm"]) if t.get("bbox_mm") else None})
                   for t in doc["texts"]],
            meta=doc.get("meta", {}),
            seed_path=tuple(doc.get("seed_path", ())),
            schema_version=doc.get("schema_version", SCHEMA_VERSION),
        )
        return plan

    # -- GT export ----------------------------------------------------------
    def to_vlm_json(self, h: np.ndarray, extra_meta: dict | None = None) -> dict:
        """Export ground truth in the canonical VLM JSON format.

        `h` is the total mm -> GT-frame transform (similarity), typically
        N @ G @ A per the alignment contract. Lengths (thickness, opening
        width, curvature) scale by the isotropic factor of `h`.
        """
        s = h_scale(h)
        walls_out = []
        for w in self.walls:
            openings_out = []
            for o in self.openings_on(w.id):
                cx, cy = apply_h(h, w.point_at(o.center_offset_mm))
                openings_out.append({
                    "type": o.kind,
                    "center": [round(cx, 2), round(cy, 2)],
                    "width": round(o.width_mm * s, 2),
                })
            sx, sy = apply_h(h, w.start)
            ex, ey = apply_h(h, w.end)
            walls_out.append({
                "id": w.id,
                "start": [round(sx, 2), round(sy, 2)],
                "end": [round(ex, 2), round(ey, 2)],
                "thickness": round(w.thickness_mm * s, 2),
                "curvature": round(w.curvature_mm * s, 2),
                "type": w.role,
                "openings": openings_out,
            })
        def quad(points: list[Point]) -> list[list[float]]:
            return [[round(x, 2), round(y, 2)] for x, y in (apply_h(h, p) for p in points)]

        # Carry the room polygon (region boundary) alongside the wall-id cycle.
        # For open-plan rooms `walls` is a SUBSET of the boundary edges — the
        # polygon is what preserves the wall/open distinction downstream
        # (fpeval.from_vlm_json + verify's region-tiling watertightness).
        rooms_out = [{"label": r.label, "walls": list(r.wall_ids),
                      "polygon": quad(r.polygon)} for r in self.rooms]

        def aabb(q: list[list[float]]) -> list[float]:
            xs = [p[0] for p in q]
            ys = [p[1] for p in q]
            return [round(min(xs), 2), round(min(ys), 2), round(max(xs), 2), round(max(ys), 2)]

        room_labels = {r.id: r.label for r in self.rooms}
        fixtures_out = []
        for f in self.fixtures:
            q = quad(f.corners())
            fixtures_out.append({
                "kind": f.kind,
                "quad": q,
                "bbox": aabb(q),
                "room_label": room_labels.get(f.room_id, ""),
            })
        texts_out = []
        for t in self.texts:
            entry: dict = {"kind": t.kind, "text": t.text}
            if t.bbox_mm is not None:
                x, y, bw, bh = t.bbox_mm
                q = quad([(x, y), (x + bw, y), (x + bw, y + bh), (x, y + bh)])
                entry["quad"] = q
                entry["bbox"] = aabb(q)
            else:
                ax, ay = apply_h(h, t.anchor_mm)
                entry["anchor"] = [round(ax, 2), round(ay, 2)]
            texts_out.append(entry)

        return {
            "walls": walls_out,
            "rooms": rooms_out,
            "fixtures": fixtures_out,
            "texts": texts_out,
            "meta": {**self.meta, **(extra_meta or {})},
        }
