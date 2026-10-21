"""Skia renderer: ScenePlan + StyleParams -> (RGB image, A matrix, text bboxes).

Everything is drawn in mm under a single affine A (mm -> px) set on the
canvas, so vector GT stays pixel-aligned by construction. Text bboxes
are measured with skia font metrics and backfilled onto plan.texts in mm.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import skia
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

from .scene import Opening, ScenePlan, TextItem, Wall
from .styles import StyleParams
from .symbols import GLYPHS

# Movable/soft furniture dropped by the line_cad bare skeleton. Built-in fixtures
# (counter, sink, stove, fridge, basin, bathtub, shower, toilet, washer, dryer,
# stairs, ac_unit) are kept — cc5k-style CAD plans still draw those.
SOFT_FURNITURE = frozenset({
    "bed", "sofa", "dining_table", "coffee_table", "desk", "tv_console",
    "bookshelf", "wardrobe", "shelving", "shoe_cabinet", "outdoor_chair",
    "planter_box",
})
from .vocab import area_text


def _color(rgb: tuple[int, int, int]) -> int:
    return skia.Color(*rgb)


class Draw:
    """mm-space drawing helper handed to glyphs (duck-typed)."""

    def __init__(self, canvas: skia.Canvas, style: StyleParams,
                 rng: np.random.Generator | None = None):
        self.canvas = canvas
        self.style = style
        self.rng = rng
        self.jitter = style.stroke_jitter_mm
        self.stroke = skia.Paint(
            AntiAlias=True, Style=skia.Paint.kStroke_Style,
            StrokeWidth=style.furniture_stroke_mm, Color=_color(style.stroke_color))

    def line(self, p1, p2, width: float | None = None):
        paint = self.stroke
        if width is not None:
            paint = skia.Paint(self.stroke)
            paint.setStrokeWidth(width)
        if self.jitter > 0 and self.rng is not None:
            # hand-drawn wobble: strokes only, GT geometry untouched
            path = skia.Path()
            path.moveTo(p1[0], p1[1])
            n = 3
            for k in range(1, n + 1):
                t = k / n
                jx = float(self.rng.normal(0, self.jitter)) if k < n else 0.0
                jy = float(self.rng.normal(0, self.jitter)) if k < n else 0.0
                path.lineTo(p1[0] + (p2[0] - p1[0]) * t + jx,
                            p1[1] + (p2[1] - p1[1]) * t + jy)
            self.canvas.drawPath(path, paint)
            return
        self.canvas.drawLine(p1[0], p1[1], p2[0], p2[1], paint)

    def rect(self, x, y, w, h, r: float = 0.0, fill=None):
        rr = skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(x, y, w, h), r, r)
        if fill is not None:
            self.canvas.drawRRect(rr, skia.Paint(
                AntiAlias=True, Style=skia.Paint.kFill_Style, Color=_color(fill)))
        self.canvas.drawRRect(rr, self.stroke)

    def rrect(self, x, y, w, h, r: float = 100.0):
        self.rect(x, y, w, h, r=r)

    def circle(self, cx, cy, r):
        self.canvas.drawCircle(cx, cy, r, self.stroke)

    def ellipse(self, cx, cy, rx, ry):
        self.canvas.drawOval(skia.Rect.MakeXYWH(cx - rx, cy - ry, 2 * rx, 2 * ry),
                             self.stroke)

    def arc(self, cx, cy, r, start_deg, sweep_deg, width: float | None = None):
        paint = self.stroke
        if width is not None:
            paint = skia.Paint(self.stroke)
            paint.setStrokeWidth(width)
        oval = skia.Rect.MakeXYWH(cx - r, cy - r, 2 * r, 2 * r)
        self.canvas.drawArc(oval, start_deg, sweep_deg, False, paint)

    def poly(self, points, fill=None, close=True, width: float | None = None):
        path = skia.Path()
        path.moveTo(points[0][0], points[0][1])
        for p in points[1:]:
            path.lineTo(p[0], p[1])
        if close:
            path.close()
        if fill is not None:
            self.canvas.drawPath(path, skia.Paint(
                AntiAlias=True, Style=skia.Paint.kFill_Style, Color=_color(fill)))
        else:
            paint = self.stroke
            if width is not None:
                paint = skia.Paint(self.stroke)
                paint.setStrokeWidth(width)
            self.canvas.drawPath(paint=paint, path=path)


def _shapely_path(geom) -> skia.Path:
    path = skia.Path()
    path.setFillType(skia.PathFillType.kEvenOdd)
    polys = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
    for poly in polys:
        for ring in [poly.exterior, *poly.interiors]:
            coords = list(ring.coords)
            path.moveTo(coords[0][0], coords[0][1])
            for x, y in coords[1:]:
                path.lineTo(x, y)
            path.close()
    return path


def wall_band_geometry(plan: ScenePlan):
    """Union of wall bands minus opening gaps (shapely, mm)."""
    bands = []
    for w in plan.walls:
        ls = LineString([w.start, w.end])
        # square caps extend t/2 past junctions, filling corner notches
        bands.append(ls.buffer(w.thickness_mm / 2, cap_style="square"))
    band = unary_union(bands)
    # clip cap overshoot beyond the envelope outer face (true footprint =
    # union of room polygons, so chamfered/L envelopes clip correctly)
    env = unary_union([Polygon(r.polygon) for r in plan.rooms]) if plan.rooms \
        else Polygon([(0, 0), (plan.bbox_mm[0], 0),
                      (plan.bbox_mm[0], plan.bbox_mm[1]), (0, plan.bbox_mm[1])])
    ext_t = max((w.thickness_mm for w in plan.walls if w.role == "exterior"),
                default=250.0)
    band = band.intersection(env.buffer(ext_t / 2, join_style="mitre"))
    walls = plan.wall_by_id()
    gaps = []
    for o in plan.openings:
        w = walls[o.wall_id]
        p1 = w.point_at(o.offset_mm)
        p2 = w.point_at(o.offset_mm + o.width_mm)
        gaps.append(LineString([p1, p2]).buffer(w.thickness_mm / 2 + 1.0,
                                                cap_style="flat"))
    if gaps:
        band = band.difference(unary_union(gaps))
    return band


def _swing_normal(w: Wall, o: Opening) -> tuple[float, float]:
    n = w.normal()
    return (-n[0], -n[1]) if o.swing == "left" else n


def _draw_opening_symbol(d: Draw, w: Wall, o: Opening, style: StyleParams):
    p1 = w.point_at(o.offset_mm)
    p2 = w.point_at(o.offset_mm + o.width_mm)
    dirv = w.direction()
    n = w.normal()
    t = w.thickness_mm
    sw = style.opening_stroke_mm

    if o.kind == "window":
        k = style.window_lines
        offsets = [0.0] if k == 1 else [
            -t / 2 + t * (i + 1) / (k + 1) for i in range(k)]
        for off in offsets:
            d.line((p1[0] + n[0] * off, p1[1] + n[1] * off),
                   (p2[0] + n[0] * off, p2[1] + n[1] * off), width=sw)
        for p in (p1, p2):  # jambs
            d.line((p[0] - n[0] * t / 2, p[1] - n[1] * t / 2),
                   (p[0] + n[0] * t / 2, p[1] + n[1] * t / 2), width=sw)
        return

    if o.kind == "sliding_door":
        half = o.width_mm * 0.55
        for sign, start in ((-1, p1), (1, p2)):
            end = (start[0] + dirv[0] * half * (1 if sign < 0 else -1),
                   start[1] + dirv[1] * half * (1 if sign < 0 else -1))
            off = sign * t / 5
            d.line((start[0] + n[0] * off, start[1] + n[1] * off),
                   (end[0] + n[0] * off, end[1] + n[1] * off), width=sw)
        return

    if o.kind == "door":
        sn = _swing_normal(w, o)
        hinge, jamb = (p1, p2) if o.hinge == "start" else (p2, p1)
        r = o.width_mm
        leaf_end = (hinge[0] + sn[0] * r, hinge[1] + sn[1] * r)
        leaf_w = sw * (2.2 if o.params.get("heavy") else 1.2)
        d.line(hinge, leaf_end, width=leaf_w)
        jd = ((jamb[0] - hinge[0]) / r, (jamb[1] - hinge[1]) / r)
        start_deg = math.degrees(math.atan2(jd[1], jd[0]))
        sweep = 90.0 if (jd[0] * sn[1] - jd[1] * sn[0]) > 0 else -90.0
        d.arc(hinge[0], hinge[1], r, start_deg, sweep, width=sw * 0.7)
    # kind == "opening": rendered as a plain gap


def _draw_annotations(plan: ScenePlan, d: Draw, style: StyleParams,
                      rng: np.random.Generator) -> None:
    """Dimension chains (top + left) and scale bar — true metric values, the
    scale-grounding signal. Appends TextItems; bboxes backfilled by the
    main text pass."""
    from .vocab import dim_text
    bw, bh = plan.bbox_mm
    sw = style.opening_stroke_mm * 0.7

    def chain(edge: str) -> None:
        pts = set()
        for w in plan.walls:
            for p in (w.start, w.end):
                if edge == "top" and abs(p[1]) < 1.0:
                    pts.add(p[0])
                if edge == "left" and abs(p[0]) < 1.0:
                    pts.add(p[1])
        stops = sorted(pts)
        if len(stops) < 2:
            return
        off = -450.0
        for a, b in zip(stops, stops[1:]):
            if b - a < 400:
                continue
            if edge == "top":
                d.line((a, off), (b, off), width=sw)
                d.line((a, off - 100), (a, off + 100), width=sw)
                d.line((b, off - 100), (b, off + 100), width=sw)
                anchor = ((a + b) / 2, off - style.dim_font_mm * 0.75)
            else:
                d.line((off, a), (off, b), width=sw)
                d.line((off - 100, a), (off + 100, a), width=sw)
                d.line((off - 100, b), (off + 100, b), width=sw)
                anchor = (off - style.dim_font_mm * 1.6, (a + b) / 2)
            plan.texts.append(TextItem(
                id=f"text_{len(plan.texts) + 1}", kind="dimension",
                text=dim_text(b - a), anchor_mm=anchor,
                font_size_mm=style.dim_font_mm))

    if style.show_dims:
        chain("top")
        chain("left")

    if style.title_text:
        plan.texts.append(TextItem(
            id=f"text_{len(plan.texts) + 1}", kind="logo", text=style.title_text,
            anchor_mm=(bw / 2, -max(700.0, style.label_font_mm * 2.2)),
            font_size_mm=style.label_font_mm * 1.15))

    if style.show_scalebar:
        meters = 2 if bw < 9000 else 3
        x0, y = 200.0, bh + 600.0
        for k in range(meters):
            seg = (x0 + k * 1000, x0 + (k + 1) * 1000)
            d.rect(seg[0], y - 60, 1000, 120,
                   fill=(0, 0, 0) if k % 2 == 0 else style.bg_color)
        plan.texts.append(TextItem(
            id=f"text_{len(plan.texts) + 1}", kind="scalebar_text",
            text=f"0        {meters} m",
            anchor_mm=(x0 + meters * 500, y + style.dim_font_mm),
            font_size_mm=style.dim_font_mm))


@dataclass
class RenderResult:
    image: np.ndarray  # HxWx3 uint8 RGB
    a_matrix: np.ndarray  # 3x3 mm -> px
    px_per_mm: float


def render(plan: ScenePlan, style: StyleParams,
           rng: np.random.Generator) -> RenderResult:
    bw, bh = plan.bbox_mm
    long_mm = max(bw, bh)
    target_long_px = float(rng.uniform(900, 1600))
    lo_margin = 0.08 if (style.show_dims or style.show_scalebar
                         or style.title_text) else 0.05
    margin_frac = float(rng.uniform(lo_margin, 0.13))
    margin_mm = margin_frac * long_mm
    s = target_long_px / (long_mm + 2 * margin_mm)
    img_w = int(math.ceil((bw + 2 * margin_mm) * s))
    img_h = int(math.ceil((bh + 2 * margin_mm) * s))
    a = np.array([[s, 0.0, s * margin_mm],
                  [0.0, s, s * margin_mm],
                  [0.0, 0.0, 1.0]])

    surface = skia.Surface(img_w, img_h)
    canvas = surface.getCanvas()
    canvas.clear(_color(style.bg_color))
    canvas.concat(skia.Matrix.MakeAll(s, 0, s * margin_mm, 0, s, s * margin_mm,
                                      0, 0, 1))
    d = Draw(canvas, style, rng)

    # floor fills
    if style.floor_fills:
        for room in plan.rooms:
            fill = style.floor_fills.get(room.role)
            if fill:
                canvas.drawPath(_shapely_path(Polygon(room.polygon)), skia.Paint(
                    AntiAlias=True, Style=skia.Paint.kFill_Style,
                    Color=_color(fill)))

    # wall bands
    band = wall_band_geometry(plan)
    if style.wall_render == "hollow":
        canvas.drawPath(_shapely_path(band), skia.Paint(
            AntiAlias=True, Style=skia.Paint.kFill_Style,
            Color=_color(style.bg_color)))
        canvas.drawPath(_shapely_path(band), skia.Paint(
            AntiAlias=True, Style=skia.Paint.kStroke_Style,
            StrokeWidth=style.opening_stroke_mm, Color=_color(style.wall_fill)))
    else:
        canvas.drawPath(_shapely_path(band), skia.Paint(
            AntiAlias=True, Style=skia.Paint.kFill_Style,
            Color=_color(style.wall_fill)))

    # opening symbols
    walls = plan.wall_by_id()
    for o in plan.openings:
        _draw_opening_symbol(d, walls[o.wall_id], o, style)

    # fixtures (counters were placed before in-counter fixtures, order holds).
    # line_cad bare skeleton drops movable/soft furniture (beds, sofas, tables,
    # wardrobes, ...) but keeps built-in fixtures cc5k-style plans still draw
    # (sanitary, kitchen, laundry, stairs) — these don't enter the wall-first GT.
    for f in plan.fixtures:
        if not style.draw_soft_furniture and f.kind in SOFT_FURNITURE:
            continue
        fn = GLYPHS.get(f.kind)
        if fn is None:
            fn = GLYPHS["shelving"]  # safe fallback outline
        x, y, w, h = f.rect_mm
        canvas.save()
        canvas.translate(x + w / 2, y + h / 2)
        canvas.rotate(f.rotation_deg)
        fn(d, w, h, style, rng, f.params)
        canvas.restore()

    _draw_annotations(plan, d, style, rng)

    # text (labels created by the pipeline; bboxes backfilled here)
    fm = skia.FontMgr()
    typeface = fm.matchFamilyStyle(style.font_name, skia.FontStyle.Normal())
    if typeface is None and fm.countFamilies():
        typeface = fm.matchFamilyStyle(fm.getFamilyName(0),
                                       skia.FontStyle.Normal())
    if typeface is None:
        raise RuntimeError("no usable font family found for text rendering")
    for t in plan.texts:
        if not t.text:
            continue
        size = t.font_size_mm or (
            style.label_font_mm if t.kind == "room_label" else style.area_font_mm)
        t.font_size_mm = size
        font = skia.Font(typeface, size)
        tw = font.measureText(t.text)
        metrics = font.getMetrics()
        text_h = -metrics.fAscent + metrics.fDescent
        x = t.anchor_mm[0] - tw / 2
        baseline = t.anchor_mm[1] - metrics.fAscent - text_h / 2
        canvas.drawString(t.text, x, baseline, font, skia.Paint(
            AntiAlias=True, Color=_color(style.stroke_color)))
        t.bbox_mm = (x, baseline + metrics.fAscent, tw, text_h)

    img = surface.makeImageSnapshot().toarray(
        colorType=skia.kRGBA_8888_ColorType)[:, :, :3].copy()
    return RenderResult(image=img, a_matrix=a, px_per_mm=s)
