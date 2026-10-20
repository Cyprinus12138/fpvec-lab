"""Eyeball tooling: GT-over-raster overlays and contact sheets."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

WALL_COLOR = (230, 60, 60, 200)
OPENING_COLORS = {"door": (40, 160, 240, 255), "sliding_door": (40, 110, 240, 255),
                  "opening": (120, 200, 120, 255), "window": (250, 170, 30, 255)}
FIXTURE_COLOR = (60, 200, 120, 160)
TEXT_COLOR = (200, 60, 200, 180)


def overlay_sample(png_path: Path, gt_path: Path) -> Image.Image:
    """Draw GT vectors (1024-frame) over the rendered raster."""
    img = Image.open(png_path).convert("RGBA")
    gt = json.loads(gt_path.read_text())
    s = max(img.size) / 1024.0
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    for w in gt["walls"]:
        a = (w["start"][0] * s, w["start"][1] * s)
        b = (w["end"][0] * s, w["end"][1] * s)
        draw.line([a, b], fill=WALL_COLOR,
                  width=max(1, int(w.get("thickness", 10) * s * 0.5)))
        for o in w.get("openings", []):
            cx, cy = o["center"][0] * s, o["center"][1] * s
            r = max(3, o["width"] * s / 2)
            color = OPENING_COLORS.get(o.get("type", "door"), (0, 0, 0, 255))
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=color, width=3)

    for f in gt.get("fixtures", []):
        pts = [(p[0] * s, p[1] * s) for p in f["quad"]]
        draw.polygon(pts, outline=FIXTURE_COLOR, width=2)

    for t in gt.get("texts", []):
        if "quad" in t:
            pts = [(p[0] * s, p[1] * s) for p in t["quad"]]
            draw.polygon(pts, outline=TEXT_COLOR, width=2)

    return Image.alpha_composite(img, layer).convert("RGB")


def run_overlays(run_dir: Path, take: int | None = None) -> list[Path]:
    """Write <id>.overlay.png next to each sample; returns written paths."""
    out = []
    pngs = sorted(run_dir.glob("shard-*/*.png"))
    pngs = [p for p in pngs if not p.name.endswith(".overlay.png")]
    if take:
        pngs = pngs[:take]
    for png in pngs:
        gt = png.with_suffix("").with_suffix(".gt.json")
        if not gt.exists():
            continue
        ov = overlay_sample(png, gt)
        dest = png.with_name(png.stem + ".overlay.png")
        ov.save(dest)
        out.append(dest)
    return out


def contact_sheet(run_dir: Path, cols: int = 5, rows: int = 4,
                  cell: int = 380, overlays: bool = True) -> Path:
    """Grid of the first cols*rows samples (overlaid unless overlays=False)."""
    suffix = ".overlay.png" if overlays else ".png"
    if overlays:
        run_overlays(run_dir, take=cols * rows)
        images = sorted(run_dir.glob(f"shard-*/*{suffix}"))[: cols * rows]
    else:
        images = [p for p in sorted(run_dir.glob("shard-*/*.png"))
                  if not p.name.endswith(".overlay.png")][: cols * rows]
    sheet = Image.new("RGB", (cols * cell, rows * cell), (245, 245, 245))
    for i, path in enumerate(images):
        img = Image.open(path).convert("RGB")
        img.thumbnail((cell - 8, cell - 8))
        x = (i % cols) * cell + (cell - img.width) // 2
        y = (i // cols) * cell + (cell - img.height) // 2
        sheet.paste(img, (x, y))
    dest = run_dir / f"contact_sheet{'_overlay' if overlays else ''}.png"
    sheet.save(dest)
    return dest
