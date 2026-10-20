"""The ``clean_band_v1`` render protocol of ResPlan-FP.

Every benchmark image is produced by :func:`render_row` from the row's ground
truth alone: a white 256x256 canvas, each wall centerline drawn as a solid dark
band, every opening cut out of the band as a white gap, doors given a
quarter-circle swing arc plus leaf, windows a thin double line across the gap.
The band thickness is one integer per plan:

    thickness_px = max(5, round(median(wall thickness) * 256 / 1024))

This module is byte-for-byte the renderer used to build the released benchmark
(verified against the released images); keep it frozen. Re-render a jsonl with

    python -m resplan_fp.render rows.jsonl out_dir
"""
from __future__ import annotations

import math

import numpy as np

FRAME = 1024.0          # coordinate frame of the ground-truth JSON (longer edge)
SIZE = 256              # rendered image size (square canvas, FRAME -> SIZE)
MIN_THICKNESS_PX = 5
STYLE = "clean_band_v1"
INK = (20, 20, 20)
WHITE = (255, 255, 255)


def render(walls, openings=None, thickness_px=9, size=SIZE):
    """Render ``walls`` = [(a, b)] and ``openings`` = [(a, b, kind)] in FRAME
    coordinates onto a ``size`` x ``size`` BGR uint8 canvas."""
    from shapely.geometry import LineString
    import cv2
    sc = size / FRAME
    bands = [LineString([a * sc, b * sc]).buffer(thickness_px / 2, cap_style=2)
             for a, b in walls if math.dist(a, b) > 1]
    img = np.full((size, size, 3), 255, np.uint8)
    for band in bands:                     # fill each wall band separately (no room fill)
        polys = list(band.geoms) if hasattr(band, "geoms") else [band]
        for pp in polys:
            cv2.fillPoly(img, [np.array(pp.exterior.coords, np.int32)], INK)
    # openings: cut a white gap in the wall band + door-swing arc / window double-line,
    # so the image carries visual evidence of each opening.
    for a, b, kind in (openings or []):
        a2 = np.array(a) * sc; b2 = np.array(b) * sc
        w = float(np.linalg.norm(b2 - a2))
        if w < 2:
            continue
        d = (b2 - a2) / w; n = np.array([-d[1], d[0]])
        cv2.line(img, tuple(a2.astype(int)), tuple(b2.astype(int)), WHITE, thickness_px + 2)  # gap
        if kind == "door":
            th0 = math.atan2(d[1], d[0])                      # quarter-circle swing from a2
            arc = [a2 + w * np.array([math.cos(th0 + s), math.sin(th0 + s)])
                   for s in np.linspace(0, math.pi / 2, 14)]
            cv2.polylines(img, [np.array(arc).astype(np.int32)], False, INK, 1)
            leaf = a2 + n * w
            cv2.line(img, tuple(a2.astype(int)), tuple(leaf.astype(int)), INK, 1)
        else:                                                 # window: thin double line across the gap
            for off in (-2, 2):
                p0 = a2 + n * off; p1 = b2 + n * off
                cv2.line(img, tuple(p0.astype(int)), tuple(p1.astype(int)), INK, 1)
    return img


def geometry_from_gt(gt: dict, frame: float = FRAME):
    """Walls + opening spans (FRAME coordinates) from a wall-first GT document."""
    sc = FRAME / frame
    walls, openings = [], []
    for w in gt["walls"]:
        a = np.array(w["start"], float) * sc; b = np.array(w["end"], float) * sc
        walls.append((a, b))
        d = b - a; L = float(np.linalg.norm(d)) or 1.0; d = d / L
        for o in w.get("openings", []):
            c = np.array(o["center"], float) * sc; wd = o["width"] * sc
            openings.append((c - d * wd / 2, c + d * wd / 2, o.get("type", "door")))
    return walls, openings


def thickness_px_for(gt: dict, frame: float = FRAME) -> int:
    med = float(np.median([w["thickness"] for w in gt["walls"]])) * FRAME / frame
    return max(MIN_THICKNESS_PX, int(round(med * SIZE / FRAME)))


def render_row(row: dict):
    """Render one benchmark row (``row["gt"]`` + optional ``row["frame"]``)."""
    gt = row["gt"]; frame = float(row.get("frame", FRAME))
    walls, openings = geometry_from_gt(gt, frame)
    return render(walls, openings, thickness_px=thickness_px_for(gt, frame), size=SIZE)


def main():
    import argparse, json
    from pathlib import Path
    import cv2
    ap = argparse.ArgumentParser(prog="python -m resplan_fp.render",
                                 description="re-render a rows.jsonl under clean_band_v1")
    ap.add_argument("rows"); ap.add_argument("out_dir")
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    n = 0
    for line in open(a.rows):
        r = json.loads(line)
        cv2.imwrite(str(out / (r["id"].replace("/", "_") + ".png")), render_row(r)); n += 1
    print(f"rendered {n} plans -> {out}")


if __name__ == "__main__":
    main()
