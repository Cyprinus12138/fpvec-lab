"""Sample orchestration: seeds -> layout -> openings -> furnish -> label ->
render -> QA gate -> shard files + manifest.jsonl.

Determinism contract: one np.random.SeedSequence per run; every sample and
every internal retry derives from spawned children, so the same GenConfig
produces byte-identical outputs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import Polygon

from . import degrade
from . import meta as meta_mod
from .layout import LayoutReject, generate_layout
from .openings import OpeningsReject, place_openings
from .programs import build_program
from .qa import QAReject, gate
from .render import render
from .scene import TextItem
from .semantics import SemanticsReject, furnish
from .styles import sample_style
from .vocab import area_text, pick_label

FRAME = 1024.0
SHARD_SIZE = 1000


@dataclass
class GenConfig:
    seed: int = 0
    n: int = 10
    out_dir: str = "out/fpgen"
    run_id: str | None = None
    style_mix: dict[str, float] = field(default_factory=lambda: {"clean_cad": 1.0})
    typology: str | None = None  # pin e.g. "2B2b"; None = sample from meta stats
    degrade_level: str | None = None  # none|light|medium|heavy; None = sample
    open_plan: bool | None = None  # pin open-plan LDK on/off; None = sample per plan
    sample_retries: int = 6
    workers: int = 1


@dataclass
class SampleRecord:
    id: str
    ok: bool
    manifest: dict | None = None
    rejects: dict[str, int] = field(default_factory=dict)


def _pick_style_family(rng: np.random.Generator, mix: dict[str, float]) -> str:
    keys = sorted(mix)
    probs = np.array([mix[k] for k in keys], dtype=float)
    return keys[rng.choice(len(keys), p=probs / probs.sum())]


def _label_rooms(layout, style, rng) -> None:
    """Assign printed labels + text items (labels land in GT room entries).

    With p~0.35 per plan one label is deliberately shifted onto a wall band —
    the REQ-8 controlled partial-occlusion condition; the target fraction is
    recorded in plan.meta.
    """
    from .semantics import _inward_normal

    plan = layout.plan
    id_to_key = {v: k for k, v in layout.room_key_to_id.items()}
    occlude_room = None
    occl_frac = 0.0
    labelable = [r for r in plan.rooms if r.area_sqm >= 4.0]
    if labelable and rng.random() < 0.35:
        occlude_room = labelable[int(rng.integers(len(labelable)))].id
        occl_frac = float(rng.uniform(0.1, 0.6))

    walls = plan.wall_by_id()
    for room in plan.rooms:
        spec = layout.specs[id_to_key[room.id]]
        room.label = pick_label(rng, spec.label_role, spec.label_index)
        if not room.label:
            continue
        poly = Polygon(room.polygon)
        anchor = poly.representative_point()
        # rooms too small for their label at this font get a smaller font
        font = style.label_font_mm
        est_w = font * 0.62 * len(room.label)
        room_w = poly.bounds[2] - poly.bounds[0]
        if est_w > room_w * 0.9:
            font = max(140.0, font * room_w * 0.9 / est_w)
        ax, ay = anchor.x, anchor.y
        if room.id == occlude_room and room.wall_ids:
            long_walls = sorted(room.wall_ids,
                                key=lambda wid: -walls[wid].length)
            wall = walls[long_walls[0]]
            n = _inward_normal(layout, wall, id_to_key[room.id])
            mid = wall.point_at(wall.length * float(rng.uniform(0.35, 0.65)))
            off = wall.thickness_mm / 2 + font * (0.5 - occl_frac)
            ax, ay = mid[0] + n[0] * off, mid[1] + n[1] * off
        plan.texts.append(TextItem(
            id=f"text_{len(plan.texts) + 1}", kind="room_label", text=room.label,
            anchor_mm=(ax, ay), font_size_mm=font, room_id=room.id))
        if style.show_area and room.area_sqm >= 3.0:
            plan.texts.append(TextItem(
                id=f"text_{len(plan.texts) + 1}", kind="area",
                text=area_text(rng, room.area_sqm),
                anchor_mm=(anchor.x, anchor.y + font * 1.05),
                font_size_mm=style.area_font_mm, room_id=room.id))
    plan.meta["occlusion_frac"] = round(occl_frac, 3)


def generate_sample(sample_seq: np.random.SeedSequence, cfg: GenConfig,
                    sample_id: str) -> tuple[dict, np.ndarray, str, dict]:
    """One sample attempt chain. Returns (gt, image, scene_json, manifest)."""
    rejects: dict[str, int] = {}
    (meta_seq, layout_seq, open_seq, furnish_seq, style_seq,
     render_seq) = sample_seq.spawn(6)

    rng_meta = np.random.default_rng(meta_seq)
    unit = meta_mod.sample(rng_meta, typology=cfg.typology)
    if cfg.open_plan is not None:
        unit.flags["open_plan"] = cfg.open_plan
    program = build_program(unit, rng_meta)

    layout, layout_rejects = generate_layout(program, unit, layout_seq)
    for k, v in layout_rejects.items():
        rejects[f"layout:{k}"] = rejects.get(f"layout:{k}", 0) + v

    placed = False
    for child in open_seq.spawn(20):
        try:
            layout.plan.openings = []
            place_openings(layout, np.random.default_rng(child))
            placed = True
            break
        except OpeningsReject as e:
            rejects[f"openings:{e.cause}"] = rejects.get(f"openings:{e.cause}", 0) + 1
    if not placed:
        raise OpeningsReject("exhausted")

    furnished = False
    for child in furnish_seq.spawn(8):
        try:
            layout.plan.fixtures = []
            stats = furnish(layout, np.random.default_rng(child))
            furnished = True
            break
        except SemanticsReject as e:
            rejects[f"semantics:{e.cause}"] = rejects.get(f"semantics:{e.cause}", 0) + 1
    if not furnished:
        raise SemanticsReject("exhausted")

    rng_style = np.random.default_rng(style_seq)
    family = _pick_style_family(rng_style, cfg.style_mix)
    style = sample_style(rng_style, family)
    # Bare skeleton (line_cad): drop soft furniture at the PLAN level so the GT
    # fixtures list stays consistent with the rendered image — otherwise the
    # labels would claim furniture the image doesn't show. Built-in fixtures
    # (kitchen/bath/laundry/stairs) stay. Wall-first GT ignores fixtures anyway,
    # but keeping image==GT matters for any fixture-aware consumer downstream.
    if not style.draw_soft_furniture:
        from .render import SOFT_FURNITURE
        layout.plan.fixtures = [f for f in layout.plan.fixtures
                                if f.kind not in SOFT_FURNITURE]
    if style.title_text:
        style.title_text = style.title_text.format(
            n_bed=max(1, unit.n_bed))
    _label_rooms(layout, style, rng_style)

    rr = render(layout.plan, style, np.random.default_rng(render_seq))
    rng_degrade = np.random.default_rng(render_seq.spawn(1)[0])
    image, g_mat, degrade_ops = degrade.apply(
        rr.image, rng_degrade, style.bg_color, level=cfg.degrade_level)
    img_h, img_w = image.shape[:2]
    n_scale = FRAME / max(img_w, img_h)
    n_mat = np.diag([n_scale, n_scale, 1.0])
    h_total = n_mat @ g_mat @ rr.a_matrix

    plan = layout.plan
    plan.seed_path = tuple(int(x) for x in np.atleast_1d(sample_seq.entropy))
    gt = gate(plan, h_total, frame=FRAME)

    manifest = {
        "id": sample_id,
        "seed": plan.seed_path,
        "unit_meta": plan.meta["unit_meta"],
        "style_family": family,
        "degrade_ops": degrade_ops,
        "H": [[round(v, 8) for v in row] for row in h_total.tolist()],
        "mm_per_px": round(1.0 / rr.px_per_mm, 5),
        "img_w": img_w,
        "img_h": img_h,
        "n_walls": len(plan.walls),
        "n_rooms": len(plan.rooms),
        "n_openings": len(plan.openings),
        "n_fixtures": len(plan.fixtures),
        "occlusion_frac": plan.meta.get("occlusion_frac", 0.0),
        "dropped_fixtures": stats.dropped,
        "unfurnished_rooms": stats.unfurnished_rooms,
        "verify": gt["meta"]["verify"],
        "rejects_before_accept": rejects,
    }
    gt["meta"].update({"seed": manifest["seed"], "style": family,
                       "degrade_ops": degrade_ops,
                       "mm_per_px": manifest["mm_per_px"],
                       "img_w": img_w, "img_h": img_h})
    return gt, image, plan.to_json(), manifest


def _produce_one(cfg: GenConfig, root_str: str, idx: int
                 ) -> tuple[int, dict | None, dict[str, int]]:
    """One sample incl. retries + file writes (multiprocessing-safe: file
    names are unique per idx; seeds derive from (seed, spawn_key) so results
    are identical regardless of worker count)."""
    sample_id = f"{idx:06d}"
    shard = Path(root_str) / f"shard-{idx // SHARD_SIZE:04d}"
    rejects: dict[str, int] = {}
    result = None
    for attempt in range(cfg.sample_retries):
        seq = np.random.SeedSequence(
            cfg.seed, spawn_key=(idx * cfg.sample_retries + attempt,))
        try:
            result = generate_sample(seq, cfg, sample_id)
            break
        except (LayoutReject, OpeningsReject, SemanticsReject, QAReject) as e:
            cause = f"{type(e).__name__}:{getattr(e, 'cause', '?')}"
            rejects[cause] = rejects.get(cause, 0) + 1
    if result is None:
        rejects["sample_exhausted"] = rejects.get("sample_exhausted", 0) + 1
        return idx, None, rejects
    gt, image, scene_json, manifest = result
    cv2.imwrite(str(shard / f"{sample_id}.png"),
                cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    (shard / f"{sample_id}.gt.json").write_text(json.dumps(gt, indent=1))
    (shard / f"{sample_id}.scene.json").write_text(scene_json)
    manifest["shard"] = shard.name
    return idx, manifest, rejects


def generate(cfg: GenConfig) -> dict:
    """Run the generator; returns the run summary."""
    run_id = cfg.run_id or f"run_seed{cfg.seed}_n{cfg.n}"
    root = Path(cfg.out_dir) / run_id
    root.mkdir(parents=True, exist_ok=True)
    for s in range((cfg.n + SHARD_SIZE - 1) // SHARD_SIZE):
        (root / f"shard-{s:04d}").mkdir(exist_ok=True)

    summary_rejects: dict[str, int] = {}
    manifest_rows: list[dict] = []
    if cfg.workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=cfg.workers) as pool:
            results = pool.map(_produce_one, [cfg] * cfg.n, [str(root)] * cfg.n,
                               range(cfg.n), chunksize=8)
            results = list(results)
    else:
        results = [_produce_one(cfg, str(root), idx) for idx in range(cfg.n)]

    for _, manifest, rejects in sorted(results, key=lambda r: r[0]):
        for cause, count in rejects.items():
            summary_rejects[cause] = summary_rejects.get(cause, 0) + count
        if manifest is not None:
            manifest_rows.append(manifest)

    by_shard: dict[str, list[dict]] = {}
    for m in manifest_rows:
        by_shard.setdefault(m["shard"], []).append(m)
    for shard_name, rows in by_shard.items():
        with (root / shard_name / "manifest.jsonl").open("w") as fh:
            for m in rows:
                fh.write(json.dumps(m) + "\n")

    summary = {
        "run_id": run_id,
        "requested": cfg.n,
        "generated": len(manifest_rows),
        "verify_pass": sum(1 for m in manifest_rows
                           if m["verify"]["r_val"] == 1.0),
        "rejects": summary_rejects,
        "out": str(root),
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary
