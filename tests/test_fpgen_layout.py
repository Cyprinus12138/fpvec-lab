"""Layout procgen properties: every accepted layout is watertight, in-band,
and reachable. Style follows tests/test_fpeval.py (plain pytest, seeds)."""

import numpy as np
import pytest
from shapely.geometry import Polygon

from fpeval import from_vlm_json, verify
from fpgen import meta as meta_mod
from fpgen.layout import LayoutReject, generate_layout
from fpgen.openings import OpeningsReject, place_openings
from fpgen.programs import build_program

N_SEEDS = 40


def _make(seed):
    root = np.random.SeedSequence(seed)
    seqs = root.spawn(3)
    rng = np.random.default_rng(seqs[0])
    unit = meta_mod.sample(rng)
    program = build_program(unit, rng)
    layout, _ = generate_layout(program, unit, seqs[1])
    for child in seqs[2].spawn(20):
        try:
            layout.plan.openings = []
            place_openings(layout, np.random.default_rng(child))
            return unit, layout
        except OpeningsReject:
            continue
    return unit, None


def _accepted():
    out = []
    for seed in range(N_SEEDS):
        try:
            unit, layout = _make(seed)
        except LayoutReject:
            continue
        if layout is not None:
            out.append((unit, layout))
    return out


ACCEPTED = _accepted()


def test_acceptance_rate():
    # generator health: most seeds must survive layout+openings
    assert len(ACCEPTED) >= N_SEEDS * 0.6


def test_verify_watertight():
    h = np.eye(3)
    for _, layout in ACCEPTED:
        gt = layout.plan.to_vlm_json(h)
        res = verify(from_vlm_json(gt, frame=max(layout.plan.bbox_mm)))
        assert res.r_val == 1.0, res.issues


def test_room_areas_positive_and_sum():
    # rooms tile the envelope; L-shaped plans (corner notch) may lose up to
    # ~16% of the bbox rect, never gain
    for unit, layout in ACCEPTED:
        total = sum(r.area_sqm for r in layout.plan.rooms)
        bbox_sqm = layout.plan.bbox_mm[0] * layout.plan.bbox_mm[1] / 1e6
        assert total <= bbox_sqm * 1.02
        assert total >= bbox_sqm * 0.80


def test_min_segment_length():
    for _, layout in ACCEPTED:
        for w in layout.plan.walls:
            assert w.length >= 150.0


def test_windows_only_on_exterior():
    for _, layout in ACCEPTED:
        for o in layout.plan.openings:
            if o.kind != "window":
                continue
            l, r = layout.adjacency[o.wall_id]
            assert l is None or r is None


# ------------------------------------------------------------------ open-plan

def _make_open(seed):
    root = np.random.SeedSequence(seed)
    seqs = root.spawn(3)
    rng = np.random.default_rng(seqs[0])
    unit = meta_mod.sample(rng)
    unit.flags["open_plan"] = True                 # pin open-plan LDK
    program = build_program(unit, rng)
    layout, _ = generate_layout(program, unit, seqs[1])
    for child in seqs[2].spawn(20):
        try:
            layout.plan.openings = []
            place_openings(layout, np.random.default_rng(child))
            return unit, layout
        except OpeningsReject:
            continue
    return unit, None


def _accepted_open():
    out = []
    for seed in range(N_SEEDS):
        try:
            unit, layout = _make_open(seed)
        except LayoutReject:
            continue
        if layout is not None:
            out.append((unit, layout))
    return out


ACCEPTED_OPEN = _accepted_open()


def test_open_plan_actually_produces_open_edges():
    # with open_plan pinned, a meaningful fraction of layouts leave an LDK
    # boundary wall-less (kitchen must be adjacent to living/dining to trigger)
    with_open = [lay for _, lay in ACCEPTED_OPEN if lay.open_adjacency]
    assert with_open, "no open-plan layout produced an open edge"
    # such rooms carry more polygon edges than wall_ids (the open edge)
    room_has_open_edge = any(
        len(r.wall_ids) < len(r.polygon)
        for lay in with_open for r in lay.plan.rooms)
    assert room_has_open_edge


def test_open_plan_still_watertight_and_reachable():
    # open edges must NOT break watertightness (region-tiling) or reachability
    # (reachability already enforced by place_openings succeeding above)
    h = np.eye(3)
    for _, layout in ACCEPTED_OPEN:
        gt = layout.plan.to_vlm_json(h)
        res = verify(from_vlm_json(gt, frame=max(layout.plan.bbox_mm)))
        assert res.r_val == 1.0, res.issues


def test_open_edge_pairs_are_ldk_only():
    from fpgen.layout import OPEN_PLAN_ROLES
    for _, layout in ACCEPTED_OPEN:
        for pair in layout.open_adjacency:
            roles = {layout.specs[k].role for k in pair}
            assert roles <= OPEN_PLAN_ROLES, roles


def test_shelter_rules():
    # household shelter: thick walls, no window on any of its walls
    for _, layout in ACCEPTED:
        shelter_rooms = [k for k, s in layout.specs.items()
                         if s.role == "household_shelter"]
        for key in shelter_rooms:
            rid = layout.room_key_to_id[key]
            room = layout.plan.room_by_id()[rid]
            walls = layout.plan.wall_by_id()
            for wid in room.wall_ids:
                assert walls[wid].thickness_mm >= 250
                for o in layout.plan.openings:
                    if o.wall_id == wid:
                        assert o.kind != "window"


def test_openings_fit_walls():
    for _, layout in ACCEPTED:
        walls = layout.plan.wall_by_id()
        for o in layout.plan.openings:
            w = walls[o.wall_id]
            assert o.offset_mm >= 0
            assert o.offset_mm + o.width_mm <= w.length + 1


def test_room_polygons_tile_without_overlap():
    for _, layout in ACCEPTED:
        polys = [Polygon(r.polygon) for r in layout.plan.rooms]
        for i, a in enumerate(polys):
            for b in polys[i + 1:]:
                assert a.intersection(b).area < 1e4  # <0.01 sqm slivers
