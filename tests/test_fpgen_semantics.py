"""Semantic constraint properties: fixtures land only in legal rooms, never
overlap, never block door swings."""

import math

import numpy as np
from shapely.geometry import Polygon

from fpgen.layout import LayoutReject, generate_layout
from fpgen.openings import OpeningsReject, place_openings
from fpgen.programs import build_program
from fpgen.rules import RULES
from fpgen import meta as meta_mod
from fpgen.semantics import SemanticsReject, door_obstacles, furnish

N_SEEDS = 30


def _furnished():
    out = []
    for seed in range(N_SEEDS):
        root = np.random.SeedSequence(seed)
        seqs = root.spawn(4)
        rng = np.random.default_rng(seqs[0])
        unit = meta_mod.sample(rng)
        program = build_program(unit, rng)
        try:
            layout, _ = generate_layout(program, unit, seqs[1])
        except LayoutReject:
            continue
        ok = False
        for child in seqs[2].spawn(20):
            try:
                layout.plan.openings = []
                place_openings(layout, np.random.default_rng(child))
                ok = True
                break
            except OpeningsReject:
                continue
        if not ok:
            continue
        for child in seqs[3].spawn(8):
            try:
                layout.plan.fixtures = []
                furnish(layout, np.random.default_rng(child))
                out.append(layout)
                break
            except SemanticsReject:
                continue
    return out


FURNISHED = _furnished()
ROLE_OF = {}
for layout in FURNISHED:
    for key, spec in layout.specs.items():
        ROLE_OF[(id(layout), layout.room_key_to_id[key])] = spec.role


def test_have_samples():
    assert len(FURNISHED) >= N_SEEDS * 0.5


def test_fixture_room_semantics():
    # the user's core constraint: toilets only in bath/wc, beds only in
    # bedrooms, washers only in yard/utility, stoves only in kitchens
    legal = {
        "toilet": {"bath", "wc"},
        "bed": {"bedroom"},
        "washer": {"yard", "utility"},
        "stove": {"kitchen"},
        "sofa": {"living_dining", "living"},
    }
    for layout in FURNISHED:
        for f in layout.plan.fixtures:
            role = ROLE_OF[(id(layout), f.room_id)]
            if f.kind in legal:
                assert role in legal[f.kind], (f.kind, role)
            forbidden = RULES.get(role)
            if forbidden:
                assert f.kind not in forbidden.forbidden, (f.kind, role)


def test_required_fixtures_present():
    for layout in FURNISHED:
        by_room: dict[str, set] = {}
        for f in layout.plan.fixtures:
            by_room.setdefault(f.room_id, set()).add(f.kind)
        for key, spec in layout.specs.items():
            rid = layout.room_key_to_id[key]
            if spec.role in ("bath", "wc"):
                assert "toilet" in by_room.get(rid, set()), (key, spec.role)
            if spec.role == "bedroom":
                assert "bed" in by_room.get(rid, set()), key


def test_no_fixture_overlap():
    for layout in FURNISHED:
        polys = [(f, Polygon(f.corners())) for f in layout.plan.fixtures]
        for i, (fa, pa) in enumerate(polys):
            for fb, pb in polys[i + 1:]:
                if fa.room_id != fb.room_id:
                    continue
                if "counter" in (fa.kind, fb.kind) and (
                        fa.params.get("in_counter") or fb.params.get("in_counter")):
                    continue
                assert pa.intersection(pb).area < 1000, (fa.kind, fb.kind)


def test_fixtures_inside_rooms():
    for layout in FURNISHED:
        rooms = layout.plan.room_by_id()
        for f in layout.plan.fixtures:
            rpoly = Polygon(rooms[f.room_id].polygon)
            assert Polygon(f.corners()).within(rpoly.buffer(5)), f.kind


def test_door_swings_unblocked():
    for layout in FURNISHED:
        id_to_key = {v: k for k, v in layout.room_key_to_id.items()}
        for room in layout.plan.rooms:
            obstacles = door_obstacles(layout, id_to_key[room.id])
            for f in layout.plan.fixtures:
                if f.room_id != room.id:
                    continue
                fpoly = Polygon(f.corners())
                for ob in obstacles:
                    assert fpoly.intersection(ob).area < 1000, f.kind
