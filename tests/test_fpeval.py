"""fpeval unit tests. Run from the repository root: python -m pytest tests/ -q"""

import copy
import json
import math
import pathlib

import pytest

from fpeval import evaluate, from_vlm_json, verify
from fpeval.metrics import alpha_gate, ext_iou

FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures"


def square_plan(size=800.0, off=100.0, thickness=12.0):
    """Closed 4-wall square with a door on the south wall and one room."""
    a, b = off, off + size
    return from_vlm_json({
        "walls": [
            {"id": "w1", "start": [a, a], "end": [b, a], "thickness": thickness,
             "openings": []},
            {"id": "w2", "start": [b, a], "end": [b, b], "thickness": thickness,
             "openings": [{"type": "window", "center": [b, (a + b) / 2], "width": 100}]},
            {"id": "w3", "start": [b, b], "end": [a, b], "thickness": thickness,
             "openings": [{"type": "door", "center": [(a + b) / 2, b], "width": 90}]},
            {"id": "w4", "start": [a, b], "end": [a, a], "thickness": thickness,
             "openings": []},
        ],
        "rooms": [{"label": "living", "walls": ["w1", "w2", "w3", "w4"]}],
    })


# ------------------------------------------------------------------ verify

def test_verify_closed_square_is_watertight():
    res = verify(square_plan())
    assert res.valid, res.issues
    assert res.watertight, res.issues
    assert res.r_val == 1.0
    assert res.n_faces == 1


def test_verify_open_loop_fails_watertight():
    plan = square_plan()
    plan.walls = plan.walls[:3]          # remove west wall -> loop open
    plan.rooms[0].wall_ids = ["w1", "w2", "w3"]
    res = verify(plan)
    assert not res.watertight
    assert res.r_val == 0.0


def openplan_two_room(right_x0=400.0):
    """Two rooms tiling [0,800]x[0,400] with NO divider wall at x=400 — the
    shared boundary is an OPEN edge. Rooms carry explicit polygons; each has
    fewer wall_ids than polygon edges. `right_x0` shifts the kitchen to create
    overlap (<400) or a gap (>400) for the negative cases."""
    return from_vlm_json({
        "walls": [
            {"id": "bl", "start": [0, 0], "end": [400, 0], "thickness": 12},
            {"id": "br", "start": [400, 0], "end": [800, 0], "thickness": 12},
            {"id": "r", "start": [800, 0], "end": [800, 400], "thickness": 12},
            {"id": "tr", "start": [800, 400], "end": [400, 400], "thickness": 12},
            {"id": "tl", "start": [400, 400], "end": [0, 400], "thickness": 12},
            {"id": "l", "start": [0, 400], "end": [0, 0], "thickness": 12},
        ],
        "rooms": [
            {"label": "living", "walls": ["bl", "l", "tl"],
             "polygon": [[0, 0], [400, 0], [400, 400], [0, 400]]},
            {"label": "kitchen", "walls": ["br", "r", "tr"],
             "polygon": [[right_x0, 0], [800, 0], [800, 400], [right_x0, 400]]},
        ],
    })


def test_verify_openplan_open_edge_is_watertight():
    plan = openplan_two_room()
    # representation: a room has fewer wall_ids than polygon edges (open edge)
    assert any(len(r.wall_ids) < len(r.polygon) for r in plan.rooms)
    res = verify(plan)
    assert res.valid, res.issues
    assert res.watertight, res.issues        # region-tiling path, no wall divider
    assert res.r_val == 1.0


def test_verify_openplan_overlap_fails():
    res = verify(openplan_two_room(right_x0=300.0))   # kitchen overlaps living
    assert not res.watertight
    assert any("overlap" in i for i in res.issues), res.issues


def test_verify_openplan_gap_fails():
    res = verify(openplan_two_room(right_x0=500.0))   # gap between the rooms
    assert not res.watertight


def test_verify_flags_bad_references_and_degenerates():
    plan = square_plan()
    plan.rooms[0].wall_ids.append("w99")            # missing ref
    plan.walls[0].openings.append(                   # opening off its wall
        type(plan.walls[1].openings[0])(kind="door", center=(500, 500), width=90))
    res = verify(plan)
    assert not res.valid
    assert any("w99" in i for i in res.issues)
    assert any("off its wall" in i for i in res.issues)


# ------------------------------------------------------------------ metrics

def test_perfect_match_scores_one():
    gt = square_plan()
    rep = evaluate(square_plan(), gt)
    assert rep.ext_iou > 0.99
    assert rep.walls["f1"] == 1.0
    assert rep.openings["f1"] == 1.0
    assert rep.rooms["f1"] == 1.0
    assert rep.reward > 0.99


def test_shifted_walls_reduce_ext_iou_and_report_distance():
    gt = square_plan()
    pred = square_plan(off=110.0)        # 10-unit shift, inside match tolerance
    rep = evaluate(pred, gt)
    assert rep.walls["f1"] == 1.0
    assert 8.0 < rep.walls["mean_dist"] < 12.0
    assert 0.9 < rep.ext_iou < 1.0


def test_missing_wall_hits_recall_and_watertightness():
    gt = square_plan()
    pred = square_plan()
    pred.walls = pred.walls[:3]
    pred.rooms[0].wall_ids = ["w1", "w2", "w3"]
    rep = evaluate(pred, gt)
    assert rep.r_val == 0.0
    assert rep.walls["recall"] == 0.75
    assert rep.walls["missing_frac"] == 0.25


def test_wrong_opening_kind_not_matched():
    gt = square_plan()
    pred = square_plan()
    pred.walls[2].openings[0].kind = "window"   # door -> window on south wall
    rep = evaluate(pred, gt)
    assert rep.openings["matched"] == 1          # only the real window pairs up
    assert rep.openings["f1"] == 0.5


def test_alpha_gate_shape():
    assert alpha_gate(0.0) == 0.1
    assert alpha_gate(0.3) == 0.1
    assert alpha_gate(0.7) == 1.0
    assert alpha_gate(1.0) == 1.0
    assert 0.1 < alpha_gate(0.5) < 1.0
    assert math.isclose(alpha_gate(0.5), 0.55)


def test_hallucinated_walls_hit_precision():
    """The Ballota lesson: additive invention must be punished, not rewarded."""
    gt = square_plan()
    pred = square_plan()
    pred.walls.append(type(pred.walls[0])(
        id="w_phantom", start=(100, 500), end=(900, 500), thickness=12))
    rep = evaluate(pred, gt)
    assert rep.walls["precision"] < 1.0
    assert rep.walls["recall"] == 1.0


# ------------------------------------------------------------------ fixtures

def test_ballota_fixture_loads_and_selfevaluates():
    doc = json.loads((FIXTURES / "manual_ballota_v5.dsl.json").read_text())
    gt = from_vlm_json(doc)
    assert len(gt.walls) == 24
    rep = evaluate(from_vlm_json(doc), gt)
    assert rep.walls["f1"] == 1.0
    assert rep.ext_iou > 0.99
    assert rep.openings["f1"] == 1.0


def test_ballota_v4_vs_v5_shows_the_manual_diff():
    v4 = from_vlm_json(json.loads((FIXTURES / "manual_ballota_v4.dsl.json").read_text()))
    v5 = from_vlm_json(json.loads((FIXTURES / "manual_ballota_v5.dsl.json").read_text()))
    rep = evaluate(v4, v5)
    # v4 had phantom walls (additive errors) -> precision below 1, envelope similar
    assert rep.ext_iou > 0.8
    assert rep.walls["f1"] < 1.0
