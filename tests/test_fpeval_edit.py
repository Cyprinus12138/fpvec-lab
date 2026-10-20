"""fpeval edit-cost metric — the paper's human-edit-cost contribution (§5).

Constructs a 2-room open-plan GT (shared middle wall + one open boundary edge)
and two predictions with IDENTICAL rooms:
  - wall-first: shared wall once, open edge left open (correct by construction);
  - room-centric: the shared wall DOUBLED + a PHANTOM wall on the open edge
    (the r2s polygon->wall reconstruction failure mode).
Asserts the room-centric output costs strictly more to edit, driven by the
delete (double wall) + open_convert (phantom-over-open) ops — the exact
operations wall-first avoids by construction.
"""

import copy

from fpeval import evaluate
from fpeval.metrics import edit_cost
from fpeval.model import FloorPlan, Wall, Room

F = 1024.0
TOL = 0.015 * F

# shared coordinates
L = [(100, 100), (500, 100), (500, 500), (100, 500)]   # left room
R = [(500, 100), (900, 100), (900, 500), (500, 500)]   # right room; bottom OPEN


def _gt():
    walls = [
        Wall("w1", (100, 100), (500, 100)),
        Wall("w2", (500, 100), (900, 100)),
        Wall("w3", (900, 100), (900, 500)),
        Wall("w4", (100, 100), (100, 500)),
        Wall("w5", (100, 500), (500, 500)),
        Wall("w6", (500, 100), (500, 500)),   # shared middle wall (once)
        # (500,500)-(900,500) intentionally OPEN — no wall
    ]
    rooms = [Room("L", ["w1", "w6", "w5", "w4"], L),
             Room("R", ["w2", "w3", "w6"], R)]     # 3 walls, 4-edge polygon => open
    return FloorPlan(walls, rooms, F)


def _wall_first_pred():
    return _gt()  # correct by construction


def _room_centric_pred():
    p = _gt()
    # double the shared wall (near-coincident) + phantom wall on the open edge
    p.walls.append(Wall("w6b", (501, 100), (501, 500)))            # double wall
    p.walls.append(Wall("phantom", (500, 500), (900, 500)))        # phantom over open
    return p


def test_perfect_is_near_zero():
    ec = edit_cost(_gt(), _gt(), TOL)
    assert ec["total"] < 1e-6
    assert ec["ops"]["create"] == 0 and ec["ops"]["delete"] == 0


def test_room_centric_costs_more_to_edit():
    wf = edit_cost(_wall_first_pred(), _gt(), TOL)
    rc = edit_cost(_room_centric_pred(), _gt(), TOL)
    assert wf["total"] < 1e-6
    assert rc["total"] > wf["total"]
    # the extra cost is specifically a delete (double wall) + an open_convert
    assert rc["ops"]["delete"] >= 1, rc["ops"]
    assert rc["ops"]["open_convert"] >= 1, rc["ops"]
    # magnitude: >= one delete + one open_convert
    assert rc["total"] >= 5.0 + 10.0 - 1e-6


def test_recall_understates_edit_burden():
    """Both preds recall all GT walls; F1 differs only by precision, but the
    edit cost separates them much more sharply — the paper's point."""
    rc = evaluate(_room_centric_pred(), _gt())
    wf = evaluate(_wall_first_pred(), _gt())
    assert rc.walls["recall"] == wf.walls["recall"] == 1.0
    # F1 gap is modest (precision only), edit-cost gap is large
    assert wf.edit["total"] < 1e-6 and rc.edit["total"] >= 15.0
    assert rc.walls["f1"] > 0.5   # room-centric still looks "okay" on F1


def test_evaluate_exposes_edit():
    rep = evaluate(_gt(), _gt())
    assert "edit" in rep.as_dict()
    assert rep.edit["per_gt_wall"] < 1e-6
