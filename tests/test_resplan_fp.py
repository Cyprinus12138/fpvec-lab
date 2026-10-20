"""ResPlan-FP protocol tests that need no data: frozen-split fingerprints and the renderer."""
import json
import pathlib

import numpy as np

from resplan_fp.build import FROZEN, fingerprint, freeze_splits
from resplan_fp.render import SIZE, render_row, thickness_px_for


def test_frozen_test_ids_match_fingerprint():
    sp = json.loads(pathlib.Path(FROZEN).read_text())
    assert len(sp["test_ids"]) == sp["counts"]["test"] == 1000
    assert fingerprint(sp["test_ids"]) == sp["id_sha256"]["test"]


def test_freeze_splits_is_deterministic_and_disjoint():
    ids = [f"resplan/{i}" for i in range(5000)]
    a = freeze_splits(ids, n_test=100, n_val=100)
    b = freeze_splits(list(reversed(ids)), n_test=100, n_val=100)   # input order must not matter
    assert a == b
    assert len(a["test"]) == 100 and len(a["val"]) == 100 and len(a["train"]) == 4800
    assert not (set(a["test"]) & set(a["val"])) and not (set(a["test"]) & set(a["train"]))


def _toy_gt(th=16.0):
    sq = [[100, 100], [900, 100], [900, 900], [100, 900]]
    walls = []
    for i in range(4):
        walls.append({"id": f"wall_{i+1}", "start": sq[i], "end": sq[(i + 1) % 4], "thickness": th,
                      "type": "exterior", "openings": []})
    walls[0]["openings"].append({"type": "door", "center": [500, 100], "width": 80})
    walls[1]["openings"].append({"type": "window", "center": [900, 500], "width": 120})
    return {"walls": walls, "rooms": []}


def test_thickness_rule():
    assert thickness_px_for(_toy_gt(16.0)) == 5        # round(4.0) -> clamped to 5
    assert thickness_px_for(_toy_gt(25.4)) == 6        # round(6.35) -> 6


def test_render_row_shape_and_ink():
    img = render_row({"gt": _toy_gt(), "frame": 1024.0})
    assert img.shape == (SIZE, SIZE, 3) and img.dtype == np.uint8
    dark = (img[:, :, 0] < 100)
    assert 0.02 < dark.mean() < 0.3                     # a wall band, not blank, not filled
    assert not dark[25, 125]                            # door gap (center x=500 -> 125 px) on the top wall is white
    assert dark[25, 60]                                 # the top wall band itself is ink
    assert not dark[128, 128]                           # room interior stays white (no fill)
    assert np.array_equal(img, render_row({"gt": _toy_gt(), "frame": 1024.0}))   # deterministic
