"""End-to-end pipeline properties: determinism, GT contract, manifest, and
fpeval round-trip (evaluate(gt, gt) ~ 1.0)."""

import json

import numpy as np
import pytest

from fpeval import evaluate, from_vlm_json, verify
from fpgen.pipeline import GenConfig, generate, generate_sample

CFG = GenConfig(seed=123, n=1)


def _sample(seed=123):
    root = np.random.SeedSequence(seed)
    seqs = root.spawn(CFG.sample_retries)
    last = None
    for seq in seqs:
        try:
            return generate_sample(seq, CFG, "000000")
        except Exception as e:  # rejects covered elsewhere
            last = e
    raise last


GT, IMAGE, SCENE_JSON, MANIFEST = _sample()


def test_gt_loads_in_fpeval_with_aux_keys():
    plan = from_vlm_json(GT, frame=1024.0)
    assert len(plan.walls) == MANIFEST["n_walls"]
    assert len(plan.rooms) == MANIFEST["n_rooms"]
    res = verify(plan)
    assert res.r_val == 1.0, res.issues


def test_self_evaluate_is_perfect():
    plan = from_vlm_json(GT, frame=1024.0)
    report = evaluate(plan, plan)
    assert report.walls["f1"] == pytest.approx(1.0)
    assert report.rooms["f1"] == pytest.approx(1.0)
    assert report.ext_iou > 0.99


def test_gt_coords_in_frame():
    for w in GT["walls"]:
        for x, y in (w["start"], w["end"]):
            assert -1 <= x <= 1025 and -1 <= y <= 1025


def test_deterministic():
    gt2, image2, scene2, manifest2 = _sample()
    assert json.dumps(GT, sort_keys=True) == json.dumps(gt2, sort_keys=True)
    assert np.array_equal(IMAGE, image2)
    assert SCENE_JSON == scene2


def test_manifest_schema():
    for key in ("id", "seed", "unit_meta", "style_family", "H", "mm_per_px",
                "img_w", "img_h", "verify", "rejects_before_accept"):
        assert key in MANIFEST
    assert MANIFEST["verify"]["r_val"] == 1.0


def test_scene_json_roundtrip():
    from fpgen.scene import ScenePlan
    plan = ScenePlan.from_json(SCENE_JSON)
    assert len(plan.walls) == MANIFEST["n_walls"]
    assert plan.to_json() == SCENE_JSON


def test_generate_run(tmp_path):
    cfg = GenConfig(seed=5, n=3, out_dir=str(tmp_path))
    summary = generate(cfg)
    assert summary["generated"] == 3
    assert summary["verify_pass"] == 3
    shard = tmp_path / summary["run_id"] / "shard-0000"
    assert len(list(shard.glob("*.png"))) == 3
    rows = [json.loads(l) for l in (shard / "manifest.jsonl").read_text().splitlines()]
    assert len(rows) == 3
