"""fpseq codec round-trip: fpeval FloorPlan <-> events <-> tensors <-> FloorPlan.

Scenes are synthesized in-process via fpgen (open_plan on AND off, mixed
typologies) — mirroring tests/test_fpgen_pipeline.py's sampling — so the suite is
self-contained (no run10k artifacts required). The gates:

  * lossless geometry: wall endpoints, opening (t,width), room polygons;
  * fpeval-valid: verify(decoded).r_val == 1.0 and evaluate(decoded, gt) all F1
    == 1.0 (the load-bearing contract — decode must produce a scoreable plan);
  * semantics (wall type, opening kind, room role) survive;
  * grammar well-formedness (every emitted stream parses under LEGAL_NEXT);
  * augmentation equivariance across all 8 flip x rot90 symmetries;
  * capacity bounds (< SEQ_LEN tokens, <= MAX_WALLS walls);
  * role-vocab drift guard against fpgen.vocab.
"""

import math

import numpy as np
import pytest
from shapely.geometry import Polygon

import fpseq
from fpseq import canonical, schema
from fpgen.pipeline import GenConfig, generate_sample
from fpgen.scene import ScenePlan
from fpgen import vocab
from fpeval import evaluate, from_vlm_json, verify

FRAME = 1024.0


def _sample(seed: int, open_plan: bool):
    cfg = GenConfig(seed=seed, n=1, open_plan=open_plan)
    for seq in np.random.SeedSequence(seed).spawn(cfg.sample_retries):
        try:
            return generate_sample(seq, cfg, "000000")
        except Exception:
            continue
    return None


def _corpus():
    """(plan, roles, open_plan, tag) for ~30 scenes, generated once."""
    out = []
    for seed in range(15):
        for op in (False, True):
            s = _sample(seed, op)
            if s is None:
                continue
            gt, _img, scene_json, _man = s
            plan = from_vlm_json(gt, frame=FRAME)
            roles = [r.role for r in ScenePlan.from_json(scene_json).rooms]
            out.append((plan, roles, op, f"seed{seed}_{'op' if op else 'std'}"))
    return out


CORPUS = _corpus()
OPEN_COMBOS = [(flip, k) for flip in (False, True) for k in range(4)]


def _roundtrip(plan, roles):
    ev = fpseq.plan_to_events(plan, roles=roles)
    tensors = fpseq.events_to_tensors(ev)
    ev2 = fpseq.tensors_to_events(tensors)
    dec = fpseq.events_to_plan(ev2)
    return ev, tensors, ev2, dec


def test_corpus_nonempty_and_mixed():
    assert len(CORPUS) >= 20, f"only {len(CORPUS)} scenes generated"
    assert any(op for *_, op, _ in CORPUS), "no open-plan scenes"
    assert any(not op for *_, op, _ in CORPUS), "no standard scenes"


def test_semantic_class_layout():
    assert schema.SEMANTIC_CLASSES == (
        len(schema.ROOM_ROLES) + len(schema.WALL_TYPES)
        + len(schema.OPENING_KINDS) + 1)
    assert schema.SEMANTIC_CLASSES == 33
    # round-trip every class id
    for r in schema.ROOM_ROLES:
        assert schema.class_role(schema.role_class(r)) == r
    for t in schema.WALL_TYPES:
        assert schema.class_wall_type(schema.wall_type_class(t)) == t
    for k in schema.OPENING_KINDS:
        assert schema.class_opening_kind(schema.opening_kind_class(k)) == k


def test_role_vocab_no_drift():
    """ROOM_ROLES is the frozen superset of Room.role; keep it in sync with the
    fpgen printed-label vocabulary (labeling is keyed by role)."""
    assert set(schema.ROOM_ROLES) == set(vocab.LABELS), (
        "fpseq.schema.ROOM_ROLES drifted from fpgen.vocab.LABELS keys")
    observed = {role for _p, roles, _op, _t in CORPUS for role in roles}
    assert observed <= set(schema.ROOM_ROLES), (
        f"roles not covered by ROOM_ROLES: {observed - set(schema.ROOM_ROLES)}")


@pytest.mark.parametrize("idx", range(len(CORPUS)))
def test_roundtrip_identity(idx):
    plan, roles, _op, tag = CORPUS[idx]
    ev, tensors, ev2, dec = _roundtrip(plan, roles)

    # fpeval contract — the load-bearing gate
    v = verify(dec)
    assert v.r_val == 1.0, f"{tag}: verify failed {v.issues}"
    rep = evaluate(dec, plan)
    assert rep.walls["f1"] == pytest.approx(1.0), f"{tag} wall F1 {rep.walls}"
    assert rep.rooms["f1"] == pytest.approx(1.0), f"{tag} room F1 {rep.rooms}"
    assert rep.openings["f1"] == pytest.approx(1.0), f"{tag} open F1 {rep.openings}"
    # ext_iou uses order-sensitive endpoint welding + thickness buffering, so a
    # canonically-reordered (but geometrically identical) plan lands ~0.994, not
    # exactly 1.0 — wall/room F1 above are the exact-geometry gates.
    assert rep.ext_iou > 0.99, f"{tag} ext_iou {rep.ext_iou}"

    # wall geometry + type, compared in canonical order (encode is deterministic)
    cw = canonical.canonical_walls(plan)
    assert len(cw) == len(dec.walls)
    for a, b in zip(cw, dec.walls):
        for pa, pb in ((a.start, b.start), (a.end, b.end)):
            assert math.dist(pa, pb) <= 1e-3, f"{tag} endpoint {pa}->{pb}"
        assert a.type == b.type
        # openings on this wall: match by nearest center (encode sorts by t)
        bs = list(b.openings)
        assert len(a.openings) == len(bs)
        for oa in a.openings:
            j = min(range(len(bs)),
                    key=lambda k: math.dist(oa.center, bs[k].center))
            ob = bs.pop(j)
            # center bound is loose: GT rounds coords to 2 dp, nudging the
            # center off the chord; the codec stores the on-chord foot.
            assert math.dist(oa.center, ob.center) <= 3e-2, f"{tag} opening center"
            assert abs(oa.width - ob.width) <= 1e-3, f"{tag} opening width"
            assert oa.kind == ob.kind

    # room polygons geometrically equal (extra collinear vertices are fine)
    for r, d in zip(plan.rooms, dec.rooms):
        if not (r.polygon and d.polygon and len(d.polygon) >= 3):
            continue
        gp, dp = Polygon(r.polygon), Polygon(d.polygon)
        if gp.area > 0:
            assert gp.symmetric_difference(dp).area / gp.area <= 1e-3, tag

    # room role survives through events<->tensors
    assert [e.role for e in ev.rooms] == [e.role for e in ev2.rooms], f"{tag} roles"


def test_grammar_wellformed():
    """Every encoded token stream must parse under the LEGAL_NEXT grammar (the
    day-1 grammar-masked decoder relies on this table matching the encoder)."""
    for plan, roles, _op, tag in CORPUS:
        tt = fpseq.events_to_tensors(fpseq.plan_to_events(plan, roles))["token_type"]
        prev = None
        for ty in tt:
            ty = schema.TT(int(ty))
            assert ty in schema.LEGAL_NEXT[prev], f"{tag}: {prev}->{ty} illegal"
            prev = ty
        assert prev == schema.TT.EOS, f"{tag}: stream does not end in EOS"


def test_open_edges_only_in_open_plan():
    """OPEN markers appear in open-plan scenes and never spuriously in a fully
    closed scene (the ring-seam false-open regression guard)."""
    def n_open(plan, roles):
        ev = fpseq.plan_to_events(plan, roles)
        return sum(1 for r in ev.rooms for m in r.members if m[0] == "OPEN")

    assert any(n_open(p, r) > 0 for p, r, op, _ in CORPUS if op), \
        "open-plan scenes produced no OPEN edges"
    # standard scenes: every room boundary is fully walled -> zero opens
    for p, r, op, tag in CORPUS:
        if not op:
            assert n_open(p, r) == 0, f"{tag}: spurious OPEN in closed scene"


def test_capacity_bounds():
    for plan, roles, _op, tag in CORPUS:
        tensors = fpseq.events_to_tensors(fpseq.plan_to_events(plan, roles))
        assert len(tensors["token_type"]) < schema.SEQ_LEN, f"{tag} too long"
        assert len(plan.walls) <= schema.MAX_WALLS, f"{tag} too many walls"


@pytest.mark.parametrize("idx", range(len(CORPUS)))
def test_aug_equivariance(idx):
    """Canonical ordering is recomputed post-augmentation, so the codec must
    round-trip identically under all 8 flip x rot90 symmetries."""
    plan, roles, _op, tag = CORPUS[idx]
    for flip, k in OPEN_COMBOS:
        pt = canonical.transform(plan, flip, k)
        _ev, _t, _ev2, dec = _roundtrip(pt, roles)
        v = verify(dec)
        assert v.r_val == 1.0, f"{tag} flip={flip} k={k}: {v.issues}"
        rep = evaluate(dec, pt)
        assert rep.walls["f1"] == pytest.approx(1.0), f"{tag} f{flip}k{k} wall"
        assert rep.rooms["f1"] == pytest.approx(1.0), f"{tag} f{flip}k{k} room"
        assert rep.openings["f1"] == pytest.approx(1.0), f"{tag} f{flip}k{k} open"
