# fpgen — synthetic floorplan data engine

Generates stylized SG-condo floorplan rasters with pixel-perfect vector ground
truth. Implements REQ-6 (pixel-aligned supervision via a tracked `H = N·G·A`
transform) and REQ-8 (symbol priors + document realism), with SG conventions
mined from a corpus of 18k Singapore residential listings (room-role
vocabulary, printed labels, area bands, unit-mix statistics); only the fitted
aggregate statistics ship here (`assets/meta_stats.json`), no listing content.

## Quick start (from the repository root; needs `skia-python` and the
`DejaVu LGC Sans` system font)

```bash
# generate a mixed-style dataset
python -m fpgen generate -n 1000 --seed 1 \
    --style clean_cad:0.3,marketing:0.35,brochure:0.2,hand_drawn:0.15 \
    --workers 10

# open-plan LDK (kitchen~living left wall-less -> open room edge); default off
python -m fpgen generate -n 200 --seed 1 --open-plan on

# eyeball: GT-over-raster overlays + contact sheet
python -m fpgen overlay out/fpgen/<run_id>

# refit unit-mix stats from a listing-title corpus (the source corpus is not redistributed)
python fpgen/scripts/fit_meta.py --input <titles.jsonl>
```

## Architecture (4-layer compositional pipeline)

1. **Layout** (`meta.py`, `programs.py`, `layout.py`, `openings.py`) — unit
   meta sampled from corpus-fitted P(bed, bath) → SG room-program tree
   (adjacency by construction: corridor rows, service clusters, carves for
   balcony/foyer/AC-ledge/WIC) → min-width-clamped proportional guillotine
   splits → planar wall graph with thickness classes (shelter 250–400 mm on
   ALL sides) → doors along the access tree + windows on exterior walls +
   BFS reachability. Envelope variety: corner notch (L-plans) and chamfer
   (non-Manhattan diagonals).
2. **Semantics** (`rules.py` = declarative data, `semantics.py` = solver) —
   room-role→fixture rules (toilet only in bath/WC, bed+wardrobe in bedrooms,
   counter runs + stove/sink in kitchens, washer in yard, empty shelters…)
   solved by rejection sampling with collision, clearance, and door-swing
   obstacles. Extend by adding a `FixtureRule` + a glyph.
3. **Render** (`styles.py`, `symbols.py`, `render.py`) — skia, everything in
   mm under one affine `A`; families: clean_cad / marketing / brochure /
   hand_drawn (stroke-jitter only — GT untouched). Dimension chains + scale
   bar carry TRUE metric values (REQ-5). Labels sampled from the SG printed
   vocabulary; controlled label-on-wall occlusion (p≈0.35, frac recorded).
4. **Degrade** (`degrade.py`) — stage 1: similarity `G` (rot ≤3°, scale
   0.85–1.05) tracked exactly; stage 2: appearance-only whitelist
   (`SAFE_OPS`: Augraphy BadPhotoCopy/InkBleed/BleedThrough/… + own
   blur/noise/JPEG/tiled-watermark ops). Whitelist honesty is enforced by
   `tests/test_fpgen_render.py::test_registration_alignment` (registration
   crosses within 1.5 px after the full pipeline).

## Contracts

- **GT** = fpeval's canonical VLM JSON (`from_vlm_json`), coords on the
  0..1024 longer-edge grid, plus aux keys fpeval ignores: `fixtures` (quads +
  room labels — symbol supervision), `texts` (quads), `meta`.
- **QA gate**: every emitted sample passes `fpeval.verify` (r_val = 1.0) or is
  rejected and re-drawn; per-cause reject counts land in `summary.json`.
- **Determinism**: same `GenConfig` → byte-identical PNG + GT, independent of
  `--workers` (seeds derive from `(seed, spawn_key=(idx·retries+attempt,))`).
- **Output**: `out/fpgen/<run>/shard-NNNN/{id.png, id.gt.json, id.scene.json}`
  + `manifest.jsonl` (+ run `summary.json`). `scene.json` is the mm-canonical
  `ScenePlan` — re-render new styles without re-running layout.
- **RESERVED SEEDS**: `[777000, 778000)` belong to the paper's frozen synthetic
  evaluation tiers (777001–4; 777004 = the open-plan tier). Do not train on
  seeds in this range if you want to compare against the paper's numbers.

## Deliberately out of scope (v1)

Multi-unit / multi-storey / dual-key plans; curved walls (schema field
`curvature_mm` exists, renderer/eval tolerance test pending); perspective-class
degradation (thickness stops being a scalar — gated behind M4); ResPlan
real-layout arm (`resplan.py`, M4); degradation calibration vs the real corpus
(`calibrate.py`, M4).

Note: Autodesk US12518066B2 (granted 2026) covers room-conditioned synthetic
symbol placement for training detectors. fpgen is internal training tooling,
not a shipped product feature — flagged for legal review before any
productization of the generator itself.
