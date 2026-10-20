# fpvec-lab

**When should a network emit geometry, and when should it detect it?**
Code, benchmarks, and metrics for floorplan vectorization research.

This repository accompanies the paper:

> **When Should a Network Emit Geometry, and When Should It Detect It?
> Readout, Reconciliation, and Representation in Floorplan Vectorization**
> He Zhang — [arXiv:2608.25608](https://arxiv.org/abs/2608.25608)

## Findings in one table

Same trained network, two ways to read the walls out of it — an autoregressive
sequence decoder (**emit**) and an untrained graph readout over its junction /
centerline heatmaps (**detect**), wall F1 @ 0.05:

| domain | n | emit | detect | winner |
|---|---|---|---|---|
| CubiCasa5K scans (skv4 test) | 296 | 0.790 | **0.818** | detect (+2.7, and +5.1 @ 0.015) |
| ResPlan-FP zero-shot | 1000 | **0.640** | 0.621 frozen / 0.677 calibrated | thresholds decide |
| ResPlan-FP finetuned | 1000 | **0.968** | 0.915 | emit |
| open-plan synthetic | 30 | **0.908** | 0.825 | emit |

Within real scans the readout's advantage grows with plan size and reverses on
small plans. Also in the paper: reading a never-used opening heatmap lifts
opening F1 from 0.25 to 0.64 with no retraining; deterministic output-level
fusion of a room-centric and a wall-first system adds +7.1 wall F1, while
input-side conditioning gives 0/3 under a pre-registered stop rule; and with
matched data and recipe, the output representation (rooms-first vs walls-first)
matters less than evaluation convention and post-processing.

## What's here

| component | what it is | license |
|---|---|---|
| `fpeval/` | evaluator: wall/room/opening F1 over a tolerance sweep, validity + watertightness, and the **edit-cost metric** (human correction effort; ranking invariant over the admissible weight region). `python -m fpeval verify|eval` | MIT |
| `resplan_fp/` | **ResPlan-FP** benchmark builder: converter, frozen fingerprinted splits (14,998 / 1,000 / 1,000, seed 42), the `clean_band_v1` render protocol; rebuilds the released benchmark byte for byte | code MIT, data CC BY 4.0 |
| `baselines/` | scorer + the shipped predictions of all baseline tracks (zero-shot / finetuned / LLM) | MIT |
| `skv4/` | corrected CubiCasa5K annotation converter (wall-graph face rooms, ink gate, spur pruning) + official-split partition; re-scoring a fixed checkpoint against skv4 moves wall F1 0.74 → 0.825 | non-commercial: vendored CubiCasa5K parser CC BY-NC 4.0, data CC BY-NC-SA 4.0 |
| `fpgen/` | procedural floorplan generator: unit statistics fitted to 18k real listings, four render style families, tracked degradations, fpeval-gated ground truth | MIT |
| `fpseq/` | torch-free wall-first sequence codec (plan ↔ events ↔ tensors); the converters use its capacity gate | MIT |
| `data/` | dataset cards + download script; the data itself is on the Hugging Face Hub | — |
| `paper/` | LaTeX source of the paper | — |

## Install

```bash
git clone https://github.com/Cyprinus12138/fpvec-lab && cd fpvec-lab
pip install -e ".[all]"            # or pick extras: fpgen, skv4, resplan, data, test
python -m pytest tests -q          # 118 tests, no data needed
python scripts/download_data.py all   # -> data/resplan-fp, data/cubicasa5k-skv4
```

`fpgen` needs the `DejaVu LGC Sans` system font; `skv4` needs cairosvg (libcairo).

## Quick start

```bash
# score a prediction file against the ResPlan-FP test split
python -m baselines.score --bench data/resplan-fp --preds data/resplan-fp/baselines/finetuned_seq_decode.preds.jsonl
# evaluate one plan against one ground truth (wall-first JSON, 1024 frame)
python -m fpeval eval pred.json gt.json --tol-frac 0.05
# rebuild ResPlan-FP from ResPlan.pkl (aborts if the split fingerprints differ from the published freeze)
python -m resplan_fp.build --pkl ResPlan.pkl --out out/resplan_fp --workers 8
# rebuild skv4 from the CubiCasa5K release
python -m skv4.convert --svg-dir <cc5k> --out out/skv4_raw && python -m skv4.split --converted out/skv4_raw --cc5k-root <cc5k> --out out/skv4
# generate synthetic plans
python -m fpgen generate -n 100 --seed 1 --style clean_cad:0.3,marketing:0.35,brochure:0.2,hand_drawn:0.15
```

## ResPlan-FP baseline tracks (wall F1 @ 0.05)

| track | score |
|---|---|
| zero-shot, fusion system | 0.688 |
| zero-shot, sequence decode alone | 0.640 |
| zero-shot, graph readout alone | 0.621 (frozen thr.) / 0.677 (calibrated) |
| finetuned, wall-first decode | 0.968 |
| finetuned, room-centric | 0.928 |
| LLM zero-shot (Gemini 3.1 Pro, 200-plan subset) | 0.811 @ 0.05 / 0.472 @ 0.015 |

Splits are frozen by id-hash fingerprint; the full test-id list ships with the
benchmark so any later copy can be verified. Every row is re-scorable from the
shipped prediction files (`baselines/README.md`).

## Data licensing, precisely

- **ResPlan-FP** renders and labels derive from [ResPlan](https://arxiv.org/abs/2508.14006) (data CC BY 4.0, code MIT) — redistributable, released under **CC BY 4.0**.
- **skv4** derives from [CubiCasa5K](https://github.com/CubiCasa/CubiCasa5k) (data CC BY-NC-SA 4.0 per its Zenodo record; reference code CC BY-NC 4.0) — released under **CC BY-NC-SA 4.0**, research use.
- Synthetic data (`fpgen`) uses only aggregate listing statistics; no listing text or imagery is included.

## Citation

```bibtex
@article{zhang2026emitdetect,
  title   = {When Should a Network Emit Geometry, and When Should It Detect It?
             Readout, Reconciliation, and Representation in Floorplan Vectorization},
  author  = {Zhang, He},
  journal = {arXiv preprint arXiv:2608.25608},
  year    = {2026}
}
```
