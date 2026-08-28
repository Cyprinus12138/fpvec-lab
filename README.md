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
| CubiCasa5K scans | 296 | 0.790 | **0.818** | detect (+2.7, and +5.1 @ 0.015) |
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
| `fpeval/` | evaluator: wall/room/opening F1 over a tolerance sweep, watertightness + double-wall joint read, and the **edit-cost metric** (human correction effort; ranking invariant over the admissible weight region) | MIT |
| `resplan-fp/` | **ResPlan-FP** benchmark builder: 16,998 plans, frozen fingerprinted splits (14,998 / 1,000 / 1,000, seed 42), published render protocol, three baseline tracks (zero-shot / finetuned / LLM) | code MIT, data CC BY 4.0 |
| `skv4/` | corrected CubiCasa5K annotation converter (room-cycle closure, ink gate, spur pruning); re-scoring a fixed checkpoint against skv4 moves wall F1 0.74 → 0.825 | CC BY-NC-SA 4.0 (inherits CubiCasa5K) |
| `fpgen/` | procedural floorplan generator: unit statistics fitted to 18k real listings, four render style families, tracked degradations, fpeval-gated ground truth | MIT |
| `paper/` | LaTeX source of the paper | — |

<!-- TODO: adjust directory names to the final layout before publishing -->

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
benchmark so any later copy can be verified.

## Data licensing, precisely

- **ResPlan-FP** renders and labels derive from [ResPlan](https://arxiv.org/abs/2508.14006) (data CC BY 4.0, code MIT) — redistributable, released under **CC BY 4.0**.
- **skv4** derives from [CubiCasa5K](https://github.com/CubiCasa/CubiCasa5k) (CC BY-NC-SA 4.0) — released under **CC BY-NC-SA 4.0**, research use.
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
