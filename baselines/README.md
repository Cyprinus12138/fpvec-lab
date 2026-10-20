# ResPlan-FP baseline tracks

Predictions for every track are shipped with the benchmark
(`data/resplan-fp/baselines/*.preds.jsonl`, `{"id", "pred"}` per line, plans in
the 1024 frame) so the table below can be re-scored without any model:

```bash
python scripts/download_data.py resplan-fp
python -m baselines.score --bench data/resplan-fp --preds data/resplan-fp/baselines/zeroshot_seq_decode.preds.jsonl
python -m baselines.score --bench data/resplan-fp --preds data/resplan-fp/baselines/llm_gemini.preds.jsonl \
    --ids data/resplan-fp/baselines/llm_subset_ids.json
```

Protocol: all plans of the test split count, a missing or null prediction
scores zero, coverage is reported. Wall F1 is the fpeval wall match at
tolerance × 1024; rooms, openings and edit cost come from `fpeval.evaluate`.

| track | file | n | covered | wall F1 @0.05 | @0.015 |
|---|---|---|---|---|---|
| zero-shot, wall-first sequence decode | `zeroshot_seq_decode` | 1000 | 1000 | 0.640 | 0.505 |
| zero-shot, graph readout, thresholds frozen on CubiCasa5K | `zeroshot_readout_frozen` | 1000 | 1000 | 0.621 | 0.553 |
| zero-shot, graph readout, thresholds calibrated on ResPlan-FP val | `zeroshot_readout_calibrated` | 1000 | 1000 | 0.677 | 0.570 |
| zero-shot, room-centric (Raster2Seq continued from its checkpoint) | `zeroshot_roomcentric` | 1000 | 733 | 0.383 | 0.325 |
| zero-shot, fusion (room-centric base + frozen-readout donor, readout fallback) | `zeroshot_fusion` | 1000 | 1000 | 0.688 | 0.599 |
| finetuned, wall-first sequence decode | `finetuned_seq_decode` | 1000 | 1000 | 0.968 | 0.957 |
| finetuned, graph readout (val-calibrated thresholds) | `finetuned_readout` | 1000 | 1000 | 0.915 | 0.896 |
| finetuned, room-centric (Raster2Seq finetuned on ResPlan-FP train) | `finetuned_roomcentric` | 1000 | 1000 | 0.928 | 0.902 |
| LLM zero-shot, Gemini 3.1 Pro, seeded 200-plan subset | `llm_gemini` | 200 | 200 | 0.811 | 0.472 |

The zero-shot room-centric model emits no rooms on 267 plans (scored zero);
the fusion row falls back to the readout walls there. The LLM file is the raw
completion parsed with fence stripping, array-level JSON repair and a
well-formedness filter (walls need numeric `start`/`end`), the same
extraction used for the paper. How each system is trained and read out is in
the paper (Sections 3–5 and Appendix "LLM zero-shot protocol").

`score.py` also works on the skv4 splits (`--bench data/cubicasa5k-skv4`).
