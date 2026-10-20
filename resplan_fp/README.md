# resplan_fp — the ResPlan-FP benchmark builder

ResPlan-FP is 16,998 residential plans from [ResPlan](https://arxiv.org/abs/2508.14006)
converted to wall-first vector ground truth and rendered under one frozen
protocol. The released data lives on the Hugging Face Hub
(`Cyprinus12138/resplan-fp`, CC BY 4.0; see `../data/README.md`). This package
is what produced it, so anyone can rebuild and verify it.

```bash
pip install -e ".[resplan]"
# ResPlan.pkl comes from the ResPlan release (MIT): https://github.com/... (see paper)
python -m resplan_fp.build --pkl ResPlan.pkl --out out/resplan_fp --workers 8
```

`build` converts every plan (`convert.py`), accepts those that pass the fpeval
validity check and the sequence-capacity gate (≤ 80 walls, < 512 tokens),
freezes the splits, renders the images and writes `train/val/test.jsonl`,
`images/`, `splits.json`, `stats.json`. It aborts if the split fingerprints
differ from `splits.json` in this directory, which is the published freeze.

| protocol element | value |
|---|---|
| split | ids sorted as strings, `random.Random(42).shuffle`, first 1000 test, next 1000 val, rest train |
| fingerprint | `sha256(",".join(sorted(ids)))[:16]` → test `669656eabb515960`, val `efb5e82b808acec4`, train `5af77996b313bba2` |
| render | `clean_band_v1`: 256 px, solid band, per-plan thickness `max(5, round(median thickness/4))`, white opening gaps, door arcs, window double lines (`render.py`) |
| frame | ground truth in the 1024 longer-edge frame |

`python -m resplan_fp.render rows.jsonl out_dir` re-renders any rows file.
The renderer reproduces the released PNGs byte for byte (checked on the full
test split and a random 300-plan sample of train/val).

Converter limitations are listed in the dataset card (`../data/cards/resplan-fp.md`).
