# Data

The datasets are hosted on the Hugging Face Hub, not in git:

| dataset | HF repo | license |
|---|---|---|
| ResPlan-FP (16,998 plans, renders + vector GT + baseline predictions) | `Tristan12138/resplan-fp` | CC BY 4.0 |
| CubiCasa5K skv4 annotations (3,328 / 318 / 296) | `Tristan12138/cubicasa5k-skv4` | CC BY-NC-SA 4.0 |

```bash
pip install huggingface_hub
python scripts/download_data.py all        # -> data/resplan-fp, data/cubicasa5k-skv4
```

Dataset cards (what each file is, split protocol, render protocol, licenses)
are in `cards/` and mirrored as each repo's README on the Hub.
