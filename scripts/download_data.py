"""Download the released datasets from the Hugging Face Hub and unpack the image tars.

    python scripts/download_data.py resplan-fp          # -> data/resplan-fp/
    python scripts/download_data.py cubicasa5k-skv4     # -> data/cubicasa5k-skv4/
    python scripts/download_data.py all

Set HF_USER to override the dataset namespace (default: Cyprinus12138).
"""
from __future__ import annotations

import os
import sys
import tarfile
from pathlib import Path

NAMES = ("resplan-fp", "cubicasa5k-skv4")
USER = os.environ.get("HF_USER", "Cyprinus12138")
ROOT = Path(__file__).resolve().parents[1] / "data"


def fetch(name: str):
    from huggingface_hub import snapshot_download
    dst = ROOT / name
    snapshot_download(repo_id=f"{USER}/{name}", repo_type="dataset", local_dir=str(dst))
    for tar in sorted(dst.glob("images*.tar")):
        print("unpacking", tar.name)
        with tarfile.open(tar) as tf:
            tf.extractall(dst, filter="data")
    print("ready:", dst)


if __name__ == "__main__":
    want = sys.argv[1:] or ["all"]
    for n in (NAMES if "all" in want else want):
        fetch(n)
