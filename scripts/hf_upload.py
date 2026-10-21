"""Upload a staged dataset folder to the Hugging Face Hub (maintainer use).

    hf auth login            # once
    python scripts/hf_upload.py <staged-folder> <name> [--user USER] [--private]

<name> is resplan-fp or cubicasa5k-skv4; the folder must already contain the
dataset card as README.md (see data/cards/).
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder"); ap.add_argument("name")
    ap.add_argument("--user", default="Tristan12138")
    ap.add_argument("--private", action="store_true")
    a = ap.parse_args()
    from huggingface_hub import HfApi
    api = HfApi()
    repo = f"{a.user}/{a.name}"
    api.create_repo(repo, repo_type="dataset", exist_ok=True, private=a.private)
    api.upload_folder(folder_path=str(Path(a.folder)), repo_id=repo, repo_type="dataset",
                      commit_message=f"release {a.name}")
    print("uploaded ->", f"https://huggingface.co/datasets/{repo}")


if __name__ == "__main__":
    main()
