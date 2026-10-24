"""Download IMDB through Hugging Face and save a local, raw JSONL copy.

Raw data is intentionally ignored by git. Running preprocess.py directly also
invokes this loader when the processed artifact is missing.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset

SEED = 42


def load_imdb(output_dir: str | Path = "data/raw", limit: int | None = None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset("imdb")
    if limit:
        dataset["train"] = dataset["train"].shuffle(seed=SEED).select(range(min(limit, len(dataset["train"]))))
        dataset["test"] = dataset["test"].shuffle(seed=SEED).select(range(min(limit, len(dataset["test"]))))
    for split in dataset:
        dataset[split].to_json(output_dir / f"{split}.jsonl")
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/raw")
    parser.add_argument("--limit", type=int, help="Limit each official split for fast experiments")
    args = parser.parse_args()
    load_imdb(args.output_dir, args.limit)
