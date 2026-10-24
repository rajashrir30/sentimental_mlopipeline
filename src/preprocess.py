"""Create a shared, deterministic cleaned IMDB train/validation/test artifact."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from datasets import load_dataset
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from text_utils import clean_text
SEED = 42
DEFAULT_OUTPUT = Path("data/processed/imdb.json")


def prepare_dataset(output_path: str | Path = DEFAULT_OUTPUT, limit: int | None = None) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw = load_dataset("imdb")
    records = []
    for split in ("train", "test"):
        rows = raw[split]
        if limit:
            rows = rows.shuffle(seed=SEED).select(range(min(limit, len(rows))))
        records.extend({"text": clean_text(row["text"]), "label": int(row["label"])} for row in rows)

    # Avoid identical cleaned reviews crossing split boundaries.
    unique_records = {}
    for record in records:
        unique_records.setdefault(record["text"], record)
    records = list(unique_records.values())

    texts = [r["text"] for r in records]
    labels = [r["label"] for r in records]
    train_x, holdout_x, train_y, holdout_y = train_test_split(
        texts, labels, test_size=0.2, random_state=SEED, stratify=labels
    )
    val_x, test_x, val_y, test_y = train_test_split(
        holdout_x, holdout_y, test_size=0.5, random_state=SEED, stratify=holdout_y
    )
    payload = {
        "seed": SEED,
        "train": {"text": train_x, "label": train_y},
        "validation": {"text": val_x, "label": val_y},
        "test": {"text": test_x, "label": test_y},
    }
    output_path.write_text(json.dumps(payload), encoding="utf-8")
    return output_path


def load_processed(path: str | Path = DEFAULT_OUTPUT) -> dict:
    path = Path(path)
    if not path.exists():
        prepare_dataset(path)
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    print(prepare_dataset(args.output, args.limit))

