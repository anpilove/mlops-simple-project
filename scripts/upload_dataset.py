"""Upload local CSV splits to ClearML Dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from clearml import Dataset

from src.common import DATASET_NAME, DATASET_PROJECT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--version", default="1.0")
    args = parser.parse_args()

    train_path = args.data_dir / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError(
            f"{train_path} not found. Run: python -m scripts.prepare_data"
        )

    dataset = Dataset.create(
        dataset_name=DATASET_NAME,
        dataset_project=DATASET_PROJECT,
        dataset_version=args.version,
        description="RuSentiTweet sentiment (text + label)",
    )
    dataset.add_files(path=str(args.data_dir))
    dataset.upload(show_progress=True)
    dataset.finalize()

    print(f"dataset_id={dataset.id}")
    print(f"version={args.version}")


if __name__ == "__main__":
    main()
