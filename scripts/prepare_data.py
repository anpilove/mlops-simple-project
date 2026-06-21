"""Download RuSentiTweet and write local CSV splits (3 sentiment classes)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from datasets import load_dataset

from src.common import SENTIMENT_TO_LABEL

HF_DATASET = "psytechlab/RuSentiTweet"
KEEP_LABELS = {"negative", "neutral", "positive"}


def _to_frame(split_df: pd.DataFrame) -> pd.DataFrame:
    df = split_df.copy()
    df["label_str"] = df["label"].astype(str).str.lower()
    df = df[df["label_str"].isin(KEEP_LABELS)]
    df["label"] = df["label_str"].map(SENTIMENT_TO_LABEL).astype(int)
    df["source"] = "Twitter"
    df["language"] = "ru"
    return df.loc[:, ["text", "label", "source", "language", "id"]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[info] loading {HF_DATASET}")
    ds = load_dataset(HF_DATASET)

    train_df = _to_frame(ds["train"].to_pandas())
    val_df = _to_frame(ds["val"].to_pandas())
    test_df = _to_frame(ds["test"].to_pandas())

    train_df.to_csv(args.out_dir / "train.csv", index=False)
    val_df.to_csv(args.out_dir / "val.csv", index=False)
    test_df.to_csv(args.out_dir / "test.csv", index=False)

    print(f"train: {len(train_df)}  val: {len(val_df)}  test: {len(test_df)}")
    print("label counts (train):", train_df["label"].value_counts().sort_index().to_dict())
    print(f"saved to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
