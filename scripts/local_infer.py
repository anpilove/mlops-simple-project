"""Local inference smoke test (without ClearML Serving)."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from src.common import LABEL_MAP, LABEL_RU, preprocess_text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("artifacts/exp2.joblib"))
    parser.add_argument("--text", required=True)
    args = parser.parse_args()

    pipeline = joblib.load(args.model)
    pred = int(pipeline.predict([preprocess_text(args.text)])[0])
    print(f"{LABEL_RU[pred]} ({LABEL_MAP[pred]})")


if __name__ == "__main__":
    main()
