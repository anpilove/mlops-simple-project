"""ClearML Serving preprocessing for sklearn pipeline (self-contained, no src imports)."""

from __future__ import annotations

from typing import Any

import numpy as np

LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}


def preprocess_text(text: str) -> str:
    if not text:
        return ""
    return " ".join(str(text).strip().lower().split())


class Preprocess:
    def __init__(self) -> None:
        pass

    def preprocess(
        self,
        body: dict,
        state: dict,
        collect_custom_statistics_fn=None,
    ) -> Any:
        text = body.get("text", "")
        return [preprocess_text(text)]

    def postprocess(
        self,
        data: Any,
        state: dict,
        collect_custom_statistics_fn=None,
    ) -> dict:
        label_id = int(data[0]) if hasattr(data, "__len__") else int(data)
        return {
            "label_id": label_id,
            "label": LABEL_MAP.get(label_id, str(label_id)),
            "y": data.tolist() if isinstance(data, np.ndarray) else data,
        }
