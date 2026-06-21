"""Shared constants for train / serving / UI."""

LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}
LABEL_RU = {0: "негатив", 1: "нейтрал", 2: "позитив"}
LABEL_TO_INT = {v: k for k, v in LABEL_MAP.items()}

SENTIMENT_TO_LABEL = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}

CLEARML_PROJECT = "mlops-simple-project"
DATASET_NAME = "rusentitweet-3class"
DATASET_PROJECT = CLEARML_PROJECT
MODEL_NAME = "tfidf-logreg-sentiment"
SERVING_ENDPOINT = "sentiment"


def preprocess_text(text: str) -> str:
    if not text:
        return ""
    return " ".join(str(text).strip().lower().split())
