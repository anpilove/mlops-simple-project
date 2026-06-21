"""Train TF-IDF + LogisticRegression via ClearML (local or agent queue)."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import subprocess

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from clearml import Dataset, Task
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline

from src.common import CLEARML_PROJECT, LABEL_MAP, MODEL_NAME, preprocess_text


def load_frames(data_dir: Path, dataset_id: str | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    if dataset_id:
        dataset = Dataset.get(dataset_id=dataset_id)
        local = Path(dataset.get_local_copy())
        train_df = pd.read_csv(local / "train.csv")
        val_path = local / "val.csv"
        val_df = pd.read_csv(val_path) if val_path.exists() else train_df.sample(frac=0.15, random_state=0)
        return train_df, val_df

    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "val.csv")
    return train_df, val_df


def train_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    max_features: int,
    c: float,
) -> tuple[Pipeline, dict]:
    x_train = train_df["text"].map(preprocess_text)
    y_train = train_df["label"].astype(int)
    x_val = val_df["text"].map(preprocess_text)
    y_val = val_df["label"].astype(int)

    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))),
            (
                "clf",
                LogisticRegression(
                    C=c,
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
    pipeline.fit(x_train, y_train)
    preds = pipeline.predict(x_val)

    metrics = {
        "accuracy": float(accuracy_score(y_val, preds)),
        "f1_macro": float(f1_score(y_val, preds, average="macro")),
    }
    report = classification_report(
        y_val, preds, target_names=[LABEL_MAP[i] for i in sorted(LABEL_MAP)], output_dict=True
    )
    cm = confusion_matrix(y_val, preds, labels=sorted(LABEL_MAP.keys()))
    return pipeline, {"metrics": metrics, "report": report, "confusion_matrix": cm}


def confusion_matrix_figure(cm, labels: list[str]):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")
    fig.tight_layout()
    return fig


def attach_project_code(task: Task) -> None:
    """Point ClearML agent at the local git checkout on this machine."""
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if head.returncode != 0:
        raise RuntimeError(
            "ClearML agent needs at least one git commit in the project. "
            "Create a WIP commit first (final commit can wait until the end)."
        )
    task.set_repo(repo=str(PROJECT_ROOT))
    diff = subprocess.run(
        ["git", "diff", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    task.set_script(
        diff=diff or None,
        working_dir=".",
        entry_point="scripts/train.py",
        binary="python3",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--dataset-id", default=None)
    parser.add_argument("--max-features", type=int, default=5000)
    parser.add_argument("--C", type=float, default=0.1, dest="c")
    parser.add_argument("--task-name", default="train-tfidf-logreg")
    parser.add_argument("--queue", default=None, help="ClearML queue, e.g. students")
    parser.add_argument("--local", action="store_true", help="Skip ClearML Task (smoke test)")
    parser.add_argument("--output", type=Path, default=Path("artifacts/model.joblib"))
    args = parser.parse_args()

    if args.local:
        task = None
    else:
        if args.queue and Task.running_locally():
            Task.add_requirements(str(PROJECT_ROOT / "requirements.txt"))
        task = Task.init(
            project_name=CLEARML_PROJECT,
            task_name=args.task_name,
            reuse_last_task_id=False if args.queue and Task.running_locally() else True,
        )
        if args.queue and task.running_locally():
            task.set_packages(str(PROJECT_ROOT / "requirements.txt"))
            attach_project_code(task)
            task.flush()
            task.execute_remotely(queue_name=args.queue)
            return

    params = {
        "max_features": args.max_features,
        "C": args.c,
        "dataset_id": args.dataset_id,
    }
    if task:
        task.connect(params)

    train_df, val_df = load_frames(args.data_dir, args.dataset_id)
    pipeline, result = train_model(train_df, val_df, args.max_features, args.c)

    if task:
        logger = task.get_logger()
        logger.report_scalar("accuracy", series="validation", value=result["metrics"]["accuracy"], iteration=0)
        logger.report_scalar("f1", series="validation", value=result["metrics"]["f1_macro"], iteration=0)
        logger.report_confusion_matrix(
            "confusion_matrix",
            "validation",
            matrix=result["confusion_matrix"],
            iteration=0,
            xlabels=[LABEL_MAP[i] for i in sorted(LABEL_MAP)],
            ylabels=[LABEL_MAP[i] for i in sorted(LABEL_MAP)],
        )
        labels = [LABEL_MAP[i] for i in sorted(LABEL_MAP)]
        fig = confusion_matrix_figure(result["confusion_matrix"], labels)
        logger.report_matplotlib_figure("confusion_matrix_image", "validation", fig, iteration=0)
        plt.close(fig)
        for label, stats in result["report"].items():
            if label in LABEL_MAP.values():
                logger.report_scalar("f1_per_class", series=label, value=float(stats["f1-score"]), iteration=0)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, args.output)
        task.upload_artifact("model", artifact_object=pipeline)
        task.update_output_model(
            model_path=str(args.output),
            model_name=MODEL_NAME,
            name="model",
            auto_delete_file=False,
            tags=["sentiment", "rusentitweet", "tfidf-logreg"],
            comment=(
                f"val accuracy={result['metrics']['accuracy']:.4f}, "
                f"f1_macro={result['metrics']['f1_macro']:.4f}"
            ),
        )
        print(f"task_id={task.id}")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, args.output)
        print(f"saved {args.output}")

    print("metrics:", result["metrics"])


if __name__ == "__main__":
    main()
