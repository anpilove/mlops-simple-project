"""Publish the best experiment model to ClearML Model Registry."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
from clearml import Model, Task

from src.common import CLEARML_PROJECT, MODEL_NAME


def task_f1(task: Task) -> float:
    metrics = task.get_last_scalar_metrics() or {}
    try:
        return float(metrics["f1"]["validation"]["last"])
    except (KeyError, TypeError, ValueError):
        return -1.0


def pick_best_task(candidates: list[str]) -> Task:
    best: Task | None = None
    best_f1 = -1.0
    for name in candidates:
        tasks = sorted(
            Task.get_tasks(project_name=CLEARML_PROJECT, task_name=name),
            key=lambda t: t.data.created,
            reverse=True,
        )
        for task in tasks:
            if task.get_status() not in ("completed", "published"):
                continue
            score = task_f1(task)
            if score > best_f1:
                best_f1 = score
                best = task
    if best is None:
        raise RuntimeError("No completed training tasks found (exp1 / exp2).")
    return best


def ensure_output_model(task: Task) -> str:
    output_ids = task.output_models_id or {}
    if output_ids:
        model_id = next(iter(output_ids.values()))
        return model_id

    if "model" not in task.artifacts:
        raise RuntimeError(f"Task {task.id} has no output model and no 'model' artifact.")

    pipeline = task.artifacts["model"].get()
    out = PROJECT_ROOT / "artifacts" / "model.joblib"
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, out)
    task.update_output_model(
        model_path=str(out),
        model_name=MODEL_NAME,
        name="model",
        auto_delete_file=False,
        tags=["sentiment", "rusentitweet", "tfidf-logreg"],
        comment=f"val f1_macro={task_f1(task):.4f}",
    )
    task.flush()
    model_id = next(iter(task.output_models_id.values()))
    return model_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", help="Explicit ClearML task id to publish")
    parser.add_argument(
        "--candidates",
        nargs="+",
        default=["exp1", "exp2"],
        help="Task names to compare when --task-id is not set",
    )
    args = parser.parse_args()

    task = Task.get_task(args.task_id) if args.task_id else pick_best_task(args.candidates)
    print(f"selected task: {task.name} ({task.id}), f1={task_f1(task):.4f}")

    model_id = ensure_output_model(task)
    model = Model(model_id)
    if not model.published:
        model.publish()
    model.tags = sorted(set(model.tags or []) | {"sentiment", "rusentitweet", "tfidf-logreg", "best"})
    print(f"published model: {model.name} ({model.id})")


if __name__ == "__main__":
    main()
