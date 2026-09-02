"""Per-class accuracy/F1 + confusion matrix for `activity/model.py`'s predictions, logged to
MLflow per CLAUDE.md's guardrail ("log real experiments to MLflow ... so later changes to the
[...] pipeline are comparable over time").
"""
from dataclasses import dataclass

import mlflow
import numpy as np
import torch

from src.activity.model import TemporalActivityClassifier


@dataclass(frozen=True)
class EvalResult:
    accuracy: float
    per_class_f1: dict[str, float]
    confusion_matrix: np.ndarray  # rows = true class, cols = predicted class, in `class_names` order
    class_names: list[str]


@torch.no_grad()
def evaluate(
    model: TemporalActivityClassifier, X: torch.Tensor, y: torch.Tensor, class_names: list[str],
) -> EvalResult:
    model.eval()
    logits = model(X)
    preds = logits.argmax(dim=1)
    y_np, preds_np = y.numpy(), preds.numpy()
    num_classes = len(class_names)

    confusion = np.zeros((num_classes, num_classes), dtype=int)
    for true_label, pred_label in zip(y_np, preds_np):
        confusion[true_label, pred_label] += 1

    accuracy = float((preds_np == y_np).mean())

    per_class_f1 = {}
    for c in range(num_classes):
        tp = confusion[c, c]
        fp = confusion[:, c].sum() - tp
        fn = confusion[c, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class_f1[class_names[c]] = f1

    return EvalResult(accuracy=accuracy, per_class_f1=per_class_f1, confusion_matrix=confusion,
                       class_names=class_names)


def log_to_mlflow(result: EvalResult, prefix: str = "") -> None:
    mlflow.log_metric(f"{prefix}accuracy", result.accuracy)
    for class_name, f1 in result.per_class_f1.items():
        mlflow.log_metric(f"{prefix}f1_{class_name}", f1)
