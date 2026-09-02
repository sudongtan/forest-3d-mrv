"""Unit tests for src/activity/model.py and src/activity/evaluate.py -- architecture/math checks
(shapes, metric arithmetic against known values), not real-data validation. Real-data validation
of the full pipeline is src/activity/train.py, run manually (see its own docstring) since it needs
the real downloaded MammAlps demo clip and is too slow for the regular pytest suite.
"""
import pytest
import torch

from src.activity.evaluate import evaluate
from src.activity.model import TemporalActivityClassifier


def test_model_output_shape():
    model = TemporalActivityClassifier(num_classes=3, feature_dim=576)
    x = torch.randn(4, 16, 576)  # (batch, window_size, feature_dim)
    logits = model(x)
    assert logits.shape == (4, 3)


def test_model_handles_different_window_sizes():
    model = TemporalActivityClassifier(num_classes=2, feature_dim=576)
    for window_size in (8, 16, 32):
        x = torch.randn(2, window_size, 576)
        assert model(x).shape == (2, 2)


def test_evaluate_perfect_predictions_give_accuracy_one():
    class PerfectModel(torch.nn.Module):
        def forward(self, x):
            # x's first feature dim value directly encodes the intended class for this test
            return torch.nn.functional.one_hot(x[:, 0, 0].long(), num_classes=3).float()

    model = PerfectModel()
    y = torch.tensor([0, 1, 2, 0])
    x = torch.zeros(4, 1, 576)
    x[:, 0, 0] = y.float()

    result = evaluate(model, x, y, class_names=["a", "b", "c"])
    assert result.accuracy == 1.0
    assert all(f1 == 1.0 for f1 in result.per_class_f1.values())
    assert result.confusion_matrix.trace() == 4  # every prediction on the diagonal


def test_evaluate_confusion_matrix_known_case():
    class FixedModel(torch.nn.Module):
        def forward(self, x):
            # always predicts class 0 regardless of input
            batch = x.shape[0]
            out = torch.zeros(batch, 2)
            out[:, 0] = 1.0
            return out

    model = FixedModel()
    y = torch.tensor([0, 0, 1, 1])
    x = torch.zeros(4, 1, 576)

    result = evaluate(model, x, y, class_names=["a", "b"])
    assert result.accuracy == 0.5
    assert result.confusion_matrix.tolist() == [[2, 0], [2, 0]]
    assert result.per_class_f1["a"] == pytest.approx(2 / 3)  # tp=2,fp=2,fn=0 -> P=0.5,R=1,F1=2/3
    assert result.per_class_f1["b"] == 0.0  # tp=0,fp=0,fn=2 -> F1=0
