"""A small temporal classifier over per-frame appearance features (`features.py`) -- a 1D-CNN,
per CLAUDE.md's Tech Stack ("a lightweight temporal classifier (small transformer or 1D-CNN over
per-clip features)"). Kept deliberately small: the real data this project currently has to train
on is one demo clip's worth of windows (see `datasets.py`'s module docstring for why), and a
large model would either trivially memorize or fail to train meaningfully on that little data
either way -- this is sized for the architecture-sanity-check role CLAUDE.md's plan assigns this
step, not for a claim about MammAlps-scale performance.
"""
import torch
from torch import nn


class TemporalActivityClassifier(nn.Module):
    def __init__(self, num_classes: int, feature_dim: int = 576, hidden_dim: int = 64):
        super().__init__()
        self.conv1 = nn.Conv1d(feature_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, window_size, feature_dim) -> logits: (batch, num_classes)."""
        x = x.transpose(1, 2)  # (batch, feature_dim, window_size) -- Conv1d wants channels first
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        return self.classifier(x)
