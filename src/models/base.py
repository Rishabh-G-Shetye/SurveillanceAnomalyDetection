"""
Base interface for all anomaly detection models in this project.

Shared contract so our training loop, evaluators, and visualizers work with
any model (ConvAE, ConvLSTM, Transformer, FramePrediction, MemAE) without rewriting code.
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class BaseAnomalyModel(nn.Module, ABC):
    """Abstract base class that all 5 anomaly detection models inherit from."""

    @abstractmethod
    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        # clip: (B, T, C, H, W)
        # returns reconstructed clip (B, T, C, H, W) or predicted frame (B, C, H, W)
        raise NotImplementedError

    @abstractmethod
    def compute_loss(self, clip: torch.Tensor) -> torch.Tensor:
        # Unsupervised loss on normal training video (e.g. MSE, entropy penalty)
        raise NotImplementedError

    @abstractmethod
    def per_frame_anomaly_score(self, clip: torch.Tensor) -> torch.Tensor:
        # Returns frame-level reconstruction/prediction errors.
        # (B, T) for autoencoders, (B,) for frame prediction.
        # Higher score = higher error = more likely an anomaly.
        raise NotImplementedError

    def anomaly_score(self, clip: torch.Tensor) -> torch.Tensor:
        # Summarize frame scores into a single clip score
        scores = self.per_frame_anomaly_score(clip)
        agg = getattr(self, "score_agg", "mean")
        if scores.ndim == 2:
            return scores.max(dim=1).values if agg == "max" else scores.mean(dim=1)
        return scores
