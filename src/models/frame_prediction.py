"""
Future Frame Prediction Network for Video Anomaly Detection.

Instead of reconstructing past frames, this model observes frames 0..T-2
and predicts the future frame T-1. Normal pedestrian motion follows predictable
trajectories, so future frames can be accurately predicted. Anomalies (fast cyclists,
skaters, unauthorized vehicles) have unpredictable velocity and motion dynamics,
causing high prediction errors on the target frame.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseAnomalyModel


class _PredictorLSTMCell(nn.Module):
    """ConvLSTM cell used for predicting future frame state."""

    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.conv = nn.Conv2d(
            in_channels + hidden_channels, 4 * hidden_channels,
            kernel_size, padding=kernel_size // 2,
        )

    def forward(self, x, h, c):
        gates = self.conv(torch.cat([x, h], dim=1))
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size: int, spatial_size: tuple, device):
        H, W = spatial_size
        h = torch.zeros(batch_size, self.hidden_channels, H, W, device=device)
        c = torch.zeros(batch_size, self.hidden_channels, H, W, device=device)
        return h, c


class FramePredictionNet(BaseAnomalyModel):
    def __init__(
        self,
        in_channels: int = 1,
        feat_channels: int = 32,
        hidden_channels: int = 64,
        score_agg: str = "mean",
    ):
        super().__init__()
        assert score_agg in ("mean", "max")
        self.score_agg = score_agg

        # Spatial encoder: 128x128 -> 32x32
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, feat_channels, 4, stride=2, padding=1), nn.ReLU(),
        )
        self.lstm = _PredictorLSTMCell(feat_channels, hidden_channels)

        # Spatial decoder: 32x32 -> 128x128 predicts next frame
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_channels, 16, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, in_channels, 4, stride=2, padding=1), nn.Tanh(),
        )

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        # Observe frames 0..T-2, predict frame T-1
        B, T, C, H, W = clip.shape
        device = clip.device
        past_frames = clip[:, :-1]  # shape: (B, T-1, C, H, W)

        # Encode past frames
        x_flat = past_frames.contiguous().view(B * (T - 1), C, H, W)
        feats = self.encoder(x_flat)
        _, fC, fH, fW = feats.shape
        feats = feats.view(B, T - 1, fC, fH, fW)

        # Recurrent pass through past frames
        h, c = self.lstm.init_hidden(B, (fH, fW), device)
        for t in range(T - 1):
            h, c = self.lstm(feats[:, t], h, c)

        # Decode final hidden state into predicted future frame
        pred_frame = self.decoder(h)  # shape: (B, C, H, W)
        return pred_frame

    def compute_loss(self, clip: torch.Tensor) -> torch.Tensor:
        # Ground truth target is the actual last frame (clip[:, -1])
        target_frame = clip[:, -1]
        pred_frame = self.forward(clip)
        return F.mse_loss(pred_frame, target_frame)

    def per_frame_anomaly_score(self, clip: torch.Tensor) -> torch.Tensor:
        # Key bug fix: The prediction error belongs ONLY to the target frame (T-1)!
        # Assigning error only to the target frame prevents false alarms on normal past frames.
        with torch.no_grad():
            target_frame = clip[:, -1]
            pred_frame = self.forward(clip)
            # MSE between predicted frame and actual frame -> returns (B,)
            return ((pred_frame - target_frame) ** 2).mean(dim=(1, 2, 3))
