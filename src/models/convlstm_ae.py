"""
Spatiotemporal Convolutional LSTM Autoencoder (ConvLSTM-AE).

2D CNN encoder extracts appearance features from each frame,
ConvLSTM cell carries hidden state forward across time to model motion/velocity,
and 2D CNN decoder reconstructs frames from the spatiotemporal states.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseAnomalyModel


class ConvLSTMCell(nn.Module):
    """Standard 2D Convolutional LSTM cell."""

    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2
        # Fuses input and hidden state into 4 gates: input, forget, output, cell-candidate
        self.conv = nn.Conv2d(
            in_channels + hidden_channels, 4 * hidden_channels, kernel_size, padding=padding
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


class ConvLSTMAE(BaseAnomalyModel):
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

        # Spatial encoder: 128x128 -> 64x64 -> 32x32
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, feat_channels, 4, stride=2, padding=1), nn.ReLU(),
        )
        # Recurrent cell models temporal transition between successive frames
        self.lstm_cell = ConvLSTMCell(feat_channels, hidden_channels)

        # Spatial decoder: 32x32 -> 64x64 -> 128x128
        self.frame_decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_channels, 16, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, in_channels, 4, stride=2, padding=1), nn.Tanh(),
        )

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = clip.shape
        device = clip.device

        # Step 1: encode all frames spatially
        x_flat = clip.view(B * T, C, H, W)
        feats = self.frame_encoder(x_flat)
        _, fC, fH, fW = feats.shape
        feats = feats.view(B, T, fC, fH, fW)

        # Step 2: roll forward through ConvLSTM cell
        h, c = self.lstm_cell.init_hidden(B, (fH, fW), device)
        h_states = []
        for t in range(T):
            h, c = self.lstm_cell(feats[:, t], h, c)
            h_states.append(h)

        # Step 3: decode all hidden states back to reconstructed frames
        h_stack = torch.stack(h_states, dim=1).view(B * T, -1, fH, fW)
        recon = self.frame_decoder(h_stack).view(B, T, C, H, W)
        return recon

    def compute_loss(self, clip: torch.Tensor) -> torch.Tensor:
        recon = self.forward(clip)
        return F.mse_loss(recon, clip)

    def per_frame_anomaly_score(self, clip: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            recon = self.forward(clip)
            return ((recon - clip) ** 2).mean(dim=(2, 3, 4))
