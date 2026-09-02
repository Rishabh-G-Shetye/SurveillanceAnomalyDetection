"""
Baseline Convolutional Autoencoder (ConvAE) for Video Anomaly Detection.

Compresses each frame down to a 16x16 spatial bottleneck and tries to reconstruct it.
Normal pedestrians are reconstructed cleanly, while unseen anomalous objects
(bicycles, skaters, carts) have high reconstruction error.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseAnomalyModel


class ConvAE(BaseAnomalyModel):
    def __init__(
        self,
        in_channels: int = 1,
        latent_channels: int = 64,
        score_agg: str = "mean",
        use_skip: bool = False,
    ):
        super().__init__()
        assert score_agg in ("mean", "max")
        self.score_agg = score_agg
        # Note: keep use_skip=False by default! Skip connections pass low-level edges
        # straight to the decoder, letting anomalous bikes/carts reconstruct too well.
        self.use_skip = use_skip

        # Encoder: 128x128 -> 64x64 -> 32x32 -> 16x16
        self.enc1 = nn.Sequential(nn.Conv2d(in_channels, 32, 4, 2, 1), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU())
        self.enc3 = nn.Sequential(nn.Conv2d(64, latent_channels, 4, 2, 1), nn.ReLU())

        # Decoder: 16x16 -> 32x32 -> 64x64 -> 128x128
        dec2_in = 64 * 2 if use_skip else 64
        dec1_in = 32 * 2 if use_skip else 32
        self.dec3 = nn.Sequential(nn.ConvTranspose2d(latent_channels, 64, 4, 2, 1), nn.ReLU())
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(dec2_in, 32, 4, 2, 1), nn.ReLU())
        self.dec1 = nn.Sequential(nn.ConvTranspose2d(dec1_in, in_channels, 4, 2, 1), nn.Tanh())

    def _forward_frame(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        d3 = self.dec3(e3)
        d2 = self.dec2(torch.cat([d3, e2], dim=1) if self.use_skip else d3)
        d1 = self.dec1(torch.cat([d2, e1], dim=1) if self.use_skip else d2)
        return d1

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        # Reshape (B, T, C, H, W) to (B*T, C, H, W) to pass through 2D CNN layers
        B, T, C, H, W = clip.shape
        recon = self._forward_frame(clip.view(B * T, C, H, W))
        return recon.view(B, T, C, H, W)

    def compute_loss(self, clip: torch.Tensor) -> torch.Tensor:
        # Mean Squared Error between input and reconstruction
        recon = self.forward(clip)
        return F.mse_loss(recon, clip)

    def per_frame_anomaly_score(self, clip: torch.Tensor) -> torch.Tensor:
        # Compute MSE per individual frame in the clip -> returns (B, T)
        with torch.no_grad():
            recon = self.forward(clip)
            return ((recon - clip) ** 2).mean(dim=(2, 3, 4))
