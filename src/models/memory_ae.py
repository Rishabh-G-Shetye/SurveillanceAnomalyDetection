"""
Memory-Augmented Autoencoder (MemAE) for Video Anomaly Detection (Gong et al., ICCV 2019).

Standard autoencoders sometimes generalize "too well" and reconstruct anomalies with low error.
MemAE addresses this by routing the latent representation through a memory matrix
containing learned prototype patterns of normal video. The latent representation is
reconstructed as a sparse linear combination of normal memory items. Because anomalous
patterns cannot be represented by the normal memory prototypes, they fail to reconstruct.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseAnomalyModel


class MemoryModule(nn.Module):
    """
    Learned memory prototype bank with hard-shrinkage addressing.
    Forces sparse attention weights so latent codes only use a few normal prototypes.
    """

    def __init__(self, num_slots: int = 100, slot_dim: int = 512, shrink_thresh: float = 0.0025):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.shrink_thresh = shrink_thresh
        # Memory matrix: (num_slots, slot_dim)
        self.memory = nn.Parameter(torch.empty(num_slots, slot_dim))
        nn.init.kaiming_uniform_(self.memory, a=math.sqrt(5))

    def forward(self, z: torch.Tensor):
        # Compute cosine similarity between latent vectors and memory items
        z_norm = F.normalize(z, dim=1)
        mem_norm = F.normalize(self.memory, dim=1)
        sim = z_norm @ mem_norm.t()  # shape: (N, num_slots)
        attn = F.softmax(sim, dim=1)

        # Stability fix: replace original paper's formula with stable torch.where
        # The original formula divided by |attn - thresh| + 1e-12, which caused gradient explosion
        # when attention values were near the threshold.
        attn = torch.where(attn > self.shrink_thresh, attn, torch.zeros_like(attn))
        sum_attn = attn.sum(dim=1, keepdim=True)
        # Fallback to standard softmax if all slots are pruned below threshold
        attn = torch.where(sum_attn > 1e-8, attn / (sum_attn + 1e-12), F.softmax(sim, dim=1))

        # Reconstruct latent code using memory prototypes
        z_hat = attn @ self.memory  # shape: (N, slot_dim)
        return z_hat, attn


class MemAE(BaseAnomalyModel):
    def __init__(
        self,
        in_channels: int = 1,
        latent_dim: int = 64,
        num_slots: int = 100,
        feat_size: int = 16,
        shrink_thresh: float = 0.0025,
        entropy_weight: float = 0.0002,
        score_agg: str = "mean",
    ):
        super().__init__()
        assert score_agg in ("mean", "max")
        self.score_agg = score_agg
        self.entropy_weight = entropy_weight
        self.latent_dim = latent_dim
        self.feat_size = feat_size
        self.shrink_thresh = shrink_thresh

        # 2D Encoder: downsample 128x128 -> 16x16
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, latent_dim, 4, 2, 1), nn.ReLU(),
        )
        self.memory = MemoryModule(
            num_slots=num_slots,
            slot_dim=latent_dim * feat_size * feat_size,
            shrink_thresh=shrink_thresh,
        )
        # 2D Decoder: upsample 16x16 -> 128x128
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, in_channels, 4, 2, 1), nn.Tanh(),
        )

    def forward_frame(self, x: torch.Tensor):
        B, C, H, W = x.shape
        z = self.encoder(x)
        z_flat = z.view(B, -1)
        z_hat_flat, attn = self.memory(z_flat)
        z_hat = z_hat_flat.view(B, self.latent_dim, self.feat_size, self.feat_size)
        recon = self.decoder(z_hat)
        return recon, attn

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = clip.shape
        x_flat = clip.view(B * T, C, H, W)
        recon, _ = self.forward_frame(x_flat)
        return recon.view(B, T, C, H, W)

    def compute_loss(self, clip: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = clip.shape
        x_flat = clip.view(B * T, C, H, W)
        recon, attn = self.forward_frame(x_flat)
        # Reconstruction MSE
        recon_loss = F.mse_loss(recon, x_flat)
        # Memory entropy loss: encourages sparse memory usage
        entropy_loss = (-attn * torch.log(attn + 1e-12)).sum(dim=-1).mean()
        return recon_loss + self.entropy_weight * entropy_loss

    def per_frame_anomaly_score(self, clip: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            recon = self.forward(clip)
            return ((recon - clip) ** 2).mean(dim=(2, 3, 4))
