"""
Transformer Autoencoder (TransformerAE) for Video Anomaly Detection.

2D CNN encoder extracts a feature map per frame; spatial features are projected
into a sequence of temporal tokens with positional encodings; a Transformer Encoder
applies multi-head self-attention across time; 2D CNN decoder reconstructs each frame.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseAnomalyModel


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding so the Transformer knows frame order."""

    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, T, d_model)
        return x + self.pe[:, :x.size(1)]


class TransformerAE(BaseAnomalyModel):
    def __init__(
        self,
        in_channels: int = 1,
        feat_channels: int = 32,
        embed_dim: int = 256,
        nhead: int = 4,
        num_layers: int = 2,
        score_agg: str = "mean",
    ):
        super().__init__()
        assert score_agg in ("mean", "max")
        self.score_agg = score_agg
        self.feat_channels = feat_channels
        self.embed_dim = embed_dim

        # Spatial CNN downsampler: 128x128 -> 16x16
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, feat_channels, 4, stride=2, padding=1), nn.ReLU(),
        )

        # Token projection: flatten 16x16 features and project to embed_dim (256)
        # We upgraded embed_dim from 128 to 256 so spatial details aren't lost in the bottleneck
        self.feat_dim = feat_channels * 16 * 16
        self.proj_in = nn.Linear(self.feat_dim, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=512,
            batch_first=True, dropout=0.1,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.proj_out = nn.Linear(embed_dim, self.feat_dim)

        # Spatial CNN upsampler: 16x16 -> 128x128
        self.spatial_decoder = nn.Sequential(
            nn.ConvTranspose2d(feat_channels, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, in_channels, 4, stride=2, padding=1), nn.Tanh(),
        )

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = clip.shape

        # Step 1: encode each frame spatially
        x_flat = clip.view(B * T, C, H, W)
        feats = self.spatial_encoder(x_flat)
        feats = feats.view(B, T, self.feat_dim)

        # Step 2: project to transformer tokens, add pos encoding, run self-attention
        tokens = self.proj_in(feats)
        tokens = self.pos_encoder(tokens)
        trans_out = self.transformer(tokens)

        # Step 3: project back to spatial feature dimensions and decode
        recon_feats = self.proj_out(trans_out).view(B * T, self.feat_channels, 16, 16)
        recon = self.spatial_decoder(recon_feats).view(B, T, C, H, W)
        return recon

    def compute_loss(self, clip: torch.Tensor) -> torch.Tensor:
        recon = self.forward(clip)
        return F.mse_loss(recon, clip)

    def per_frame_anomaly_score(self, clip: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            recon = self.forward(clip)
            return ((recon - clip) ** 2).mean(dim=(2, 3, 4))
