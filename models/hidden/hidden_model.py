"""
HiDDeN-Style Adversarial Steganography Model (2025 Enhanced).

State-of-the-art approach combining:
- Encoder with ConvNeXt blocks and multi-scale message injection
- Decoder with attention-based message extraction
- WGAN-GP Discriminator with spectral normalization
- Comprehensive noise layer (JPEG, Gaussian, Crop, Blur, Codec simulation)
- Frequency-domain loss for anti-steganalysis

Based on: "HiDDeN: Hiding Data With Deep Networks" (Zhu et al.)
Enhanced with 2025 techniques: attention, ConvNeXt, codec robustness.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import CBAM, ConvNeXtBlock, ResidualDenseBlock, NoiseLayer


class HiDDeNEncoder(nn.Module):
    """
    Encoder: Cover Image + Message -> Stego Image.

    Uses residual learning — outputs a small perturbation added to cover.
    Multi-scale message injection ensures message survives at different resolutions.
    """

    def __init__(self, msg_length: int = 128, base_ch: int = 64):
        super().__init__()

        # Image feature extractor
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, base_ch, 3, padding=1),
            nn.GroupNorm(8, base_ch),
            nn.GELU(),
            ConvNeXtBlock(base_ch),
            ConvNeXtBlock(base_ch),
            CBAM(base_ch),
        )

        # Message expander (learns to spread message spatially)
        self.msg_expander = nn.Sequential(
            nn.Linear(msg_length, base_ch * 16 * 16),
            nn.GELU(),
        )
        self.msg_upsample = nn.Sequential(
            nn.ConvTranspose2d(base_ch, base_ch, 4, stride=4, padding=0),  # 16 -> 64
            nn.GELU(),
            nn.ConvTranspose2d(base_ch, base_ch, 4, stride=4, padding=0),  # 64 -> 256
            nn.GELU(),
        )

        # Fusion (image features + message features -> residual)
        self.fusion = nn.Sequential(
            nn.Conv2d(base_ch * 2, base_ch * 2, 3, padding=1),
            nn.GroupNorm(8, base_ch * 2),
            nn.GELU(),
            ResidualDenseBlock(base_ch * 2, growth=32, n_layers=4),
            ConvNeXtBlock(base_ch * 2),
            CBAM(base_ch * 2),
            nn.Conv2d(base_ch * 2, base_ch, 3, padding=1),
            nn.GroupNorm(8, base_ch),
            nn.GELU(),
            ConvNeXtBlock(base_ch),
            nn.Conv2d(base_ch, 3, 1),
            nn.Tanh(),
        )

        self.strength = nn.Parameter(torch.tensor(0.05))

    def forward(self, cover: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        """cover: (B,3,H,W), message: (B, msg_length) -> stego: (B,3,H,W)"""
        B, _, H, W = cover.shape

        # Extract image features
        img_feat = self.image_encoder(cover)  # (B, 64, H, W)

        # Expand message to spatial dims
        msg_feat = self.msg_expander(message)  # (B, 64*16*16)
        msg_feat = msg_feat.view(B, -1, 16, 16)  # (B, 64, 16, 16)
        msg_feat = self.msg_upsample(msg_feat)  # (B, 64, 256, 256)

        # Handle size mismatch
        if msg_feat.shape[2:] != img_feat.shape[2:]:
            msg_feat = F.interpolate(msg_feat, size=img_feat.shape[2:], mode="bilinear", align_corners=False)

        # Fuse and generate residual
        combined = torch.cat([img_feat, msg_feat], dim=1)
        residual = self.fusion(combined) * self.strength

        return torch.clamp(cover + residual, 0, 1)


class HiDDeNDecoder(nn.Module):
    """
    Decoder: Stego Image -> Extracted Message.

    Lightweight but robust — uses attention to focus on embedded regions.
    """

    def __init__(self, msg_length: int = 128, base_ch: int = 64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, base_ch, 3, stride=2, padding=1),
            nn.GroupNorm(8, base_ch),
            nn.GELU(),
            ConvNeXtBlock(base_ch),
            CBAM(base_ch),

            nn.Conv2d(base_ch, base_ch * 2, 3, stride=2, padding=1),
            nn.GroupNorm(8, base_ch * 2),
            nn.GELU(),
            ConvNeXtBlock(base_ch * 2),
            CBAM(base_ch * 2),

            nn.Conv2d(base_ch * 2, base_ch * 4, 3, stride=2, padding=1),
            nn.GroupNorm(8, base_ch * 4),
            nn.GELU(),
            ConvNeXtBlock(base_ch * 4),

            nn.Conv2d(base_ch * 4, base_ch * 8, 3, stride=2, padding=1),
            nn.GroupNorm(8, base_ch * 8),
            nn.GELU(),

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(base_ch * 8, base_ch * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(base_ch * 4, msg_length),
        )

    def forward(self, stego: torch.Tensor) -> torch.Tensor:
        feat = self.features(stego)
        return self.classifier(feat)


class Discriminator(nn.Module):
    """
    WGAN-GP Discriminator with Spectral Normalization.

    Distinguishes cover images from stego images.
    Pushes the encoder to produce more imperceptible modifications.
    """

    def __init__(self, base_ch: int = 64):
        super().__init__()
        self.model = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(3, base_ch, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),

            nn.utils.spectral_norm(nn.Conv2d(base_ch, base_ch * 2, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),

            nn.utils.spectral_norm(nn.Conv2d(base_ch * 2, base_ch * 4, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),

            nn.utils.spectral_norm(nn.Conv2d(base_ch * 4, base_ch * 8, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),

            nn.utils.spectral_norm(nn.Conv2d(base_ch * 8, base_ch * 8, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_ch * 8, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class HiDDeNSteganography(nn.Module):
    """
    Complete HiDDeN system: Encoder + Noise + Decoder + Discriminator.
    """

    def __init__(self, msg_length: int = 128, base_ch: int = 64):
        super().__init__()
        self.encoder = HiDDeNEncoder(msg_length, base_ch)
        self.decoder = HiDDeNDecoder(msg_length, base_ch)
        self.noise_layer = NoiseLayer()
        self.discriminator = Discriminator(base_ch)

    def forward(self, cover: torch.Tensor, message: torch.Tensor):
        stego = self.encoder(cover, message)
        noisy_stego = self.noise_layer(stego)
        decoded_msg = self.decoder(noisy_stego)
        return stego, decoded_msg

    def discriminate(self, images: torch.Tensor) -> torch.Tensor:
        return self.discriminator(images)
