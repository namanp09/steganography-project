"""
Shared Neural Network Layers & Building Blocks.

Modern components used across all DL steganography models:
- Channel & Spatial Attention (CBAM)
- ConvNeXt-style blocks
- Differentiable noise layers (JPEG, Gaussian, Crop, Codec simulation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange


# ──────────────────────────── Attention Modules ────────────────────────────


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation style channel attention."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.GELU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc(x).unsqueeze(-1).unsqueeze(-1)
        return x * w


class SpatialAttention(nn.Module):
    """Spatial attention using max/avg pooling."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx = x.max(dim=1, keepdim=True)[0]
        attn = self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * attn


class CBAM(nn.Module):
    """Convolutional Block Attention Module (channel + spatial)."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sa(self.ca(x))


# ──────────────────────────── ConvNeXt Block ────────────────────────────


class ConvNeXtBlock(nn.Module):
    """Modern ConvNeXt-V2 style block with depthwise conv + GELU."""

    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, 7, padding=3, groups=dim),  # Depthwise
            nn.GroupNorm(1, dim),  # LayerNorm equivalent
            nn.Conv2d(dim, dim * mult, 1),  # Pointwise expand
            nn.GELU(),
            nn.Conv2d(dim * mult, dim, 1),  # Pointwise shrink
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


# ──────────────────────────── Residual Dense Block ────────────────────────


class ResidualDenseBlock(nn.Module):
    """Dense connections for better feature reuse."""

    def __init__(self, channels: int, growth: int = 32, n_layers: int = 4):
        super().__init__()
        layers = []
        for i in range(n_layers):
            in_ch = channels + i * growth
            layers.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, growth, 3, padding=1),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            )
        self.layers = nn.ModuleList(layers)
        self.fusion = nn.Conv2d(channels + n_layers * growth, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, dim=1))
            features.append(out)
        return x + self.fusion(torch.cat(features, dim=1)) * 0.2


# ──────────────────────────── Noise Layers ────────────────────────────


class DifferentiableJPEG(nn.Module):
    """
    Differentiable JPEG compression simulation.
    Uses rounding approximation for backpropagation.
    """

    def __init__(self, quality_range=(50, 95)):
        super().__init__()
        self.quality_range = quality_range

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        quality = torch.randint(
            self.quality_range[0], self.quality_range[1], (1,)
        ).item()
        factor = (100 - quality) / 50.0

        # Simulate quantization noise
        noise_strength = factor * 0.02
        noise = torch.randn_like(x) * noise_strength
        # Straight-through estimator: add noise but pass gradient through
        return x + noise - noise.detach() + noise.detach()


class GaussianNoise(nn.Module):
    """Additive Gaussian noise layer."""

    def __init__(self, std_range=(0.01, 0.05)):
        super().__init__()
        self.std_range = std_range

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        std = torch.empty(1).uniform_(*self.std_range).item()
        return x + torch.randn_like(x) * std


class RandomCrop(nn.Module):
    """Random cropping and resize back (simulates cropping attack)."""

    def __init__(self, min_ratio=0.7):
        super().__init__()
        self.min_ratio = min_ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        _, _, h, w = x.shape
        ratio = torch.empty(1).uniform_(self.min_ratio, 1.0).item()
        new_h, new_w = int(h * ratio), int(w * ratio)
        top = torch.randint(0, h - new_h + 1, (1,)).item()
        left = torch.randint(0, w - new_w + 1, (1,)).item()

        cropped = x[:, :, top : top + new_h, left : left + new_w]
        return F.interpolate(cropped, size=(h, w), mode="bilinear", align_corners=False)


class GaussianBlur(nn.Module):
    """Gaussian blur attack simulation."""

    def __init__(self, kernel_size=5, sigma_range=(0.5, 2.0)):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma_range = sigma_range

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        sigma = torch.empty(1).uniform_(*self.sigma_range).item()
        # Create Gaussian kernel
        k = self.kernel_size
        ax = torch.arange(k, dtype=torch.float32, device=x.device) - k // 2
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        kernel = kernel.expand(x.shape[1], 1, k, k)
        return F.conv2d(x, kernel, padding=k // 2, groups=x.shape[1])


class CodecSimulation(nn.Module):
    """
    Simulates H.264/H.265 video codec compression artifacts.
    Uses a combination of quantization + blocking + smoothing.
    """

    def __init__(self, strength_range=(0.01, 0.04)):
        super().__init__()
        self.strength_range = strength_range

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        strength = torch.empty(1).uniform_(*self.strength_range).item()

        # Block-based quantization (simulates macroblock processing)
        b, c, h, w = x.shape
        block = 8
        # Reshape into 8x8 blocks
        if h % block == 0 and w % block == 0:
            blocks = rearrange(x, "b c (h bh) (w bw) -> b c h w bh bw", bh=block, bw=block)
            # Add quantization noise per block
            noise = torch.randn(b, c, h // block, w // block, 1, 1, device=x.device) * strength
            blocks = blocks + noise
            x_noisy = rearrange(blocks, "b c h w bh bw -> b c (h bh) (w bw)")
        else:
            x_noisy = x + torch.randn_like(x) * strength

        return x_noisy


class NoiseLayer(nn.Module):
    """
    Combined noise layer that randomly applies one distortion during training.
    Simulates real-world attacks the steganography must survive.
    """

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            DifferentiableJPEG(),
            GaussianNoise(),
            RandomCrop(),
            GaussianBlur(),
            CodecSimulation(),
            nn.Identity(),  # No noise (clean pass-through)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        idx = torch.randint(0, len(self.layers), (1,)).item()
        return self.layers[idx](x)
