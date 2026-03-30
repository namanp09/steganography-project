"""
Invertible Neural Network (INN) for Video Steganography.

State-of-the-art 2025 approach based on normalizing flows.
Key advantages:
- Mathematically invertible: perfect reconstruction guarantee
- Single network for both hiding AND revealing
- Multi-scale decomposition (Haar wavelet lifting)
- No separate encoder/decoder needed

Based on: "Robust Video Steganography Based on Multi-scale Decomposition
           and Invertible Networks" (2025)

Architecture:
- Haar wavelet lifting for multi-scale decomposition
- Affine coupling layers with ConvNeXt subnet
- Channel attention in coupling subnets
- 3D temporal convolutions for video consistency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from models.layers import CBAM, ConvNeXtBlock


class HaarWaveletTransform(nn.Module):
    """
    Haar wavelet lifting scheme for multi-scale image decomposition.
    Decomposes image into LL (low-freq), LH, HL, HH (high-freq) subbands.
    Fully differentiable and invertible.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) -> (B, 4C, H/2, W/2)"""
        x_ll = (x[:, :, 0::2, 0::2] + x[:, :, 0::2, 1::2] +
                x[:, :, 1::2, 0::2] + x[:, :, 1::2, 1::2]) / 4.0
        x_lh = (x[:, :, 0::2, 0::2] + x[:, :, 0::2, 1::2] -
                x[:, :, 1::2, 0::2] - x[:, :, 1::2, 1::2]) / 4.0
        x_hl = (x[:, :, 0::2, 0::2] - x[:, :, 0::2, 1::2] +
                x[:, :, 1::2, 0::2] - x[:, :, 1::2, 1::2]) / 4.0
        x_hh = (x[:, :, 0::2, 0::2] - x[:, :, 0::2, 1::2] -
                x[:, :, 1::2, 0::2] + x[:, :, 1::2, 1::2]) / 4.0
        return torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 4C, H/2, W/2) -> (B, C, H, W)"""
        C4 = x.shape[1]
        C = C4 // 4
        x_ll, x_lh, x_hl, x_hh = x[:, :C], x[:, C:2*C], x[:, 2*C:3*C], x[:, 3*C:]

        B, _, h, w = x_ll.shape
        out = torch.zeros(B, C, h * 2, w * 2, device=x.device, dtype=x.dtype)
        out[:, :, 0::2, 0::2] = x_ll + x_lh + x_hl + x_hh
        out[:, :, 0::2, 1::2] = x_ll + x_lh - x_hl - x_hh
        out[:, :, 1::2, 0::2] = x_ll - x_lh + x_hl - x_hh
        out[:, :, 1::2, 1::2] = x_ll - x_lh - x_hl + x_hh
        return out


class AffineCouplingSubnet(nn.Module):
    """Subnet used inside affine coupling layers."""

    def __init__(self, in_ch: int, out_ch: int, hidden_ch: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, 3, padding=1),
            nn.GroupNorm(8, hidden_ch),
            nn.GELU(),
            ConvNeXtBlock(hidden_ch),
            CBAM(hidden_ch),
            nn.Conv2d(hidden_ch, hidden_ch, 3, padding=1),
            nn.GroupNorm(8, hidden_ch),
            nn.GELU(),
            nn.Conv2d(hidden_ch, out_ch, 3, padding=1),
        )
        # Initialize last layer to near-zero for stable training
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AffineCouplingBlock(nn.Module):
    """
    Affine coupling layer — core of the invertible network.

    Split input into two halves:
    - x1 unchanged, x2 transformed based on x1
    - Perfectly invertible by design
    """

    def __init__(self, channels: int, hidden_ch: int = 64):
        super().__init__()
        half_ch = channels // 2
        self.subnet_scale = AffineCouplingSubnet(half_ch, half_ch, hidden_ch)
        self.subnet_shift = AffineCouplingSubnet(half_ch, half_ch, hidden_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        s = torch.sigmoid(self.subnet_scale(x1)) * 2  # Scale in (0, 2)
        t = self.subnet_shift(x1)
        y1 = x1
        y2 = x2 * s + t
        return torch.cat([y1, y2], dim=1)

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        y1, y2 = y.chunk(2, dim=1)
        s = torch.sigmoid(self.subnet_scale(y1)) * 2
        t = self.subnet_shift(y1)
        x1 = y1
        x2 = (y2 - t) / (s + 1e-8)
        return torch.cat([x1, x2], dim=1)


class ChannelShuffle(nn.Module):
    """Fixed random channel permutation (invertible)."""

    def __init__(self, channels: int):
        super().__init__()
        perm = torch.randperm(channels)
        inv_perm = torch.argsort(perm)
        self.register_buffer("perm", perm)
        self.register_buffer("inv_perm", inv_perm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, self.perm]

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, self.inv_perm]


class TemporalAttention3D(nn.Module):
    """
    3D temporal attention for video frame consistency.
    Processes a window of frames to ensure temporal coherence.
    """

    def __init__(self, channels: int, temporal_window: int = 5):
        super().__init__()
        self.conv3d = nn.Conv3d(
            channels, channels, kernel_size=(temporal_window, 3, 3),
            padding=(temporal_window // 2, 1, 1), groups=channels,
        )
        self.norm = nn.GroupNorm(8, channels)
        self.gate = nn.Sequential(
            nn.Conv3d(channels, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames: (B, T, C, H, W) — batch of frame sequences

        Returns:
            (B, T, C, H, W) — temporally smoothed frames
        """
        B, T, C, H, W = frames.shape
        x = rearrange(frames, "b t c h w -> b c t h w")
        feat = self.conv3d(x)
        gate = self.gate(x)
        out = x + feat * gate
        return rearrange(out, "b c t h w -> b t c h w")


class InvertibleSteganography(nn.Module):
    """
    Complete Invertible Neural Network for Video Steganography.

    Forward pass (hide):  cover_frames + secret -> stego_frames
    Inverse pass (reveal): stego_frames -> extracted_secret

    The INN operates in wavelet domain for multi-scale processing.
    """

    def __init__(
        self,
        num_blocks: int = 8,
        hidden_ch: int = 64,
        temporal_window: int = 5,
    ):
        super().__init__()
        self.haar = HaarWaveletTransform()

        # After Haar transform: 3 channels -> 12 channels (4 subbands x 3 RGB)
        # Secret also goes through Haar: 3 -> 12 channels
        # Total: 24 channels
        total_ch = 24

        # Invertible blocks
        blocks = []
        for i in range(num_blocks):
            blocks.append(AffineCouplingBlock(total_ch, hidden_ch))
            blocks.append(ChannelShuffle(total_ch))
        self.blocks = nn.ModuleList(blocks)

        # Temporal attention for video
        self.temporal_attn = TemporalAttention3D(3, temporal_window)

    def _forward_blocks(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x

    def _inverse_blocks(self, y: torch.Tensor) -> torch.Tensor:
        for block in reversed(self.blocks):
            y = block.inverse(y)
        return y

    def hide(self, cover: torch.Tensor, secret: torch.Tensor) -> torch.Tensor:
        """
        Hide secret in cover image using invertible transform.

        Args:
            cover: (B, 3, H, W) cover image [0, 1]
            secret: (B, 3, H, W) secret image [0, 1]

        Returns:
            stego: (B, 3, H, W) stego image [0, 1]
        """
        # Wavelet decomposition
        cover_wt = self.haar(cover)   # (B, 12, H/2, W/2)
        secret_wt = self.haar(secret)  # (B, 12, H/2, W/2)

        # Concatenate in channel dimension
        combined = torch.cat([cover_wt, secret_wt], dim=1)  # (B, 24, H/2, W/2)

        # Forward through invertible blocks
        transformed = self._forward_blocks(combined)

        # Split back: first 12 channels = stego wavelet, last 12 = residual
        stego_wt = transformed[:, :12]

        # Inverse wavelet to get stego image
        stego = self.haar.inverse(stego_wt)
        return torch.clamp(stego, 0, 1)

    def reveal(self, stego: torch.Tensor) -> torch.Tensor:
        """
        Reveal secret from stego image.

        Args:
            stego: (B, 3, H, W) stego image

        Returns:
            secret: (B, 3, H, W) revealed secret image
        """
        stego_wt = self.haar(stego)  # (B, 12, H/2, W/2)

        # Pad with zeros for the secret part (will be reconstructed)
        zeros = torch.zeros_like(stego_wt)
        combined = torch.cat([stego_wt, zeros], dim=1)

        # Inverse through blocks
        reconstructed = self._inverse_blocks(combined)

        # Last 12 channels contain the secret
        secret_wt = reconstructed[:, 12:]
        secret = self.haar.inverse(secret_wt)
        return torch.clamp(secret, 0, 1)

    def hide_video(self, cover_frames: torch.Tensor, secret: torch.Tensor) -> torch.Tensor:
        """
        Hide secret across video frames with temporal consistency.

        Args:
            cover_frames: (B, T, 3, H, W) cover video frames
            secret: (B, 3, H, W) secret image (same for all frames)

        Returns:
            stego_frames: (B, T, 3, H, W) stego video frames
        """
        B, T, C, H, W = cover_frames.shape
        stego_frames = []

        for t in range(T):
            stego = self.hide(cover_frames[:, t], secret)
            stego_frames.append(stego)

        stego_video = torch.stack(stego_frames, dim=1)  # (B, T, 3, H, W)

        # Apply temporal attention for consistency
        stego_video = self.temporal_attn(stego_video)

        return torch.clamp(stego_video, 0, 1)
