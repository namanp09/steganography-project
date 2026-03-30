"""
Attention U-Net++ Steganography Model.

Modern architecture combining:
- U-Net++ dense skip connections
- CBAM attention at every level
- ConvNeXt-style blocks for feature extraction
- Separate encoder (hides message) and decoder (extracts message)
- Multi-scale message injection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import CBAM, ConvNeXtBlock, ResidualDenseBlock, NoiseLayer


class DownBlock(nn.Module):
    """Encoder downsampling block with attention."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.GELU(),
            ConvNeXtBlock(out_ch),
            CBAM(out_ch),
        )

    def forward(self, x):
        return self.conv(x)


class UpBlock(nn.Module):
    """Decoder upsampling block with skip connection and attention."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch // 2 + skip_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.GELU(),
            ConvNeXtBlock(out_ch),
            CBAM(out_ch),
        )

    def forward(self, x, skip):
        x = self.up(x)
        # Handle size mismatch
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class MessageProcessor(nn.Module):
    """
    Processes binary message into spatial feature maps.
    Expands message bits to match spatial dimensions at each scale.
    """

    def __init__(self, msg_length: int, channels: int, spatial_size: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(msg_length, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels * spatial_size * spatial_size),
        )
        self.channels = channels
        self.spatial_size = spatial_size

    def forward(self, msg: torch.Tensor) -> torch.Tensor:
        """msg: (B, msg_length) -> (B, channels, H, W)"""
        out = self.fc(msg)
        return out.view(-1, self.channels, self.spatial_size, self.spatial_size)


class AttentionUNetEncoder(nn.Module):
    """
    Encoder network: takes cover image + message -> stego image.

    Architecture:
    - Multi-scale message injection (message features added at every level)
    - Dense skip connections (U-Net++ style)
    - CBAM attention at each level
    - Residual learning: outputs residual added to cover image
    """

    def __init__(self, msg_length: int = 128, base_ch: int = 64, image_size: int = 256):
        super().__init__()
        self.msg_length = msg_length

        # Initial convolution
        self.init_conv = nn.Sequential(
            nn.Conv2d(3, base_ch, 3, padding=1),
            nn.GroupNorm(8, base_ch),
            nn.GELU(),
        )

        # Encoder path
        self.down1 = DownBlock(base_ch, base_ch * 2)       # 256 -> 128
        self.down2 = DownBlock(base_ch * 2, base_ch * 4)   # 128 -> 64
        self.down3 = DownBlock(base_ch * 4, base_ch * 8)   # 64 -> 32
        self.down4 = DownBlock(base_ch * 8, base_ch * 16)  # 32 -> 16

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResidualDenseBlock(base_ch * 16),
            ResidualDenseBlock(base_ch * 16),
        )

        # Message processors at each scale
        self.msg_proc_4 = MessageProcessor(msg_length, base_ch * 16, image_size // 16)
        self.msg_proc_3 = MessageProcessor(msg_length, base_ch * 8, image_size // 8)
        self.msg_proc_2 = MessageProcessor(msg_length, base_ch * 4, image_size // 4)
        self.msg_proc_1 = MessageProcessor(msg_length, base_ch * 2, image_size // 2)

        # Decoder path
        self.up4 = UpBlock(base_ch * 16 + base_ch * 16, base_ch * 8, base_ch * 8)
        self.up3 = UpBlock(base_ch * 8 + base_ch * 8, base_ch * 4, base_ch * 4)
        self.up2 = UpBlock(base_ch * 4 + base_ch * 4, base_ch * 2, base_ch * 2)
        self.up1 = UpBlock(base_ch * 2 + base_ch * 2, base_ch, base_ch)

        # Final output: 3-channel residual
        self.final = nn.Sequential(
            ConvNeXtBlock(base_ch),
            nn.Conv2d(base_ch, 3, 1),
            nn.Tanh(),  # Residual in [-1, 1]
        )

        # Residual strength (learnable)
        self.strength = nn.Parameter(torch.tensor(0.1))

    def forward(self, cover: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cover: (B, 3, H, W) cover image in [0, 1]
            message: (B, msg_length) binary message

        Returns:
            stego: (B, 3, H, W) stego image in [0, 1]
        """
        # Encode cover image
        x0 = self.init_conv(cover)           # (B, 64, 256, 256)
        x1 = self.down1(x0)                  # (B, 128, 128, 128)
        x2 = self.down2(x1)                  # (B, 256, 64, 64)
        x3 = self.down3(x2)                  # (B, 512, 32, 32)
        x4 = self.down4(x3)                  # (B, 1024, 16, 16)

        # Inject message at bottleneck
        msg4 = self.msg_proc_4(message)
        x4 = self.bottleneck(x4 + msg4)

        # Inject message at each decoder level
        msg3 = self.msg_proc_3(message)
        msg2 = self.msg_proc_2(message)
        msg1 = self.msg_proc_1(message)

        d4 = self.up4(torch.cat([x4, msg4], dim=1), x3 + msg3)
        d3 = self.up3(torch.cat([d4, msg3], dim=1), x2 + msg2)
        d2 = self.up2(torch.cat([d3, msg2], dim=1), x1 + msg1)
        d1 = self.up1(torch.cat([d2, msg1], dim=1), x0)

        # Output residual
        residual = self.final(d1) * self.strength

        stego = torch.clamp(cover + residual, 0, 1)
        return stego


class AttentionUNetDecoder(nn.Module):
    """
    Decoder network: takes stego image -> extracted message.

    Uses a lightweight CNN with attention to extract the hidden message.
    """

    def __init__(self, msg_length: int = 128, base_ch: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            # Encoder
            nn.Conv2d(3, base_ch, 3, stride=2, padding=1),
            nn.GroupNorm(8, base_ch),
            nn.GELU(),
            CBAM(base_ch),

            nn.Conv2d(base_ch, base_ch * 2, 3, stride=2, padding=1),
            nn.GroupNorm(8, base_ch * 2),
            nn.GELU(),
            CBAM(base_ch * 2),

            nn.Conv2d(base_ch * 2, base_ch * 4, 3, stride=2, padding=1),
            nn.GroupNorm(8, base_ch * 4),
            nn.GELU(),
            CBAM(base_ch * 4),

            nn.Conv2d(base_ch * 4, base_ch * 8, 3, stride=2, padding=1),
            nn.GroupNorm(8, base_ch * 8),
            nn.GELU(),

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        self.fc = nn.Sequential(
            nn.Linear(base_ch * 8, base_ch * 4),
            nn.GELU(),
            nn.Linear(base_ch * 4, msg_length),
        )

    def forward(self, stego: torch.Tensor) -> torch.Tensor:
        """
        Args:
            stego: (B, 3, H, W) stego image

        Returns:
            message: (B, msg_length) predicted message logits
        """
        features = self.network(stego)
        return self.fc(features)


class UNetSteganography(nn.Module):
    """
    Complete U-Net steganography system.
    Combines encoder, noise layer, and decoder.
    """

    def __init__(self, msg_length: int = 128, base_ch: int = 64, image_size: int = 256):
        super().__init__()
        self.encoder = AttentionUNetEncoder(msg_length, base_ch, image_size)
        self.decoder = AttentionUNetDecoder(msg_length, base_ch)
        self.noise_layer = NoiseLayer()

    def forward(self, cover: torch.Tensor, message: torch.Tensor):
        """
        Full forward pass: cover + message -> stego -> (noise) -> extracted message.

        Returns:
            stego: stego image
            decoded_msg: extracted message logits
        """
        stego = self.encoder(cover, message)
        noisy_stego = self.noise_layer(stego)
        decoded_msg = self.decoder(noisy_stego)
        return stego, decoded_msg
