"""
Spatio-Temporal GAN for Video Steganography.

Architecture:
- 3D CNN generator for frame-level + temporal consistency
- Motion-aware embedding (focus on high-motion regions)
- Temporal discriminator for frame coherence
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import ConvNeXtBlock, CBAM, ResidualDenseBlock


class MotionAwareModule(nn.Module):
    """
    Computes motion map from optical flow or frame differences.
    Identifies regions safe for embedding.
    """

    def __init__(self, base_ch: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, base_ch, 3, padding=1),  # 2 = (dx, dy) from optical flow
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, 1, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, flow: torch.Tensor) -> torch.Tensor:
        """
        Args:
            flow: (B, 2, H, W) optical flow

        Returns:
            motion_mask: (B, 1, H, W) in [0, 1]
        """
        return self.net(flow)


class Conv3DBlock(nn.Module):
    """3D convolutional residual block for temporal coherence."""

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size, padding=(padding, padding, padding)),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv3d(channels, channels, kernel_size, padding=(padding, padding, padding)),
            nn.GroupNorm(8, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv(x)


class VideoGANGenerator(nn.Module):
    """
    3D CNN generator for video steganography.
    Embeds message across frames with temporal consistency.
    """

    def __init__(
        self,
        msg_length: int = 128,
        base_ch: int = 32,
        temporal_window: int = 5,
        frame_size: int = 256,
    ):
        super().__init__()
        self.msg_length = msg_length
        self.base_ch = base_ch
        self.temporal_window = temporal_window
        self.frame_size = frame_size

        # Motion awareness
        self.motion_module = MotionAwareModule(base_ch)

        # Message processor for temporal distribution
        self.msg_processor = nn.Sequential(
            nn.Linear(msg_length, 512),
            nn.GELU(),
            nn.Linear(512, base_ch * (frame_size // 4) * (frame_size // 4) * temporal_window),
        )

        # Embedding strength (learnable)
        self.strength = nn.Parameter(torch.tensor(0.05))

        # 3D encoder for spatio-temporal embedding
        self.encoder_3d = nn.Sequential(
            nn.Conv3d(3 + base_ch, base_ch, (3, 3, 3), padding=(1, 1, 1)),
            nn.GroupNorm(8, base_ch),
            nn.GELU(),
            Conv3DBlock(base_ch),
            nn.Conv3d(base_ch, base_ch * 2, (3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.GroupNorm(8, base_ch * 2),
            nn.GELU(),
            Conv3DBlock(base_ch * 2),
            nn.Conv3d(base_ch * 2, base_ch, (3, 3, 3), padding=(1, 1, 1)),
            nn.GroupNorm(8, base_ch),
            nn.GELU(),
            nn.ConvTranspose3d(base_ch, 3, (3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)),
        )

    def forward(
        self,
        cover_frames: torch.Tensor,
        message: torch.Tensor,
        flow: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Embed message into video frames.

        Args:
            cover_frames: (B, T, 3, H, W) video clip
            message: (B, msg_length) message bits
            flow: (B, T-1, 2, H, W) optional optical flow

        Returns:
            stego_frames: (B, T, 3, H, W)
        """
        B, T, C, H, W = cover_frames.shape

        # Rearrange to (B, C, T, H, W) for 3D conv
        x = cover_frames.permute(0, 2, 1, 3, 4)

        # Compute motion mask if flow provided
        if flow is not None:
            # Average flow across time
            flow_avg = flow.mean(dim=1)  # (B, 2, H, W)
            motion_mask = self.motion_module(flow_avg)  # (B, 1, H, W)
        else:
            motion_mask = torch.ones(B, 1, H, W, device=x.device)

        # Process message for temporal distribution
        msg_temporal = self.msg_processor(message)
        msg_temporal = msg_temporal.view(B, self.base_ch, T, H // 4, W // 4)
        # Upsample to match input size
        msg_temporal = F.interpolate(
            msg_temporal.view(B * T, self.base_ch, H // 4, W // 4),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )
        msg_temporal = msg_temporal.view(B, T, self.base_ch, H, W).permute(0, 2, 1, 3, 4)

        # Concatenate cover with message
        x_concat = torch.cat([x, msg_temporal], dim=1)

        # Encode residual
        residual = self.encoder_3d(x_concat)

        # Expand motion mask to temporal dimension
        motion_mask_3d = motion_mask.unsqueeze(2).expand(-1, -1, T, -1, -1)

        # Apply motion-aware embedding
        stego = x + residual * self.strength * motion_mask_3d
        stego = torch.clamp(stego, 0, 1)

        # Rearrange back to (B, T, 3, H, W)
        stego = stego.permute(0, 2, 1, 3, 4)

        return stego


class TemporalDiscriminator(nn.Module):
    """
    Discriminator with temporal modeling for video.
    Checks frame quality and temporal smoothness.
    """

    def __init__(self, base_ch: int = 32):
        super().__init__()

        # Spatial feature extractor (per-frame)
        self.spatial_features = nn.Sequential(
            nn.Conv2d(3, base_ch, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch, base_ch * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_ch * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch * 2, base_ch * 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_ch * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Temporal aggregation (3D conv)
        self.temporal_conv = nn.Sequential(
            nn.Conv3d(base_ch * 4, base_ch * 4, (3, 1, 1), padding=(1, 0, 0)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(base_ch * 4, base_ch * 8, (3, 1, 1), padding=(1, 0, 0)),
        )

        # Classification heads
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(base_ch * 8, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
        )

        self.temporal_smoothness = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(base_ch * 8, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, frames: torch.Tensor) -> tuple:
        """
        Args:
            frames: (B, T, 3, H, W)

        Returns:
            (real_score, temporal_score)
        """
        B, T, C, H, W = frames.shape

        # Extract spatial features per frame
        frames_flat = frames.view(B * T, C, H, W)
        spatial_feats = self.spatial_features(frames_flat)  # (B*T, ch*4, h', w')

        _, ch, h, w = spatial_feats.shape
        spatial_feats = spatial_feats.view(B, T, ch, h, w)

        # Rearrange for 3D conv: (B, ch, T, h, w)
        spatial_feats = spatial_feats.permute(0, 2, 1, 3, 4)

        # Temporal modeling
        temporal_feats = self.temporal_conv(spatial_feats)

        # Classification
        real_score = self.classifier(temporal_feats)
        temporal_score = self.temporal_smoothness(temporal_feats)

        return real_score, temporal_score


class VideoGANSteganography(nn.Module):
    """
    Complete video GAN steganography system.
    """

    def __init__(
        self,
        msg_length: int = 128,
        base_ch: int = 32,
        temporal_window: int = 5,
        frame_size: int = 256,
    ):
        super().__init__()
        self.msg_length = msg_length
        self.temporal_window = temporal_window

        self.generator = VideoGANGenerator(msg_length, base_ch, temporal_window, frame_size)
        self.discriminator = TemporalDiscriminator(base_ch)

        # Message decoder from stego frames (improved with more capacity)
        self.decoder = nn.Sequential(
            nn.Conv3d(3, base_ch, (3, 3, 3), padding=(1, 1, 1)),
            nn.GroupNorm(8, base_ch),
            nn.GELU(),
            Conv3DBlock(base_ch),
            nn.Dropout(0.1),
            nn.Conv3d(base_ch, base_ch * 2, (3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.GroupNorm(8, base_ch * 2),
            nn.GELU(),
            Conv3DBlock(base_ch * 2),
            nn.Conv3d(base_ch * 2, base_ch * 4, (3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.GroupNorm(8, base_ch * 4),
            nn.GELU(),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(base_ch * 4, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, msg_length),
        )

    def forward(
        self,
        cover_frames: torch.Tensor,
        message: torch.Tensor,
        flow: torch.Tensor = None,
    ) -> tuple:
        """
        Full forward: embed message and extract it back.

        Args:
            cover_frames: (B, T, 3, H, W)
            message: (B, msg_length)
            flow: (B, T-1, 2, H, W) optional

        Returns:
            (stego_frames, decoded_msg)
        """
        stego = self.generator(cover_frames, message, flow)

        # Rearrange for decoder: (B, T, 3, H, W) -> (B, 3, T, H, W)
        stego_for_decoder = stego.permute(0, 2, 1, 3, 4)
        decoded = self.decoder(stego_for_decoder)

        return stego, decoded

    def discriminate(self, frames: torch.Tensor) -> tuple:
        """Discriminate video frames."""
        return self.discriminator(frames)
