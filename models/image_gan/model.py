"""
Adaptive Cost Learning GAN for Image Steganography.

Architecture:
- Generator: Learn adaptive embedding cost map + perform frequency-domain embedding
- Discriminator: Dual-task (real/fake classifier + steganalyzer)
- Uses residual learning with learnable strength parameter
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import (
    CBAM,
    ConvNeXtBlock,
    ResidualDenseBlock,
    NoiseLayer,
)


class CostMapGenerator(nn.Module):
    """
    Generate adaptive cost map for embedding guidance.
    Learns which regions are safe for embedding (high visual importance = low cost).
    """

    def __init__(self, in_ch: int = 3, base_ch: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            ConvNeXtBlock(base_ch),
            nn.Conv2d(base_ch, base_ch * 2, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            ConvNeXtBlock(base_ch * 2),
            nn.Conv2d(base_ch * 2, base_ch, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_ch, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),  # Cost in [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) cover image

        Returns:
            cost: (B, 1) scalar cost factor per image
        """
        return self.net(x)


class ImageGANGenerator(nn.Module):
    """
    Generator for adaptive cost learning GAN steganography.
    Encodes message into cover image with learned embedding cost.
    """

    def __init__(self, msg_length: int = 128, base_ch: int = 64, image_size: int = 256):
        super().__init__()
        self.msg_length = msg_length
        self.base_ch = base_ch
        self.image_size = image_size

        # Cost map generator
        self.cost_generator = CostMapGenerator(in_ch=3, base_ch=32)

        # Message processor: expand message to spatial feature map
        self.msg_processor = nn.Sequential(
            nn.Linear(msg_length, 256),
            nn.GELU(),
            nn.Linear(256, base_ch * (image_size // 4) * (image_size // 4)),
        )

        # Main encoder network
        self.encoder = nn.Sequential(
            nn.Conv2d(3 + base_ch, base_ch, 3, padding=1),
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
            ResidualDenseBlock(base_ch * 4),
            nn.Conv2d(base_ch * 4, base_ch * 2, 3, padding=1),
            nn.GroupNorm(8, base_ch * 2),
            nn.GELU(),
            nn.ConvTranspose2d(base_ch * 2, base_ch, 4, stride=2, padding=1),
            nn.GroupNorm(8, base_ch),
            nn.GELU(),
            ConvNeXtBlock(base_ch),
            nn.ConvTranspose2d(base_ch, 3, 4, stride=2, padding=1),
        )

        # Learnable embedding strength
        self.strength = nn.Parameter(torch.tensor(0.1))

    def forward(self, cover: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        """
        Embed message into cover image.

        Args:
            cover: (B, 3, H, W) cover image
            message: (B, msg_length) message bits

        Returns:
            stego: (B, 3, H, W) stego image
        """
        # Get adaptive cost
        cost = self.cost_generator(cover)

        # Process message to spatial features
        msg_spatial = self.msg_processor(message)
        msg_spatial = msg_spatial.view(
            -1, self.base_ch, self.image_size // 4, self.image_size // 4
        )

        # Upsample message features to match cover resolution
        msg_spatial = F.interpolate(
            msg_spatial, size=(self.image_size, self.image_size),
            mode='bilinear', align_corners=False
        )

        # Concatenate cover with message features
        x = torch.cat([cover, msg_spatial], dim=1)

        # Encode residual
        residual = self.encoder(x)

        # Combine with learnable strength and cost
        # Expand cost to match residual shape
        cost_expanded = cost.view(-1, 1, 1, 1)
        stego = cover + residual * self.strength * cost_expanded
        stego = torch.clamp(stego, 0, 1)

        return stego


class ImageGANDiscriminator(nn.Module):
    """
    Dual-task discriminator for image steganography GAN.
    - Task 1: Real vs Fake classification
    - Task 2: Steganalysis (detect hidden message)
    """

    def __init__(self, base_ch: int = 64):
        super().__init__()

        # Shared feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(3, base_ch, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch, base_ch * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_ch * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch * 2, base_ch * 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_ch * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch * 4, base_ch * 8, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_ch * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        # Task 1: Real/Fake classifier
        self.classifier = nn.Sequential(
            nn.Linear(base_ch * 8, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

        # Task 2: Steganalyzer
        self.steganalyzer = nn.Sequential(
            nn.Linear(base_ch * 8, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x: (B, 3, H, W) image

        Returns:
            (real_score, stego_score): (B, 1) each
        """
        features = self.features(x)
        real_score = self.classifier(features)
        stego_score = self.steganalyzer(features)
        return real_score, stego_score


class ImageGANSteganography(nn.Module):
    """
    Complete GAN-based steganography system for images.
    Combines generator, discriminator, and provides encode/decode interface.
    """

    def __init__(self, msg_length: int = 128, base_ch: int = 64, image_size: int = 256):
        super().__init__()
        self.msg_length = msg_length
        self.base_ch = base_ch
        self.image_size = image_size

        self.generator = ImageGANGenerator(msg_length, base_ch, image_size)
        self.discriminator = ImageGANDiscriminator(base_ch)
        self.noise_layer = NoiseLayer()

        # Message decoder: extract message from stego (improved with more capacity)
        self.decoder = nn.Sequential(
            nn.Conv2d(3, base_ch, 3, padding=1),
            nn.BatchNorm2d(base_ch),
            nn.GELU(),
            ConvNeXtBlock(base_ch),
            CBAM(base_ch),
            nn.Dropout2d(0.1),
            nn.Conv2d(base_ch, base_ch * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_ch * 2),
            nn.GELU(),
            ConvNeXtBlock(base_ch * 2),
            nn.Dropout2d(0.1),
            nn.Conv2d(base_ch * 2, base_ch * 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_ch * 4),
            nn.GELU(),
            ConvNeXtBlock(base_ch * 4),
            nn.Conv2d(base_ch * 4, base_ch * 8, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_ch * 8),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_ch * 8, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, msg_length),
        )

    def forward(self, cover: torch.Tensor, message: torch.Tensor) -> tuple:
        """
        Full forward pass: embed message and extract it back.

        Args:
            cover: (B, 3, H, W)
            message: (B, msg_length)

        Returns:
            (stego, decoded_msg): stego image and extracted logits
        """
        stego = self.generator(cover, message)

        # Apply noise during training for robustness
        if self.training:
            stego = self.noise_layer(stego)

        decoded = self.decoder(stego)
        return stego, decoded

    def discriminate(self, x: torch.Tensor) -> tuple:
        """
        Discriminate image: real/fake + steganalysis.

        Args:
            x: (B, 3, H, W)

        Returns:
            (real_score, stego_score)
        """
        return self.discriminator(x)
