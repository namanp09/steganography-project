"""
Spectrogram-based GAN for Audio Steganography.

Works in STFT domain with psychoacoustic masking:
- Generator: Embeds data in perceptually masked regions
- Discriminator: Detects spectral anomalies + hidden data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from models.layers import ConvNeXtBlock, CBAM, ResidualDenseBlock


class PsychoacousticMask(nn.Module):
    """
    Learns psychoacoustic masking threshold.
    Identifies perceptually less-sensitive frequency regions.
    """

    def __init__(self, freq_bins: int = 513, base_ch: int = 32):
        super().__init__()
        self.freq_bins = freq_bins

        self.net = nn.Sequential(
            nn.Linear(freq_bins, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, freq_bins),
            nn.Sigmoid(),  # Mask in [0, 1]
        )

    def forward(self, magnitude: torch.Tensor) -> torch.Tensor:
        """
        Compute masking threshold from magnitude spectrogram.

        Args:
            magnitude: (B, freq_bins, time_frames)

        Returns:
            mask: (B, freq_bins, time_frames) in [0, 1]
        """
        # Average over time
        avg_mag = magnitude.mean(dim=2)  # (B, freq_bins)
        mask_freq = self.net(avg_mag)  # (B, freq_bins)

        # Expand to time dimension
        mask = mask_freq.unsqueeze(2)
        return mask


class AudioGANGenerator(nn.Module):
    """
    Generator for audio steganography in spectrogram domain.
    Embeds message in psychoacoustically masked regions.
    """

    def __init__(
        self,
        msg_length: int = 128,
        freq_bins: int = 513,
        base_ch: int = 32,
        n_mels: int = 64,
    ):
        super().__init__()
        self.msg_length = msg_length
        self.freq_bins = freq_bins
        self.base_ch = base_ch

        # Psychoacoustic masking
        self.mask_generator = PsychoacousticMask(freq_bins, base_ch)

        # Message embedding network (2D spectrogram processing)
        self.msg_embed = nn.Sequential(
            nn.Linear(msg_length, 256),
            nn.GELU(),
            nn.Linear(256, base_ch * freq_bins),
        )

        # Embedding strength
        self.strength = nn.Parameter(torch.tensor(0.05))

        # Residual encoder in spectrogram domain (no striding to preserve dimensions)
        self.encoder = nn.Sequential(
            nn.Conv2d(1 + base_ch, base_ch, 3, padding=1),
            nn.GELU(),
            ConvNeXtBlock(base_ch),
            nn.Conv2d(base_ch, base_ch * 2, 3, padding=1),
            nn.GELU(),
            ConvNeXtBlock(base_ch * 2),
            nn.Conv2d(base_ch * 2, base_ch, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(base_ch, 1, 3, padding=1),
        )

    def forward(
        self,
        magnitude: torch.Tensor,
        phase: torch.Tensor,
        message: torch.Tensor,
    ) -> torch.Tensor:
        """
        Embed message in spectrogram magnitude.

        Args:
            magnitude: (B, 1, freq_bins, time_frames) magnitude spectrogram
            phase: (B, 1, freq_bins, time_frames) phase
            message: (B, msg_length) message bits

        Returns:
            stego_magnitude: (B, 1, freq_bins, time_frames)
        """
        # Get masking threshold
        mask = self.mask_generator(magnitude.squeeze(1))  # (B, freq_bins, time_frames)
        mask = mask.unsqueeze(1)  # (B, 1, freq_bins, time_frames)

        # Embed message
        msg_embed = self.msg_embed(message)
        msg_embed = msg_embed.view(-1, self.base_ch, self.freq_bins, 1)
        msg_embed = msg_embed.expand(-1, -1, -1, magnitude.shape[-1])

        # Concatenate magnitude with embedded message
        x = torch.cat([magnitude, msg_embed], dim=1)

        # Encode residual
        residual = self.encoder(x)

        # Apply masked embedding
        stego_mag = magnitude + residual * self.strength * mask
        stego_mag = torch.clamp(stego_mag, min=1e-5)

        return stego_mag


class AudioGANDiscriminator(nn.Module):
    """
    Discriminator for audio spectrogram (2D CNN).
    """

    def __init__(self, base_ch: int = 32):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1, base_ch, 4, stride=2, padding=1),
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

        # Task heads
        self.classifier = nn.Sequential(
            nn.Linear(base_ch * 8, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
        )

        self.steganalyzer = nn.Sequential(
            nn.Linear(base_ch * 8, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, magnitude: torch.Tensor) -> tuple:
        """
        Args:
            magnitude: (B, 1, freq_bins, time_frames)

        Returns:
            (real_score, stego_score)
        """
        features = self.net(magnitude)
        real_score = self.classifier(features)
        stego_score = self.steganalyzer(features)
        return real_score, stego_score


class AudioGANSteganography(nn.Module):
    """
    Complete audio GAN steganography system (spectrogram domain).
    """

    def __init__(
        self,
        msg_length: int = 128,
        freq_bins: int = 513,
        base_ch: int = 32,
    ):
        super().__init__()
        self.msg_length = msg_length
        self.freq_bins = freq_bins

        self.generator = AudioGANGenerator(msg_length, freq_bins, base_ch)
        self.discriminator = AudioGANDiscriminator(base_ch)

        # Message decoder (improved with more capacity)
        self.decoder = nn.Sequential(
            nn.Conv2d(1, base_ch, 3, padding=1),
            nn.BatchNorm2d(base_ch),
            nn.GELU(),
            ConvNeXtBlock(base_ch),
            nn.Dropout2d(0.1),
            nn.Conv2d(base_ch, base_ch * 2, 3, stride=(2, 2), padding=1),
            nn.BatchNorm2d(base_ch * 2),
            nn.GELU(),
            ConvNeXtBlock(base_ch * 2),
            nn.Conv2d(base_ch * 2, base_ch * 4, 3, stride=(2, 2), padding=1),
            nn.BatchNorm2d(base_ch * 4),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
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
        magnitude: torch.Tensor,
        phase: torch.Tensor,
        message: torch.Tensor,
    ) -> tuple:
        """
        Full forward: embed and extract message.

        Args:
            magnitude: (B, 1, freq_bins, time_frames)
            phase: (B, 1, freq_bins, time_frames)
            message: (B, msg_length)

        Returns:
            (stego_magnitude, decoded_msg)
        """
        stego_mag = self.generator(magnitude, phase, message)
        decoded = self.decoder(stego_mag)
        return stego_mag, decoded

    def discriminate(self, magnitude: torch.Tensor) -> tuple:
        """Discriminate spectrogram."""
        return self.discriminator(magnitude)
