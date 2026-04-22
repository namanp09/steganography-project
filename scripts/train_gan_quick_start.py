#!/usr/bin/env python3
"""
Quick-start script for training GAN-based steganography models.
Includes sample data generation and training loops.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.image_gan import ImageGANSteganography
from models.audio_gan import AudioGANSteganography
from models.video_gan import VideoGANSteganography
from models.train_gan import train_image_gan, train_audio_gan, train_video_gan
from config.settings import IMAGE_GAN, AUDIO_GAN, VIDEO_GAN


def create_sample_image_dataset(num_samples: int = 10) -> DataLoader:
    """Create synthetic image dataset for quick testing."""
    print(f"Creating {num_samples} sample images...")

    images = torch.rand(num_samples, 3, IMAGE_GAN.image_size, IMAGE_GAN.image_size)
    messages = torch.randint(0, 2, (num_samples, IMAGE_GAN.message_bits)).float()

    dataset = TensorDataset(images, messages)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    print(f"Image dataset: {num_samples} samples, batch size=2")
    return loader


def create_sample_audio_dataset(num_samples: int = 5) -> DataLoader:
    """Create synthetic audio spectrogram dataset."""
    print(f"Creating {num_samples} sample spectrograms...")

    # Synthetic magnitude spectrograms
    magnitudes = torch.rand(
        num_samples, 1, AUDIO_GAN.freq_bins, 100
    )  # (B, 1, freq_bins, time_frames)
    phases = torch.rand(num_samples, 1, AUDIO_GAN.freq_bins, 100)
    messages = torch.randint(0, 2, (num_samples, AUDIO_GAN.message_bits)).float()

    dataset = TensorDataset(magnitudes, phases, messages)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    print(f"Audio dataset: {num_samples} samples")
    return loader


def create_sample_video_dataset(num_samples: int = 3) -> DataLoader:
    """Create synthetic video dataset."""
    print(f"Creating {num_samples} sample video clips...")

    frames = torch.rand(
        num_samples, VIDEO_GAN.temporal_window, 3, VIDEO_GAN.frame_size, VIDEO_GAN.frame_size
    )
    messages = torch.randint(0, 2, (num_samples, VIDEO_GAN.message_bits)).float()
    flows = torch.randn(num_samples, VIDEO_GAN.temporal_window - 1, 2, VIDEO_GAN.frame_size, VIDEO_GAN.frame_size)

    dataset = TensorDataset(frames, messages, flows)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    print(f"Video dataset: {num_samples} clips")
    return loader


def train_image_gan_quick():
    """Quick training of Image GAN."""
    print("\n" + "=" * 80)
    print("TRAINING IMAGE GAN")
    print("=" * 80)

    model = ImageGANSteganography(
        msg_length=IMAGE_GAN.message_bits,
        base_ch=IMAGE_GAN.base_channels,
        image_size=IMAGE_GAN.image_size,
    )

    loader = create_sample_image_dataset(num_samples=10)

    train_image_gan(
        model,
        loader,
        epochs=5,  # Quick training
        device="cuda" if torch.cuda.is_available() else "cpu",
        experiment_name="image_gan_quickstart",
    )

    print("✓ Image GAN training complete")


def train_audio_gan_quick():
    """Quick training of Audio GAN."""
    print("\n" + "=" * 80)
    print("TRAINING AUDIO GAN")
    print("=" * 80)

    model = AudioGANSteganography(
        msg_length=AUDIO_GAN.message_bits,
        freq_bins=AUDIO_GAN.freq_bins,
    )

    loader = create_sample_audio_dataset(num_samples=5)

    train_audio_gan(
        model,
        loader,
        epochs=5,
        device="cuda" if torch.cuda.is_available() else "cpu",
        experiment_name="audio_gan_quickstart",
    )

    print("✓ Audio GAN training complete")


def train_video_gan_quick():
    """Quick training of Video GAN."""
    print("\n" + "=" * 80)
    print("TRAINING VIDEO GAN")
    print("=" * 80)

    model = VideoGANSteganography(
        msg_length=VIDEO_GAN.message_bits,
        temporal_window=VIDEO_GAN.temporal_window,
        frame_size=VIDEO_GAN.frame_size,
    )

    loader = create_sample_video_dataset(num_samples=3)

    train_video_gan(
        model,
        loader,
        epochs=5,
        device="cuda" if torch.cuda.is_available() else "cpu",
        experiment_name="video_gan_quickstart",
    )

    print("✓ Video GAN training complete")


def test_inference():
    """Test inference on sample data."""
    print("\n" + "=" * 80)
    print("TESTING INFERENCE")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Image
    print("\nImage GAN inference...")
    img_model = ImageGANSteganography().to(device)
    cover = torch.rand(1, 3, IMAGE_GAN.image_size, IMAGE_GAN.image_size).to(device)
    msg = torch.randint(0, 2, (1, IMAGE_GAN.message_bits)).float().to(device)

    with torch.no_grad():
        stego, decoded = img_model(cover, msg)
    print(f"  Input shape: {cover.shape}")
    print(f"  Output shape: {stego.shape}")
    print(f"  Message decoded shape: {decoded.shape}")
    print("  ✓ Image inference OK")

    # Audio
    print("\nAudio GAN inference...")
    audio_model = AudioGANSteganography().to(device)
    mag = torch.rand(1, 1, AUDIO_GAN.freq_bins, 100).to(device)
    phase = torch.rand(1, 1, AUDIO_GAN.freq_bins, 100).to(device)
    msg = torch.randint(0, 2, (1, AUDIO_GAN.message_bits)).float().to(device)

    with torch.no_grad():
        stego_mag, decoded = audio_model(mag, phase, msg)
    print(f"  Input magnitude shape: {mag.shape}")
    print(f"  Output magnitude shape: {stego_mag.shape}")
    print("  ✓ Audio inference OK")

    # Video
    print("\nVideo GAN inference...")
    video_model = VideoGANSteganography().to(device)
    frames = torch.rand(1, VIDEO_GAN.temporal_window, 3, VIDEO_GAN.frame_size, VIDEO_GAN.frame_size).to(device)
    msg = torch.randint(0, 2, (1, VIDEO_GAN.message_bits)).float().to(device)

    with torch.no_grad():
        stego_frames, decoded = video_model(frames, msg, None)
    print(f"  Input frames shape: {frames.shape}")
    print(f"  Output frames shape: {stego_frames.shape}")
    print("  ✓ Video inference OK")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="GAN Training Quick Start")
    parser.add_argument("--modality", choices=["image", "audio", "video", "all"], default="all")
    parser.add_argument("--test-only", action="store_true", help="Only test inference")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    print(f"Device: {args.device}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if args.test_only:
        test_inference()
        return

    if args.modality in ["image", "all"]:
        train_image_gan_quick()

    if args.modality in ["audio", "all"]:
        train_audio_gan_quick()

    if args.modality in ["video", "all"]:
        train_video_gan_quick()

    print("\n" + "=" * 80)
    print("ALL TRAINING COMPLETE ✓")
    print("=" * 80)


if __name__ == "__main__":
    main()
