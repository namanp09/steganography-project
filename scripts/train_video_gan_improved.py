#!/usr/bin/env python3
"""
Improved Video GAN Training with Focus on Message Recovery.
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import sys
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

import torch

import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.video_gan import VideoGANSteganography
from config.settings import VIDEO_GAN, PATHS


def create_realistic_video_dataset(num_samples: int = 100) -> DataLoader:
    """Create realistic video dataset (random frame sequences)."""
    print(f"Creating {num_samples} video sequences...")

    videos = []
    messages = []

    for _ in range(num_samples):
        # Create video sequence (5 frames)
        video = torch.rand(VIDEO_GAN.temporal_window, 3, VIDEO_GAN.frame_size, VIDEO_GAN.frame_size)
        videos.append(video)
        messages.append(torch.randint(0, 2, (VIDEO_GAN.message_bits,)).float())

    videos = torch.stack(videos)
    messages = torch.stack(messages)

    dataset = TensorDataset(videos, messages)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True, num_workers=0, pin_memory=False)

    print(f"Video dataset: {len(videos)} samples, shape {videos[0].shape}")
    return loader


def train_video_gan_improved(epochs: int = 50):
    """Train Video GAN with focus on message recovery."""
    print("\n" + "="*80)
    print("TRAINING VIDEO GAN (IMPROVED)")
    print("="*80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = VideoGANSteganography(
        msg_length=VIDEO_GAN.message_bits,
        base_ch=VIDEO_GAN.base_channels,
        temporal_window=VIDEO_GAN.temporal_window,
        frame_size=VIDEO_GAN.frame_size,
    ).to(device)

    loader = create_realistic_video_dataset(num_samples=250)

    # Optimizers with improved learning rates
    opt_dec = torch.optim.Adam(model.decoder.parameters(), lr=1e-2, betas=(0.9, 0.999))
    opt_g = torch.optim.Adam(model.generator.parameters(), lr=2e-4, betas=(0.9, 0.999))
    opt_d = torch.optim.Adam(model.discriminator.parameters(), lr=2e-4, betas=(0.9, 0.999))

    scheduler_g = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_g, mode='min', factor=0.5, patience=5)
    scheduler_dec = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_dec, mode='min', factor=0.5, patience=5)

    best_msg_loss = float('inf')
    ckpt_dir = Path(PATHS.models_dir) / "video_gan_improved"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        msg_losses = []
        dec_acc = []

        for batch_idx, (video, message) in enumerate(loader):
            video = video.to(device)
            message = message.to(device)

            # Generator + Decoder Joint Training
            opt_g.zero_grad()
            opt_dec.zero_grad()

            stego_video, decoded = model(video, message)

            # Message recovery loss
            msg_loss = F.binary_cross_entropy_with_logits(decoded, message)

            # Video quality (keep stego close to original)
            video_loss = F.mse_loss(stego_video, video)

            # Combined loss - balanced
            total_loss = msg_loss * 5.0 + video_loss * 0.5
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(model.generator.parameters(), 1.0)

            opt_g.step()
            opt_dec.step()

            msg_losses.append(msg_loss.item())

            # Bit accuracy
            with torch.no_grad():
                bit_acc = ((torch.sigmoid(decoded) > 0.5) == message.bool()).float().mean()
                dec_acc.append(bit_acc.item())

        avg_msg_loss = sum(msg_losses) / len(msg_losses)
        avg_acc = sum(dec_acc) / len(dec_acc) * 100

        print(f"Epoch {epoch+1}: Message Loss={avg_msg_loss:.4f}, Accuracy={avg_acc:.1f}%")

        scheduler_dec.step(avg_msg_loss)
        scheduler_g.step(avg_msg_loss)

        # Save checkpoint
        if avg_msg_loss < best_msg_loss:
            best_msg_loss = avg_msg_loss
            torch.save(model.state_dict(), ckpt_dir / "best_model.pth")
            print(f"  ✓ Best checkpoint saved (loss={best_msg_loss:.4f})")

        # Test on sample
        if (epoch + 1) % 5 == 0:
            with torch.no_grad():
                test_msg = torch.randint(0, 2, (1, VIDEO_GAN.message_bits)).float().to(device)
                test_video = torch.rand(1, VIDEO_GAN.temporal_window, 3, VIDEO_GAN.frame_size, VIDEO_GAN.frame_size).to(device)
                _, test_dec = model(test_video, test_msg)
                test_acc = ((torch.sigmoid(test_dec) > 0.5) == test_msg.bool()).float().mean().item() * 100
                print(f"  Test accuracy: {test_acc:.1f}%")

    print(f"✓ Video GAN training complete. Best checkpoint: {ckpt_dir / 'best_model.pth'}")
    return ckpt_dir / "best_model.pth"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    args = parser.parse_args()

    try:
        train_video_gan_improved(epochs=args.epochs)
    finally:
        import sys
        sys.exit(0)
