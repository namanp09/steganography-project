#!/usr/bin/env python3
"""
Improved Audio GAN Training with Focus on Message Recovery.
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

from models.audio_gan import AudioGANSteganography
from config.settings import AUDIO_GAN, PATHS


def create_realistic_audio_dataset(num_samples: int = 100) -> DataLoader:
    """Create realistic audio spectrogram dataset."""
    print(f"Creating {num_samples} audio spectrograms...")

    magnitudes = []
    phases = []
    messages = []

    for _ in range(num_samples):
        # Create magnitude spectrogram (freq_bins x time_frames)
        mag = torch.exp(torch.randn(1, AUDIO_GAN.freq_bins, 64) * 0.5 - 1)
        phase = torch.randn(1, AUDIO_GAN.freq_bins, 64)

        magnitudes.append(mag)
        phases.append(phase)
        messages.append(torch.randint(0, 2, (AUDIO_GAN.message_bits,)).float())

    magnitudes = torch.stack(magnitudes)
    phases = torch.stack(phases)
    messages = torch.stack(messages)

    dataset = TensorDataset(magnitudes, phases, messages)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True, num_workers=0, pin_memory=False)

    print(f"Audio dataset: {len(magnitudes)} samples, shape {magnitudes[0].shape}")
    return loader


def train_audio_gan_improved(epochs: int = 50):
    """Train Audio GAN with focus on message recovery."""
    print("\n" + "="*80)
    print("TRAINING AUDIO GAN (IMPROVED)")
    print("="*80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = AudioGANSteganography(
        msg_length=AUDIO_GAN.message_bits,
        freq_bins=AUDIO_GAN.freq_bins,
        base_ch=AUDIO_GAN.base_channels,
    ).to(device)

    loader = create_realistic_audio_dataset(num_samples=100)

    # Optimizers with improved learning rates
    opt_dec = torch.optim.Adam(model.decoder.parameters(), lr=1e-2, betas=(0.9, 0.999))
    opt_g = torch.optim.Adam(model.generator.parameters(), lr=2e-4, betas=(0.9, 0.999))
    opt_d = torch.optim.Adam(model.discriminator.parameters(), lr=2e-4, betas=(0.9, 0.999))

    scheduler_g = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_g, mode='min', factor=0.5, patience=5)
    scheduler_dec = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_dec, mode='min', factor=0.5, patience=5)

    best_msg_loss = float('inf')
    ckpt_dir = Path(PATHS.models_dir) / "audio_gan_improved"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        msg_losses = []
        dec_acc = []

        for batch_idx, (magnitude, phase, message) in enumerate(loader):
            magnitude = magnitude.to(device)
            phase = phase.to(device)
            message = message.to(device)

            # Generator + Decoder Joint Training
            opt_g.zero_grad()
            opt_dec.zero_grad()

            stego_mag, decoded = model(magnitude, phase, message)

            # Message recovery loss
            msg_loss = F.binary_cross_entropy_with_logits(decoded, message)

            # Audio quality (keep stego close to original)
            audio_loss = F.mse_loss(stego_mag, magnitude)

            # Combined loss - balanced
            total_loss = msg_loss * 5.0 + audio_loss * 0.5
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
                test_msg = torch.randint(0, 2, (1, AUDIO_GAN.message_bits)).float().to(device)
                test_mag = torch.exp(torch.randn(1, 1, AUDIO_GAN.freq_bins, 64).to(device) * 0.5 - 1)
                test_phase = torch.randn(1, 1, AUDIO_GAN.freq_bins, 64).to(device)
                _, test_dec = model(test_mag, test_phase, test_msg)
                test_acc = ((torch.sigmoid(test_dec) > 0.5) == test_msg.bool()).float().mean().item() * 100
                print(f"  Test accuracy: {test_acc:.1f}%")

    print(f"✓ Audio GAN training complete. Best checkpoint: {ckpt_dir / 'best_model.pth'}")
    return ckpt_dir / "best_model.pth"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    args = parser.parse_args()

    try:
        train_audio_gan_improved(epochs=args.epochs)
    finally:
        import sys
        sys.exit(0)
