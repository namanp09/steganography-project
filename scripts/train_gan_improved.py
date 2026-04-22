#!/usr/bin/env python3
"""
Improved GAN Training with Focus on Message Recovery.
This trains the models for full encode/decode functionality.
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import sys
import multiprocessing
if sys.platform != "win32":
    multiprocessing.set_start_method('fork', force=True)

import torch

import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.image_gan import ImageGANSteganography
from models.audio_gan import AudioGANSteganography
from models.video_gan import VideoGANSteganography
from config.settings import IMAGE_GAN, AUDIO_GAN, VIDEO_GAN, PATHS


def create_realistic_image_dataset(num_samples: int = 250) -> DataLoader:
    """Create more realistic image dataset."""
    print(f"Creating {num_samples} realistic images...")

    images = []
    messages = []

    for _ in range(num_samples):
        # Create more realistic images with patterns
        img = torch.zeros(3, IMAGE_GAN.image_size, IMAGE_GAN.image_size)

        # Add gradients
        for i in range(IMAGE_GAN.image_size):
            img[:, i, :] = torch.rand(3, IMAGE_GAN.image_size) * 0.8

        # Add some structure
        for j in range(0, IMAGE_GAN.image_size, 32):
            img[:, j:j+32, :] += torch.randn(3, 32, IMAGE_GAN.image_size) * 0.3

        img = torch.clamp(img, 0, 1)
        images.append(img)
        messages.append(torch.randint(0, 2, (IMAGE_GAN.message_bits,)).float())

    images = torch.stack(images)
    messages = torch.stack(messages)

    dataset = TensorDataset(images, messages)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True, num_workers=0, pin_memory=False)

    print(f"Image dataset: {len(images)} samples, shape {images[0].shape}")
    return loader


def train_image_gan_improved(epochs: int = 50):
    """Train Image GAN with focus on message recovery."""
    print("\n" + "="*80)
    print("TRAINING IMAGE GAN (IMPROVED)")
    print("="*80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = ImageGANSteganography(
        msg_length=IMAGE_GAN.message_bits,
        base_ch=IMAGE_GAN.base_channels,
        image_size=IMAGE_GAN.image_size,
    ).to(device)

    loader = create_realistic_image_dataset(num_samples=100)

    # Optimizers - decoder gets highest priority
    opt_dec = torch.optim.Adam(model.decoder.parameters(), lr=1e-2, betas=(0.9, 0.999))
    opt_g = torch.optim.Adam(model.generator.parameters(), lr=2e-4, betas=(0.9, 0.999))
    opt_d = torch.optim.Adam(model.discriminator.parameters(), lr=2e-4, betas=(0.9, 0.999))

    scheduler_g = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_g, mode='min', factor=0.5, patience=5)
    scheduler_dec = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_dec, mode='min', factor=0.5, patience=5)

    best_msg_loss = float('inf')
    ckpt_dir = Path(PATHS.models_dir) / "image_gan_improved"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        msg_losses = []
        dec_acc = []

        for batch_idx, (cover, message) in enumerate(loader):
            cover = cover.to(device)
            message = message.to(device)

            # ──── Generator + Decoder Joint Training ────
            opt_g.zero_grad()
            opt_dec.zero_grad()

            stego, decoded = model(cover, message)

            # Strong message recovery loss
            msg_loss = F.binary_cross_entropy_with_logits(decoded, message)

            # Image quality (keep stego close to cover)
            img_loss = F.mse_loss(stego, cover)

            # Combined loss - balanced for better learning
            total_loss = msg_loss * 5.0 + img_loss * 0.5
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(model.generator.parameters(), 1.0)

            opt_g.step()
            opt_dec.step()

            msg_losses.append(msg_loss.item())

            # Compute bit accuracy
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
                test_msg = torch.randint(0, 2, (1, IMAGE_GAN.message_bits)).float().to(device)
                test_cover = torch.rand(1, 3, IMAGE_GAN.image_size, IMAGE_GAN.image_size).to(device)
                _, test_dec = model(test_cover, test_msg)
                test_acc = ((torch.sigmoid(test_dec) > 0.5) == test_msg.bool()).float().mean().item() * 100
                print(f"  Test accuracy: {test_acc:.1f}%")

    print(f"✓ Image GAN training complete. Best checkpoint: {ckpt_dir / 'best_model.pth'}")
    return ckpt_dir / "best_model.pth"


def test_gan_encode_decode():
    """Test full encode/decode cycle."""
    print("\n" + "="*80)
    print("TESTING GAN ENCODE/DECODE")
    print("="*80)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load trained model
    model = ImageGANSteganography(
        msg_length=IMAGE_GAN.message_bits,
        base_ch=IMAGE_GAN.base_channels,
        image_size=IMAGE_GAN.image_size,
    ).to(device)

    ckpt_path = Path(PATHS.models_dir) / "image_gan_improved" / "best_model.pth"
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"✓ Loaded checkpoint: {ckpt_path}")
    else:
        print(f"✗ Checkpoint not found: {ckpt_path}")
        return

    model.eval()

    # Test
    with torch.no_grad():
        test_cover = torch.rand(1, 3, IMAGE_GAN.image_size, IMAGE_GAN.image_size).to(device)
        test_msg = torch.randint(0, 2, (1, IMAGE_GAN.message_bits)).float().to(device)

        stego, decoded = model(test_cover, test_msg)

        # Binarize
        decoded_binary = (torch.sigmoid(decoded) > 0.5).float()

        # Check match
        match = (decoded_binary == test_msg).all(dim=1).item()
        accuracy = ((decoded_binary == test_msg).float().mean()).item() * 100

        print(f"Original message: {test_msg[0].long().tolist()}")
        print(f"Decoded message:  {decoded_binary[0].long().tolist()}")
        print(f"Accuracy: {accuracy:.1f}%")
        print(f"Perfect recovery: {'✓ YES' if match else '✗ NO'}")

    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--test-only", action="store_true", help="Test only")
    args = parser.parse_args()

    try:
        if args.test_only:
            test_gan_encode_decode()
        else:
            train_image_gan_improved(epochs=args.epochs)
            test_gan_encode_decode()
    finally:
        import sys
        sys.exit(0)
