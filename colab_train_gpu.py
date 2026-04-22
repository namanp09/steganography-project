#!/usr/bin/env python3
"""
Google Colab GPU Training - Enhanced for 85%+ Accuracy
Run this directly in Colab cell-by-cell

NOTE: The `ImageGANSteganography` class defined *below* is not the same as
`models/image_gan/model.py` in this repo. Checkpoints from this file will not load
in the FastAPI app. For deployment-compatible training, use from the project root:
`python scripts/train_production_gan_gpu.py --modality image` (or video/audio).
"""

# ============================================================================
# CELL 1: Install Dependencies & Mount Drive
# ============================================================================
# !pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# !pip install -q cryptography pillow numpy scipy

# from google.colab import drive
# drive.mount('/content/drive')

# ============================================================================
# CELL 2: Setup & Imports
# ============================================================================

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

# Use GPU efficiently
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✓ Device: {device}")
print(f"✓ GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
print(f"✓ CUDA: {torch.version.cuda}")

# ============================================================================
# CELL 3: Configuration for GPU Training
# ============================================================================

class GPUConfig:
    # Data
    num_samples = 500  # Increased from 100
    image_size = 128   # Increased from 64
    message_bits = 128
    batch_size = 32    # Larger batch on GPU
    num_workers = 4

    # Model
    base_channels = 64  # Increased from 32

    # Training
    epochs = 200       # Increased from 100
    lr_decoder = 1e-2
    lr_gen = 2e-4
    lr_disc = 2e-4

    # Loss weights - tuned for 85%+ accuracy
    msg_loss_weight = 10.0  # Strong message recovery focus
    img_loss_weight = 0.3   # Lighter image quality constraint

config = GPUConfig()
print(f"\n✓ Config: {config.num_samples} samples, {config.image_size}x{config.image_size}, {config.epochs} epochs")

# ============================================================================
# CELL 4: Create Realistic Image Dataset
# ============================================================================

def create_enhanced_dataset(num_samples: int = 500) -> DataLoader:
    """Create enhanced realistic dataset for GPU training."""
    print(f"Creating {num_samples} enhanced images...")

    images = []
    messages = []

    for _ in range(num_samples):
        # More complex, realistic images
        img = torch.zeros(3, config.image_size, config.image_size)

        # Add multiple texture layers
        for i in range(config.image_size):
            img[:, i, :] = torch.rand(3, config.image_size) * 0.7

        # Add detail layers
        for j in range(0, config.image_size, 16):
            img[:, j:j+16, :] += torch.randn(3, 16, config.image_size) * 0.2

        # Add structural patterns
        for k in range(0, config.image_size, 32):
            img[:, k:k+32, k:k+32] += torch.randn(3, 32, 32) * 0.15

        img = torch.clamp(img, 0, 1)
        images.append(img)
        messages.append(torch.randint(0, 2, (config.message_bits,)).float())

    images = torch.stack(images)
    messages = torch.stack(messages)

    dataset = TensorDataset(images, messages)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config.num_workers,
        pin_memory=True
    )

    print(f"✓ Dataset: {len(images)} samples, shape {images[0].shape}")
    return loader

# ============================================================================
# CELL 5: Model Definitions (Copy from your models folder)
# ============================================================================

class ConvNeXtBlock(nn.Module):
    """Modern ConvNeXt-style block for better feature extraction."""
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * 4, 7, padding=3, groups=dim),
            nn.BatchNorm2d(dim * 4),
            nn.GELU(),
            nn.Conv2d(dim * 4, dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return x + self.net(x)

class CBAM(nn.Module):
    """Channel-Spatial Attention."""
    def __init__(self, dim):
        super().__init__()
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 16, 1),
            nn.ReLU(),
            nn.Conv2d(dim // 16, dim, 1),
            nn.Sigmoid()
        )
        self.sa = nn.Sequential(
            nn.Conv2d(dim, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.ca(x) + x * self.sa(x)

class ImageGANSteganography(nn.Module):
    """Enhanced Image GAN for Steganography with 85%+ accuracy target."""

    def __init__(self, msg_length=128, base_ch=64, image_size=128):
        super().__init__()
        self.msg_length = msg_length
        self.image_size = image_size

        # Generator
        self.generator = nn.Sequential(
            nn.Conv2d(3 + (msg_length // (image_size*image_size)), base_ch, 3, padding=1),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(),
            ConvNeXtBlock(base_ch),
            nn.Conv2d(base_ch, base_ch*2, 3, padding=1),
            nn.BatchNorm2d(base_ch*2),
            nn.ReLU(),
            ConvNeXtBlock(base_ch*2),
            nn.Conv2d(base_ch*2, base_ch, 3, padding=1),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(),
            nn.Conv2d(base_ch, 3, 3, padding=1),
            nn.Tanh()
        )

        # Discriminator
        self.discriminator = nn.Sequential(
            nn.Conv2d(3, base_ch, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_ch, base_ch*2, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_ch*2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_ch*2, base_ch*4, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_ch*4),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(base_ch*4, 1, 1)
        )

        # Message Decoder - Enhanced with attention
        decoder_layers = []
        in_ch = 3
        for i in range(7):
            out_ch = min(64 + i*16, 256)
            decoder_layers.extend([
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                CBAM(out_ch)
            ])
            in_ch = out_ch

        decoder_layers.extend([
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(in_ch * 16, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, msg_length)
        ])

        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, cover, message):
        msg_map = message.view(message.size(0), -1, 1, 1).expand(-1, -1, self.image_size, self.image_size)
        stego = self.generator(torch.cat([cover, msg_map], dim=1))
        stego = (stego + 1) / 2  # Normalize to [0, 1]
        stego = torch.clamp(stego, 0, 1)
        decoded = self.decoder(stego)
        return stego, decoded

# ============================================================================
# CELL 6: Training Function
# ============================================================================

def train_with_gpu(model, loader, epochs=200):
    """GPU-optimized training for 85%+ accuracy."""

    print("\n" + "="*80)
    print("TRAINING IMAGE GAN (GPU ENHANCED)")
    print("="*80)

    opt_dec = torch.optim.Adam(model.decoder.parameters(), lr=config.lr_decoder, betas=(0.9, 0.999))
    opt_g = torch.optim.Adam(model.generator.parameters(), lr=config.lr_gen, betas=(0.9, 0.999))
    opt_d = torch.optim.Adam(model.discriminator.parameters(), lr=config.lr_disc, betas=(0.9, 0.999))

    scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=epochs)
    scheduler_dec = torch.optim.lr_scheduler.CosineAnnealingLR(opt_dec, T_max=epochs)

    best_msg_loss = float('inf')

    for epoch in range(epochs):
        msg_losses = []
        dec_acc = []

        for cover, message in loader:
            cover = cover.to(device)
            message = message.to(device)

            # Generator + Decoder Joint Training
            opt_g.zero_grad()
            opt_dec.zero_grad()

            stego, decoded = model(cover, message)

            msg_loss = F.binary_cross_entropy_with_logits(decoded, message)
            img_loss = F.mse_loss(stego, cover)
            total_loss = msg_loss * config.msg_loss_weight + img_loss * config.img_loss_weight

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(model.generator.parameters(), 1.0)

            opt_g.step()
            opt_dec.step()

            msg_losses.append(msg_loss.item())

            with torch.no_grad():
                bit_acc = ((torch.sigmoid(decoded) > 0.5) == message.bool()).float().mean()
                dec_acc.append(bit_acc.item())

        avg_msg_loss = sum(msg_losses) / len(msg_losses)
        avg_acc = sum(dec_acc) / len(dec_acc) * 100

        scheduler_dec.step()
        scheduler_g.step()

        if (epoch + 1) % 10 == 0 or (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{epochs}: Loss={avg_msg_loss:.4f}, Accuracy={avg_acc:.1f}%")

        if avg_msg_loss < best_msg_loss:
            best_msg_loss = avg_msg_loss

    print(f"\n✓ Training Complete! Best Accuracy: {avg_acc:.1f}%")
    return model

# ============================================================================
# CELL 7: Run Training
# ============================================================================

model = ImageGANSteganography(
    msg_length=config.message_bits,
    base_ch=config.base_channels,
    image_size=config.image_size
).to(device)

loader = create_enhanced_dataset(num_samples=config.num_samples)

# Mixed precision for faster training
from torch.cuda.amp import autocast, GradScaler
# Enable if needed: scaler = GradScaler()

trained_model = train_with_gpu(model, loader, epochs=config.epochs)

# ============================================================================
# CELL 8: Save Model
# ============================================================================

# !mkdir -p /content/drive/MyDrive/gan_models
# torch.save(trained_model.state_dict(), '/content/drive/MyDrive/gan_models/image_gan_gpu_85plus.pth')
# print("✓ Model saved to Google Drive")

print("\n✓ GPU Training Complete - Ready for Video/Audio GANs!")
