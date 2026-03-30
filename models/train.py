"""
Training Pipeline for Deep Learning Steganography Models.

Supports training:
1. U-Net++ Attention model
2. HiDDeN adversarial model
3. Invertible Neural Network

Features:
- Mixed precision training (AMP)
- Cosine annealing with warm restarts
- WandB experiment tracking
- Checkpoint saving/loading
- Multi-GPU support
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from pathlib import Path
from tqdm import tqdm

from config.settings import TRAINING, PATHS
from models.losses import SteganoLoss, WGANGPLoss

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


class StegoImageDataset(Dataset):
    """Dataset of images for steganography training."""

    def __init__(self, image_dir: str, image_size: int = 256, transform=None):
        from torchvision import transforms
        import cv2

        self.image_paths = []
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp"]:
            self.image_paths.extend(Path(image_dir).glob(ext))

        self.transform = transform or transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),  # Converts to [0, 1]
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        import cv2
        img = cv2.imread(str(self.image_paths[idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.transform(img)


class StegoVideoDataset(Dataset):
    """Dataset of video clips for video steganography training."""

    def __init__(
        self, video_dir: str, frame_size: int = 256,
        temporal_window: int = 5, transform=None,
    ):
        from torchvision import transforms
        self.video_paths = []
        for ext in ["*.mp4", "*.avi", "*.mkv"]:
            self.video_paths.extend(Path(video_dir).glob(ext))

        self.frame_size = frame_size
        self.temporal_window = temporal_window
        self.transform = transform or transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(frame_size),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.video_paths) * 10  # Multiple clips per video

    def __getitem__(self, idx):
        import cv2
        video_idx = idx % len(self.video_paths)
        cap = cv2.VideoCapture(str(self.video_paths[video_idx]))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        start = np.random.randint(0, max(1, total_frames - self.temporal_window))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

        frames = []
        for _ in range(self.temporal_window):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(self.transform(frame))

        cap.release()

        # Pad if needed
        while len(frames) < self.temporal_window:
            frames.append(frames[-1].clone())

        return torch.stack(frames)  # (T, 3, H, W)


def train_hidden(
    model,
    train_loader: DataLoader,
    epochs: int = None,
    msg_length: int = 128,
    device: str = "cuda",
    experiment_name: str = "hidden_training",
    checkpoint_dir: str = None,
):
    """
    Train HiDDeN model with adversarial training.

    Uses:
    - AdamW optimizer with cosine annealing
    - Mixed precision (AMP)
    - WGAN-GP for discriminator
    - Combined loss (image quality + message + frequency + adversarial)
    """
    epochs = epochs or TRAINING.epochs
    checkpoint_dir = checkpoint_dir or PATHS.models_dir
    os.makedirs(checkpoint_dir, exist_ok=True)

    model = model.to(device)

    # Separate optimizers for generator (encoder+decoder) and discriminator
    gen_params = list(model.encoder.parameters()) + list(model.decoder.parameters())
    opt_gen = torch.optim.AdamW(
        gen_params, lr=TRAINING.learning_rate, weight_decay=TRAINING.weight_decay
    )
    opt_disc = torch.optim.AdamW(
        model.discriminator.parameters(),
        lr=TRAINING.learning_rate * 0.1,
        weight_decay=TRAINING.weight_decay,
    )

    # Cosine annealing scheduler
    scheduler_gen = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt_gen, T_0=10, T_mult=2
    )

    scaler = GradScaler() if TRAINING.use_amp else None
    criterion = SteganoLoss(TRAINING).to(device)

    if HAS_WANDB:
        wandb.init(project="steganography", name=experiment_name)

    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        epoch_losses = {}

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, images in enumerate(pbar):
            images = images.to(device)
            B = images.shape[0]

            # Generate random binary message
            message = torch.randint(0, 2, (B, msg_length), dtype=torch.float32, device=device)

            # ─── Train Discriminator ───
            opt_disc.zero_grad()
            with torch.no_grad():
                stego = model.encoder(images, message)

            d_real = model.discriminator(images)
            d_fake = model.discriminator(stego)
            d_loss = WGANGPLoss.discriminator_loss(d_real, d_fake)
            gp = WGANGPLoss.gradient_penalty(model.discriminator, images, stego)
            d_total = d_loss + gp
            d_total.backward()
            opt_disc.step()

            # ─── Train Generator (Encoder + Decoder) ───
            opt_gen.zero_grad()

            if TRAINING.use_amp and scaler:
                with autocast():
                    stego, decoded_msg = model(images, message)
                    d_out = model.discriminator(stego)
                    losses = criterion(stego, images, decoded_msg, message, d_out)

                scaler.scale(losses["total"]).backward()
                scaler.step(opt_gen)
                scaler.update()
            else:
                stego, decoded_msg = model(images, message)
                d_out = model.discriminator(stego)
                losses = criterion(stego, images, decoded_msg, message, d_out)
                losses["total"].backward()
                opt_gen.step()

            scheduler_gen.step(epoch + batch_idx / len(train_loader))

            # Compute bit accuracy
            with torch.no_grad():
                predicted_bits = (torch.sigmoid(decoded_msg) > 0.5).float()
                bit_acc = (predicted_bits == message).float().mean().item()

            # Track losses
            losses["bit_accuracy"] = bit_acc
            losses["d_loss"] = d_loss.item()

            for k, v in losses.items():
                val = v.item() if isinstance(v, torch.Tensor) else v
                epoch_losses.setdefault(k, []).append(val)

            pbar.set_postfix({
                "loss": f"{losses['total'].item():.4f}",
                "bit_acc": f"{bit_acc:.4f}",
                "psnr": f"{-10 * np.log10(losses['mse'].item() + 1e-10):.1f}dB",
            })

        # Epoch summary
        avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
        print(f"\nEpoch {epoch+1} — Loss: {avg_losses['total']:.4f}, "
              f"Bit Acc: {avg_losses['bit_accuracy']:.4f}, "
              f"PSNR: {-10 * np.log10(avg_losses['mse'] + 1e-10):.1f}dB")

        if HAS_WANDB:
            wandb.log({f"train/{k}": v for k, v in avg_losses.items()}, step=epoch)

        # Save checkpoint
        if avg_losses["total"] < best_loss:
            best_loss = avg_losses["total"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "opt_gen_state_dict": opt_gen.state_dict(),
                "opt_disc_state_dict": opt_disc.state_dict(),
                "best_loss": best_loss,
            }, os.path.join(checkpoint_dir, "best_hidden.pth"))

        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
            }, os.path.join(checkpoint_dir, f"hidden_epoch_{epoch+1}.pth"))

    if HAS_WANDB:
        wandb.finish()

    return model


def train_unet(
    model,
    train_loader: DataLoader,
    epochs: int = None,
    msg_length: int = 128,
    device: str = "cuda",
    experiment_name: str = "unet_training",
):
    """Train U-Net steganography model (simpler, no discriminator)."""
    epochs = epochs or TRAINING.epochs
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=TRAINING.learning_rate, weight_decay=TRAINING.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )

    criterion = SteganoLoss(TRAINING).to(device)

    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for images in pbar:
            images = images.to(device)
            B = images.shape[0]
            message = torch.randint(0, 2, (B, msg_length), dtype=torch.float32, device=device)

            optimizer.zero_grad()
            stego, decoded_msg = model(images, message)
            losses = criterion(stego, images, decoded_msg, message)
            losses["total"].backward()
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                bit_acc = ((torch.sigmoid(decoded_msg) > 0.5).float() == message).float().mean()

            pbar.set_postfix({
                "loss": f"{losses['total'].item():.4f}",
                "bit_acc": f"{bit_acc:.4f}",
            })

    return model


def train_inn(
    model,
    train_loader: DataLoader,
    epochs: int = None,
    device: str = "cuda",
    experiment_name: str = "inn_training",
):
    """Train Invertible Neural Network for image-in-image hiding."""
    epochs = epochs or TRAINING.epochs
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=TRAINING.learning_rate, weight_decay=TRAINING.weight_decay
    )

    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for images in pbar:
            images = images.to(device)
            B = images.shape[0]

            # Split batch: first half = cover, second half = secret
            mid = B // 2
            if mid == 0:
                continue
            cover = images[:mid]
            secret = images[mid : mid * 2]
            if secret.shape[0] != cover.shape[0]:
                secret = secret[:cover.shape[0]]

            optimizer.zero_grad()

            # Hide and reveal
            stego = model.hide(cover, secret)
            revealed = model.reveal(stego)

            # Losses
            cover_loss = F.mse_loss(stego, cover)
            secret_loss = F.mse_loss(revealed, secret)
            total = cover_loss + secret_loss * 5.0  # Prioritize secret recovery

            total.backward()
            optimizer.step()

            pbar.set_postfix({
                "cover_psnr": f"{-10 * np.log10(cover_loss.item() + 1e-10):.1f}dB",
                "secret_psnr": f"{-10 * np.log10(secret_loss.item() + 1e-10):.1f}dB",
            })

    return model
