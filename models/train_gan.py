"""
Training Pipeline for GAN-based Steganography Models.

Supports training:
1. Image GAN (Adaptive Cost Learning)
2. Audio GAN (Spectrogram-based)
3. Video GAN (Spatio-Temporal)

Features:
- WGAN-GP adversarial training
- Mixed precision (AMP)
- Cosine annealing scheduler
- WandB tracking
- Checkpoint management
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np
from pathlib import Path
from tqdm import tqdm

from config.settings import GAN_TRAINING, PATHS, IMAGE_GAN, AUDIO_GAN, VIDEO_GAN
from models.losses import SteganoLoss, WGANGPLoss, ImageQualityLoss, FrequencyLoss, MessageLoss

try:
    import wandb
    HAS_WANDB = True
except (ImportError, Exception):
    HAS_WANDB = False


def _create_checkpoint_dir(name: str) -> Path:
    """Create checkpoint directory."""
    ckpt_dir = Path(PATHS.models_dir) / name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return ckpt_dir


# ──────────────────── Image GAN Training ────────────────────


def train_image_gan(
    model,
    train_loader: DataLoader,
    epochs: int = 100,
    device: str = "cuda",
    experiment_name: str = "image_gan",
):
    """
    Train Image GAN with WGAN-GP.

    Args:
        model: ImageGANSteganography instance
        train_loader: DataLoader yielding (cover_image, message) tuples
        epochs: Number of epochs
        device: Device to use
        experiment_name: Name for logging
    """
    if HAS_WANDB:
        try:
            wandb.init(project="steganography", name=f"{experiment_name}", mode="offline")
        except Exception:
            pass

    model = model.to(device)
    ckpt_dir = _create_checkpoint_dir(experiment_name)

    # Optimizers
    opt_g = torch.optim.AdamW(
        model.generator.parameters(),
        lr=GAN_TRAINING.learning_rate_g,
        weight_decay=GAN_TRAINING.weight_decay,
    )
    opt_d = torch.optim.AdamW(
        model.discriminator.parameters(),
        lr=GAN_TRAINING.learning_rate_d,
        weight_decay=GAN_TRAINING.weight_decay,
    )

    # Schedulers
    sched_g = CosineAnnealingWarmRestarts(opt_g, T_0=10, T_mult=2)
    sched_d = CosineAnnealingWarmRestarts(opt_d, T_0=10, T_mult=2)

    # Losses
    stego_loss = SteganoLoss()
    msg_loss = MessageLoss()
    freq_loss = FrequencyLoss()
    scaler = GradScaler()

    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        epoch_losses = {
            "g_total": 0,
            "d_total": 0,
            "img_quality": 0,
            "msg_recovery": 0,
            "adversarial": 0,
        }
        bit_accuracy = 0

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            cover, message = batch
            cover = cover.to(device)
            message = message.float().to(device)

            # ──── Discriminator step ────
            opt_d.zero_grad()

            with autocast():
                stego, decoded = model(cover, message)

                # Real vs Fake
                real_real, real_stego_real = model.discriminate(cover)
                fake_real, fake_stego_fake = model.discriminate(stego.detach())

                # WGAN-GP loss
                d_loss = WGANGPLoss.discriminator_loss(real_real, fake_real)

                # Gradient penalty (wrapper for dual-output discriminator)
                def disc_forward(x):
                    real_score, _ = model.discriminator(x)
                    return real_score

                gp = WGANGPLoss.gradient_penalty(
                    disc_forward, cover, stego.detach(), lambda_gp=GAN_TRAINING.lambda_gp
                )

                d_total = d_loss + gp

            scaler.scale(d_total).backward()
            scaler.unscale_(opt_d)
            torch.nn.utils.clip_grad_norm_(model.discriminator.parameters(), 1.0)
            scaler.step(opt_d)

            # ──── Generator step ────
            opt_g.zero_grad()

            with autocast():
                stego, decoded = model(cover, message)

                # Adversarial
                fake_real, fake_stego = model.discriminate(stego)
                g_loss = -fake_real.mean()  # WGAN: maximize real score

                # Message recovery
                msg_loss_val = F.binary_cross_entropy_with_logits(decoded, message)

                # Image quality
                img_loss = F.mse_loss(stego, cover)

                # Frequency domain
                freq_loss_val = freq_loss(stego, cover)

                # Combined
                g_total = (
                    GAN_TRAINING.lambda_image * img_loss
                    + GAN_TRAINING.lambda_message * msg_loss_val
                    + GAN_TRAINING.lambda_adversarial * g_loss
                    + GAN_TRAINING.lambda_frequency * freq_loss_val
                )

            scaler.scale(g_total).backward()
            scaler.unscale_(opt_g)
            torch.nn.utils.clip_grad_norm_(model.generator.parameters(), 1.0)
            scaler.step(opt_g)
            scaler.update()

            # ──── Metrics ────
            with torch.no_grad():
                bit_acc = ((torch.sigmoid(decoded) > 0.5) == message.bool()).float().mean()
                bit_accuracy += bit_acc.item()

                epoch_losses["g_total"] += g_total.item()
                epoch_losses["d_total"] += d_total.item()
                epoch_losses["img_quality"] += img_loss.item()
                epoch_losses["msg_recovery"] += msg_loss_val.item()
                epoch_losses["adversarial"] += g_loss.item()

        # ──── Logging ────
        n_batches = len(train_loader)
        for k, v in epoch_losses.items():
            epoch_losses[k] = v / n_batches
        bit_accuracy /= n_batches

        print(
            f"Epoch {epoch+1}: G_loss={epoch_losses['g_total']:.4f}, "
            f"D_loss={epoch_losses['d_total']:.4f}, BER={1-bit_accuracy:.4f}"
        )

        if HAS_WANDB:
            wandb.log({f"train/{k}": v for k, v in epoch_losses.items()})
            wandb.log({"train/bit_accuracy": bit_accuracy})

        # ──── Checkpointing ────
        if epoch_losses["g_total"] < best_loss:
            best_loss = epoch_losses["g_total"]
            torch.save(model.state_dict(), ckpt_dir / "best_model.pth")

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), ckpt_dir / f"epoch_{epoch+1}.pth")

        sched_g.step(epoch + batch_idx / n_batches)
        sched_d.step(epoch + batch_idx / n_batches)

    print(f"Training complete. Best checkpoint saved to {ckpt_dir}")


# ──────────────────── Audio GAN Training ────────────────────


def train_audio_gan(
    model,
    train_loader: DataLoader,
    epochs: int = 100,
    device: str = "cuda",
    experiment_name: str = "audio_gan",
):
    """
    Train Audio GAN (spectrogram domain).

    Args:
        model: AudioGANSteganography instance
        train_loader: DataLoader yielding (magnitude, phase, message) tuples
        epochs: Number of epochs
        device: Device to use
        experiment_name: Name for logging
    """
    if HAS_WANDB:
        try:
            wandb.init(project="steganography", name=f"{experiment_name}", mode="offline")
        except Exception:
            pass

    model = model.to(device)
    ckpt_dir = _create_checkpoint_dir(experiment_name)

    # Optimizers
    opt_g = torch.optim.AdamW(
        model.generator.parameters(),
        lr=GAN_TRAINING.learning_rate_g,
        weight_decay=GAN_TRAINING.weight_decay,
    )
    opt_d = torch.optim.AdamW(
        model.discriminator.parameters(),
        lr=GAN_TRAINING.learning_rate_d,
        weight_decay=GAN_TRAINING.weight_decay,
    )

    sched_g = CosineAnnealingWarmRestarts(opt_g, T_0=10, T_mult=2)
    sched_d = CosineAnnealingWarmRestarts(opt_d, T_0=10, T_mult=2)

    scaler = GradScaler()
    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        epoch_losses = {
            "g_total": 0,
            "d_total": 0,
            "magnitude_loss": 0,
            "msg_loss": 0,
        }
        bit_accuracy = 0

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            magnitude, phase, message = batch
            magnitude = magnitude.to(device)
            phase = phase.to(device)
            message = message.float().to(device)

            # ──── Discriminator step ────
            opt_d.zero_grad()

            with autocast():
                stego_mag, decoded = model(magnitude, phase, message)

                real_real, real_stego = model.discriminate(magnitude)
                fake_real, fake_stego = model.discriminate(stego_mag.detach())

                d_loss = WGANGPLoss.discriminator_loss(real_real, fake_real)

                # Gradient penalty (wrapper for dual-output discriminator)
                def disc_forward(x):
                    real_score, _ = model.discriminator(x)
                    return real_score

                gp = WGANGPLoss.gradient_penalty(
                    disc_forward, magnitude, stego_mag.detach(), lambda_gp=GAN_TRAINING.lambda_gp
                )

                d_total = d_loss + gp

            scaler.scale(d_total).backward()
            scaler.unscale_(opt_d)
            torch.nn.utils.clip_grad_norm_(model.discriminator.parameters(), 1.0)
            scaler.step(opt_d)

            # ──── Generator step ────
            opt_g.zero_grad()

            with autocast():
                stego_mag, decoded = model(magnitude, phase, message)

                fake_real, fake_stego = model.discriminate(stego_mag)
                g_loss = -fake_real.mean()

                mag_loss = F.mse_loss(stego_mag, magnitude)
                msg_loss_val = F.binary_cross_entropy_with_logits(decoded, message)

                g_total = (
                    GAN_TRAINING.lambda_image * mag_loss
                    + GAN_TRAINING.lambda_message * msg_loss_val
                    + GAN_TRAINING.lambda_adversarial * g_loss
                )

            scaler.scale(g_total).backward()
            scaler.unscale_(opt_g)
            torch.nn.utils.clip_grad_norm_(model.generator.parameters(), 1.0)
            scaler.step(opt_g)
            scaler.update()

            # Metrics
            with torch.no_grad():
                bit_acc = ((torch.sigmoid(decoded) > 0.5) == message.bool()).float().mean()
                bit_accuracy += bit_acc.item()
                epoch_losses["g_total"] += g_total.item()
                epoch_losses["d_total"] += d_total.item()
                epoch_losses["magnitude_loss"] += mag_loss.item()
                epoch_losses["msg_loss"] += msg_loss_val.item()

        # Logging
        n_batches = len(train_loader)
        for k, v in epoch_losses.items():
            epoch_losses[k] = v / n_batches
        bit_accuracy /= n_batches

        print(
            f"Epoch {epoch+1}: G_loss={epoch_losses['g_total']:.4f}, "
            f"D_loss={epoch_losses['d_total']:.4f}, BER={1-bit_accuracy:.4f}"
        )

        if HAS_WANDB:
            wandb.log({f"train/{k}": v for k, v in epoch_losses.items()})

        # Checkpointing
        if epoch_losses["g_total"] < best_loss:
            best_loss = epoch_losses["g_total"]
            torch.save(model.state_dict(), ckpt_dir / "best_model.pth")

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), ckpt_dir / f"epoch_{epoch+1}.pth")

        sched_g.step(epoch + batch_idx / n_batches)
        sched_d.step(epoch + batch_idx / n_batches)


# ──────────────────── Video GAN Training ────────────────────


def train_video_gan(
    model,
    train_loader: DataLoader,
    epochs: int = 100,
    device: str = "cuda",
    experiment_name: str = "video_gan",
):
    """
    Train Video GAN (spatio-temporal).

    Args:
        model: VideoGANSteganography instance
        train_loader: DataLoader yielding (frames, message, flow) tuples
        epochs: Number of epochs
        device: Device to use
        experiment_name: Name for logging
    """
    if HAS_WANDB:
        try:
            wandb.init(project="steganography", name=f"{experiment_name}", mode="offline")
        except Exception:
            pass

    model = model.to(device)
    ckpt_dir = _create_checkpoint_dir(experiment_name)

    # Optimizers
    opt_g = torch.optim.AdamW(
        model.generator.parameters(),
        lr=GAN_TRAINING.learning_rate_g,
        weight_decay=GAN_TRAINING.weight_decay,
    )
    opt_d = torch.optim.AdamW(
        model.discriminator.parameters(),
        lr=GAN_TRAINING.learning_rate_d,
        weight_decay=GAN_TRAINING.weight_decay,
    )

    sched_g = CosineAnnealingWarmRestarts(opt_g, T_0=10, T_mult=2)
    sched_d = CosineAnnealingWarmRestarts(opt_d, T_0=10, T_mult=2)

    scaler = GradScaler()
    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        epoch_losses = {
            "g_total": 0,
            "d_total": 0,
            "frame_loss": 0,
            "msg_loss": 0,
            "temporal_loss": 0,
        }
        bit_accuracy = 0

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            if len(batch) == 3:
                frames, message, flow = batch
            else:
                frames, message = batch
                flow = None

            frames = frames.to(device)
            message = message.float().to(device)
            if flow is not None:
                flow = flow.to(device)

            # ──── Discriminator step ────
            opt_d.zero_grad()

            with autocast():
                stego_frames, decoded = model(frames, message, flow)

                real_real, real_temporal = model.discriminate(frames)
                fake_real, fake_temporal = model.discriminate(stego_frames.detach())

                d_loss = WGANGPLoss.discriminator_loss(real_real, fake_real)

                # Gradient penalty (wrapper for dual-output discriminator)
                def disc_forward(x):
                    real_score, _ = model.discriminator(x)
                    return real_score

                gp = WGANGPLoss.gradient_penalty(
                    disc_forward, frames, stego_frames.detach(), lambda_gp=GAN_TRAINING.lambda_gp
                )

                d_total = d_loss + gp

            scaler.scale(d_total).backward()
            scaler.unscale_(opt_d)
            torch.nn.utils.clip_grad_norm_(model.discriminator.parameters(), 1.0)
            scaler.step(opt_d)

            # ──── Generator step ────
            opt_g.zero_grad()

            with autocast():
                stego_frames, decoded = model(frames, message, flow)

                fake_real, fake_temporal = model.discriminate(stego_frames)
                g_loss = -fake_real.mean()

                frame_loss = F.mse_loss(stego_frames, frames)
                msg_loss_val = F.binary_cross_entropy_with_logits(decoded, message)
                temporal_loss = -fake_temporal.mean()  # Maximize temporal smoothness

                g_total = (
                    GAN_TRAINING.lambda_image * frame_loss
                    + GAN_TRAINING.lambda_message * msg_loss_val
                    + GAN_TRAINING.lambda_adversarial * g_loss
                    + VIDEO_GAN.lambda_temporal * temporal_loss
                )

            scaler.scale(g_total).backward()
            scaler.unscale_(opt_g)
            torch.nn.utils.clip_grad_norm_(model.generator.parameters(), 1.0)
            scaler.step(opt_g)
            scaler.update()

            # Metrics
            with torch.no_grad():
                bit_acc = ((torch.sigmoid(decoded) > 0.5) == message.bool()).float().mean()
                bit_accuracy += bit_acc.item()
                epoch_losses["g_total"] += g_total.item()
                epoch_losses["d_total"] += d_total.item()
                epoch_losses["frame_loss"] += frame_loss.item()
                epoch_losses["msg_loss"] += msg_loss_val.item()
                epoch_losses["temporal_loss"] += temporal_loss.item()

        # Logging
        n_batches = len(train_loader)
        for k, v in epoch_losses.items():
            epoch_losses[k] = v / n_batches
        bit_accuracy /= n_batches

        print(
            f"Epoch {epoch+1}: G_loss={epoch_losses['g_total']:.4f}, "
            f"D_loss={epoch_losses['d_total']:.4f}, BER={1-bit_accuracy:.4f}"
        )

        if HAS_WANDB:
            wandb.log({f"train/{k}": v for k, v in epoch_losses.items()})

        # Checkpointing
        if epoch_losses["g_total"] < best_loss:
            best_loss = epoch_losses["g_total"]
            torch.save(model.state_dict(), ckpt_dir / "best_model.pth")

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), ckpt_dir / f"epoch_{epoch+1}.pth")

        sched_g.step(epoch + batch_idx / n_batches)
        sched_d.step(epoch + batch_idx / n_batches)

    print(f"Training complete. Best checkpoint saved to {ckpt_dir}")
