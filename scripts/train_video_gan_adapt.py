#!/usr/bin/env python3
"""
Domain Adaptation — fine-tune the synthetic-trained model on real video clips.

Problem: model achieves 90.2% on synthetic data but only ~60% on real video.
Root cause: synthetic textures (gradients, checkerboards) ≠ real video content.

Fix: fine-tune the checkpoint on real MP4 clips from uploads/video/ using
very low LR (no catastrophic forgetting) for 50 epochs.

Usage (from project root):
    python3 -u scripts/train_video_gan_adapt.py
"""

from __future__ import annotations

import os, sys, time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset

from config.settings import VIDEO_GAN, PATHS
from models.video_gan import VideoGANSteganography
from core.video.frame_utils import extract_frames


def focal_bce(logits, targets, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = targets * torch.sigmoid(logits) + (1 - targets) * (1 - torch.sigmoid(logits))
    return ((1 - p_t) ** gamma * bce).mean()


def pick_device():
    if torch.cuda.is_available():
        print("Using CUDA"); return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps and mps.is_available():
        print("Using MPS (Apple GPU)"); return torch.device("mps")
    print("Using CPU"); return torch.device("cpu")


def build_real_clips(video_dir: str, max_frames_per_video=60, stride=3):
    T = VIDEO_GAN.temporal_window
    fs = VIDEO_GAN.frame_size
    clips = []
    paths = sorted(Path(video_dir).glob("*.mp4"))
    print(f"Loading clips from {len(paths)} real videos…")
    for vp in paths:
        try:
            frames, _ = extract_frames(str(vp), max_frames=max_frames_per_video, resize=(fs, fs))
            if len(frames) < T:
                continue
            arr = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]).astype(np.float32) / 255.0
            t = torch.from_numpy(arr).permute(0, 3, 1, 2)
            for start in range(0, len(t) - T + 1, stride):
                clips.append(t[start:start + T])
        except Exception as e:
            print(f"  skip {vp.name}: {e}")
    print(f"Total real clips: {len(clips)}")
    return clips


class RealClipDataset(Dataset):
    def __init__(self, clips):
        self.clips = clips
    def __len__(self): return len(self.clips)
    def __getitem__(self, i):
        return self.clips[i], torch.randint(0, 2, (VIDEO_GAN.message_bits,)).float()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    accs, psnrs = [], []
    for clips, messages in loader:
        clips, messages = clips.to(device), messages.to(device)
        stego, decoded = model(clips, messages, flow=None)
        acc = ((torch.sigmoid(decoded) > 0.5) == messages.bool()).float().mean().item() * 100
        psnr = 10 * np.log10(1.0 / max(F.mse_loss(stego, clips).item(), 1e-10))
        accs.append(acc); psnrs.append(psnr)
    model.train()
    return float(np.mean(accs)), float(np.mean(psnrs))


def train(epochs=50):
    device = pick_device()

    video_dir = os.path.join(PATHS.project_root, "uploads", "video")
    all_clips = build_real_clips(video_dir, max_frames_per_video=60, stride=3)
    if len(all_clips) < 10:
        print("Not enough clips — check uploads/video/"); return

    n_val = max(1, len(all_clips) // 5)
    n_train = len(all_clips) - n_val
    train_clips, val_clips = all_clips[:n_train], all_clips[n_train:]
    print(f"Train clips: {n_train}   Val clips: {n_val}")

    train_loader = DataLoader(RealClipDataset(train_clips), batch_size=8, shuffle=True, drop_last=True, num_workers=0)
    val_loader   = DataLoader(RealClipDataset(val_clips),   batch_size=8, shuffle=False, num_workers=0)

    model = VideoGANSteganography(
        msg_length=VIDEO_GAN.message_bits, base_ch=VIDEO_GAN.base_channels,
        temporal_window=VIDEO_GAN.temporal_window, frame_size=VIDEO_GAN.frame_size,
    ).to(device)

    ckpt_dir = Path(PATHS.models_dir) / "video_gan_improved"
    ckpt_path = ckpt_dir / "best_model.pth"

    if not ckpt_path.exists():
        print("No synthetic checkpoint found — run train_video_gan_v2.py first"); return

    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    print(f"Loaded synthetic checkpoint. Strength: {model.generator.strength.item():.4f}")

    # Very low LR — preserve synthetic knowledge, adapt to real video distribution
    opt_dec = torch.optim.AdamW(model.decoder.parameters(), lr=5e-4, weight_decay=1e-4)
    opt_g   = torch.optim.AdamW(model.generator.parameters(), lr=2e-5, weight_decay=1e-4)
    sched_dec = torch.optim.lr_scheduler.CosineAnnealingLR(opt_dec, T_max=epochs, eta_min=1e-6)
    sched_g   = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g,   T_max=epochs, eta_min=1e-7)

    best_val_acc = 0.0
    t_start = time.time()
    print(f"\nDomain adaptation: {epochs} epochs on real video\n")

    for epoch in range(1, epochs + 1):
        model.train()
        # Keep message loss high throughout — accuracy is the priority
        LAMBDA_MSG = 20.0
        LAMBDA_IMG = 2.0

        train_accs = []
        for clips, messages in train_loader:
            clips, messages = clips.to(device), messages.to(device)
            opt_g.zero_grad(); opt_dec.zero_grad()
            stego, decoded = model(clips, messages, flow=None)
            loss = LAMBDA_MSG * focal_bce(decoded, messages) + LAMBDA_IMG * F.mse_loss(stego, clips)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(model.generator.parameters(), 1.0)
            opt_dec.step(); opt_g.step()
            with torch.no_grad():
                model.generator.strength.data.clamp_(min=0.05)
                acc = ((torch.sigmoid(decoded) > 0.5) == messages.bool()).float().mean().item() * 100
            train_accs.append(acc)

        sched_dec.step(); sched_g.step()

        train_acc = float(np.mean(train_accs))
        val_acc, val_psnr = evaluate(model, val_loader, device)

        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), ckpt_path)
            torch.save(model.state_dict(), ckpt_dir / f"adapt_ep{epoch:03d}_acc{val_acc:.1f}.pth")
            marker = "  ← saved"

        elapsed = (time.time() - t_start) / epoch
        eta_m = int(elapsed * (epochs - epoch)) // 60
        print(f"Ep {epoch:03d}/{epochs}  train={train_acc:5.1f}%  val={val_acc:5.1f}%  "
              f"psnr={val_psnr:5.1f}dB  str={model.generator.strength.item():.3f}  "
              f"ETA={eta_m}min{marker}")

    print(f"\nDone. Best real-video val accuracy: {best_val_acc:.1f}%  →  {ckpt_path}")


if __name__ == "__main__":
    try:
        train(epochs=50)
    finally:
        sys.exit(0)
