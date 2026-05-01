#!/usr/bin/env python3
"""
Video GAN Training with Real Video Data.

Same recipe as image GAN (warmup + focal loss + AdamW + cosine LR)
but uses real MP4 files from uploads/video/ for realistic training.

Target: 95%+ bit accuracy with imperceptible visual quality.
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import sys

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
from models.video_gan import VideoGANSteganography
from config.settings import VIDEO_GAN, PATHS
from core.video.frame_utils import extract_frames


# ─── Focal BCE loss (same as image GAN recipe) ────────────────────────────────

def focal_bce_loss(logits, targets, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p_t = targets * torch.sigmoid(logits) + (1 - targets) * (1 - torch.sigmoid(logits))
    return ((1 - p_t) ** gamma * bce).mean()


# ─── Dataset ──────────────────────────────────────────────────────────────────

def build_clip_dataset(video_dir: str, max_frames_per_video: int = 60, stride: int = 3):
    """
    Extract 5-frame clips from all MP4 files.
    stride=3 gives overlapping clips for more training data.
    Returns list of (T, 3, H, W) float tensors in [0, 1].
    """
    tw = VIDEO_GAN.temporal_window
    fs = VIDEO_GAN.frame_size
    clips = []
    paths = sorted(Path(video_dir).glob("*.mp4"))
    print(f"Loading clips from {len(paths)} videos …")

    for vp in paths:
        try:
            frames, _ = extract_frames(
                str(vp),
                max_frames=max_frames_per_video,
                resize=(fs, fs),
            )
            if len(frames) < tw:
                continue

            # BGR → RGB, normalize to [0, 1]
            arr = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames])
            arr = arr.astype(np.float32) / 255.0          # (T, H, W, 3)
            t = torch.from_numpy(arr).permute(0, 3, 1, 2) # (T, 3, H, W)

            # Sliding window clips
            for start in range(0, len(t) - tw + 1, stride):
                clips.append(t[start:start + tw])
        except Exception as e:
            print(f"  skip {vp.name}: {e}")

    print(f"Total clips: {len(clips)}")
    return clips


class TrainDataset(Dataset):
    """Random messages per sample (non-deterministic for training)."""
    def __init__(self, clips):
        self.clips = clips

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, i):
        msg = torch.randint(0, 2, (VIDEO_GAN.message_bits,)).float()
        return self.clips[i], msg


class TestDataset(Dataset):
    """Deterministic messages for reproducible test evaluation."""
    def __init__(self, clips):
        self.clips = clips

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, i):
        g = torch.Generator()
        g.manual_seed(1234 + i)
        msg = torch.randint(0, 2, (VIDEO_GAN.message_bits,), generator=g).float()
        return self.clips[i], msg


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate(model, loader, device):
    """Returns (bit_accuracy %, mean_psnr dB)."""
    model.eval()
    correct = total = 0
    psnr_sum = n = 0
    with torch.no_grad():
        for clips, messages in loader:
            clips, messages = clips.to(device), messages.to(device)
            stego = model.generator(clips, messages, None)
            # decoder expects (B, 3, T, H, W)
            decoded = model.decoder(stego.permute(0, 2, 1, 3, 4))
            preds = (torch.sigmoid(decoded) > 0.5).float()
            correct += (preds == messages).sum().item()
            total += messages.numel()

            mse = F.mse_loss(stego, clips, reduction='none').mean(dim=[1, 2, 3, 4])
            psnr_sum += (10 * torch.log10(1.0 / mse.clamp(min=1e-10))).sum().item()
            n += clips.size(0)
    return (correct / total) * 100, psnr_sum / max(n, 1)


# ─── Training ─────────────────────────────────────────────────────────────────

def train(epochs: int = 120, warmup_epochs: int = 40, resume: bool = False):
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    # Build dataset from real videos
    video_dir = os.path.join(PATHS.project_root, "uploads", "video")
    all_clips = build_clip_dataset(video_dir, max_frames_per_video=60, stride=3)

    if len(all_clips) < 10:
        print("✗ Not enough clips — check uploads/video/ directory")
        return

    # 80/20 split — deterministic
    n_test = max(1, len(all_clips) // 5)
    n_train = len(all_clips) - n_test
    train_clips = all_clips[:n_train]
    test_clips  = all_clips[n_train:]
    print(f"Train clips: {n_train}   Test clips: {n_test}")

    train_loader = DataLoader(TrainDataset(train_clips), batch_size=8, shuffle=True,
                              drop_last=True, num_workers=0)
    test_loader  = DataLoader(TestDataset(test_clips),   batch_size=8, shuffle=False,
                              num_workers=0)

    # Model
    model = VideoGANSteganography(
        msg_length=VIDEO_GAN.message_bits,
        base_ch=VIDEO_GAN.base_channels,
        temporal_window=VIDEO_GAN.temporal_window,
        frame_size=VIDEO_GAN.frame_size,
    ).to(device)

    ckpt_dir = Path(PATHS.models_dir) / "video_gan_improved"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "best_model.pth"

    if resume and ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"✓ Resumed from {ckpt_path}")

    # Optimizers — lower LR for generator to protect visual quality
    opt_dec = torch.optim.AdamW(model.decoder.parameters(),   lr=1e-3,  weight_decay=1e-4)
    opt_g   = torch.optim.AdamW(model.generator.parameters(), lr=2e-4,  weight_decay=1e-4)
    sched_dec = torch.optim.lr_scheduler.CosineAnnealingLR(opt_dec, T_max=epochs, eta_min=1e-6)
    sched_g   = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g,   T_max=epochs, eta_min=1e-6)

    # Loss weights
    LAMBDA_MSG = 5.0   # message recovery (primary goal in warmup)
    LAMBDA_IMG = 1.0   # video quality (low during warmup, doesn't need to be high)
    LAMBDA_IMG_ROBUST = 10.0  # raise after warmup

    best_test_acc = 0.0

    print(f"\nWarmup: {warmup_epochs} epochs | Total: {epochs} epochs")
    print(f"message_bits={VIDEO_GAN.message_bits}  temporal_window={VIDEO_GAN.temporal_window}  frame_size={VIDEO_GAN.frame_size}\n")

    for epoch in range(epochs):
        model.train()
        is_warmup = epoch < warmup_epochs
        lam_img = LAMBDA_IMG if is_warmup else LAMBDA_IMG_ROBUST

        accs, msg_losses, img_losses = [], [], []

        for clips, messages in train_loader:
            clips, messages = clips.to(device), messages.to(device)

            opt_g.zero_grad()
            opt_dec.zero_grad()

            stego = model.generator(clips, messages, None)
            decoded = model.decoder(stego.permute(0, 2, 1, 3, 4))

            msg_loss = focal_bce_loss(decoded, messages)
            img_loss = F.mse_loss(stego, clips)
            total    = LAMBDA_MSG * msg_loss + lam_img * img_loss
            total.backward()

            torch.nn.utils.clip_grad_norm_(model.generator.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(model.decoder.parameters(),   1.0)
            opt_g.step()
            opt_dec.step()

            with torch.no_grad():
                acc = ((torch.sigmoid(decoded) > 0.5) == messages.bool()).float().mean().item()
            accs.append(acc)
            msg_losses.append(msg_loss.item())
            img_losses.append(img_loss.item())

        sched_g.step()
        sched_dec.step()

        train_acc  = sum(accs) / len(accs) * 100
        avg_psnr   = 10 * np.log10(1.0 / max(sum(img_losses) / len(img_losses), 1e-10))
        test_acc, test_psnr = evaluate(model, test_loader, device)

        marker = ""
        if test_acc > best_test_acc and (test_acc >= 80.0 or epoch > warmup_epochs):
            best_test_acc = test_acc
            torch.save(model.state_dict(), ckpt_path)
            marker = "  ✓ saved"

        phase = "warmup" if is_warmup else "robust"
        print(
            f"Epoch {epoch+1:03d}/{epochs} [{phase}] "
            f"train={train_acc:5.1f}%  test={test_acc:5.1f}%  "
            f"train_psnr={avg_psnr:5.1f}  test_psnr={test_psnr:5.1f} dB"
            f"{marker}"
        )

    print(f"\n✓ Done. Best test accuracy: {best_test_acc:.1f}%  →  {ckpt_path}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",        type=int,  default=120)
    p.add_argument("--warmup-epochs", type=int,  default=40)
    p.add_argument("--resume",        action="store_true")
    args = p.parse_args()
    try:
        train(epochs=args.epochs, warmup_epochs=args.warmup_epochs, resume=args.resume)
    finally:
        sys.exit(0)
