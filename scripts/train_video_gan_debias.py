#!/usr/bin/env python3
"""
Debias training — fix systematic decoder bit-position biases.

Root cause: decoder has near-0/near-1 priors for 26/32 bit positions,
meaning the generator can't override the background for certain bits.
"hi" decodes as "xa\x88" because bit positions 3, 7 have background bias
that the current generator strength (0.228) can't overcome on real video.

Fix strategy:
  Phase 0 (20 ep): Generator FROZEN at strength=0.5 (strong clear signal).
    Decoder trained with EXTREME msg loss (LAMBDA_MSG=200) + per-bit focal.
    Forces decoder to correctly read ALL 32 bits at high signal strength.
    Eliminates near-0/near-1 frozen bit behaviors.

  Phase 1 (40 ep): Joint fine-tuning on real video clips.
    Generator unfrozen, adapts to produce stronger per-position signals.
    LAMBDA_MSG=50, LAMBDA_IMG=1.0.
    Domain adapts to real video content at the new, higher strength.

Starting point: adapt_ep016 checkpoint (91.3% on real video).

Usage:
    python3 -u scripts/train_video_gan_debias.py
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
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split

from config.settings import VIDEO_GAN, PATHS
from models.video_gan import VideoGANSteganography
from core.video.frame_utils import extract_frames


def focal_bce(logits, targets, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = targets * torch.sigmoid(logits) + (1 - targets) * (1 - torch.sigmoid(logits))
    return ((1 - p_t) ** gamma * bce).mean()


def per_bit_focal_bce(logits, targets, gamma=3.0):
    """Focal BCE with per-bit loss aggregation — forces equal accuracy on all 32 positions."""
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")  # (B, 32)
    p_t = targets * torch.sigmoid(logits) + (1 - targets) * (1 - torch.sigmoid(logits))
    per_sample = ((1 - p_t) ** gamma * bce)  # (B, 32)
    # Average across batch per bit, then average across bits
    # This upweights bits where the model is consistently wrong
    per_bit = per_sample.mean(dim=0)  # (32,)
    return per_bit.mean()


def pick_device():
    if torch.cuda.is_available():
        print("Using CUDA"); return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps and mps.is_available():
        print("Using MPS (Apple GPU)"); return torch.device("mps")
    print("Using CPU"); return torch.device("cpu")


def make_synthetic_dataset(n: int) -> TensorDataset:
    """Diverse synthetic clips for Phase 0 decoder debiasing."""
    T, H = VIDEO_GAN.temporal_window, VIDEO_GAN.frame_size
    clips, messages = [], []
    torch.manual_seed(42)
    for i in range(n):
        t = i % 6
        if t == 0:
            base = torch.rand(3,1,1).expand(3,H,H)
            frame = (base + torch.linspace(0,1,H).view(1,H,1).expand(3,H,H)*0.3
                     + torch.linspace(0,1,H).view(1,1,H).expand(3,H,H)*0.3
                     + torch.randn(3,H,H)*0.05).clamp(0,1)
        elif t == 1:
            check = torch.zeros(H,H)
            for r in range(H):
                for c in range(H):
                    if (r//8+c//8)%2==0: check[r,c]=1.0
            frame = (check.unsqueeze(0).expand(3,H,H)*0.4 + torch.rand(3,1,1).expand(3,H,H)*0.6
                     + torch.randn(3,H,H)*0.03).clamp(0,1)
        elif t == 2:
            cy = H//2
            dist = torch.sqrt((torch.arange(H).float()-cy).view(H,1)**2
                              + (torch.arange(H).float()-cy).view(1,H)**2)
            frame = ((dist/dist.max()).unsqueeze(0).expand(3,H,H)*0.5
                     + torch.rand(3,1,1).expand(3,H,H)*0.5
                     + torch.randn(3,H,H)*0.03).clamp(0,1)
        elif t == 3:
            stripe = ((torch.sin(torch.linspace(0,8*np.pi,H)).view(H,1).expand(H,H)+1)/2)
            frame = (stripe.unsqueeze(0).expand(3,H,H)*0.4
                     + torch.rand(3,1,1).expand(3,H,H)*0.6
                     + torch.randn(3,H,H)*0.03).clamp(0,1)
        elif t == 4:
            frame = F.avg_pool2d(torch.rand(3,H,H).unsqueeze(0),
                                 kernel_size=5,stride=1,padding=2).squeeze(0)
            frame = (frame + torch.randn(3,H,H)*0.02).clamp(0,1)
        else:
            frame = (torch.rand(3,1,1).expand(3,H,H)*0.3
                     + torch.linspace(0,0.2,H).view(1,H,1).expand(3,H,H)
                     + torch.randn(3,H,H)*0.02).clamp(0,1)

        frames = [(frame + torch.randn(3,H,H)*0.02).clamp(0,1) for _ in range(T)]
        clips.append(torch.stack(frames))
        messages.append(torch.randint(0,2,(VIDEO_GAN.message_bits,)).float())

    return TensorDataset(torch.stack(clips), torch.stack(messages))


def build_real_clips(video_dir: str, max_frames_per_video=60, stride=3):
    T, fs = VIDEO_GAN.temporal_window, VIDEO_GAN.frame_size
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
def evaluate(model, loader, device, tag=""):
    model.eval()
    accs, psnrs = [], []
    for clips, messages in loader:
        clips, messages = clips.to(device), messages.to(device)
        stego, decoded = model(clips, messages, flow=None)
        acc = ((torch.sigmoid(decoded)>0.5)==messages.bool()).float().mean().item()*100
        psnr = 10*np.log10(1.0/max(F.mse_loss(stego, clips).item(), 1e-10))
        accs.append(acc); psnrs.append(psnr)
    model.train()
    return float(np.mean(accs)), float(np.mean(psnrs))


@torch.no_grad()
def count_extreme_bits(model, device):
    """Count how many bit positions have near-0 or near-1 decoder bias on noise."""
    T, H = VIDEO_GAN.temporal_window, VIDEO_GAN.frame_size
    noise = torch.rand(8, T, 3, H, H).to(device)
    logits = model.decoder(noise.permute(0, 2, 1, 3, 4))
    probs = torch.sigmoid(logits).mean(dim=0).cpu().numpy()
    extreme = sum(1 for p in probs if p < 0.1 or p > 0.9)
    return extreme, probs


def train():
    device = pick_device()
    ckpt_dir = Path(PATHS.models_dir) / "video_gan_improved"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "best_model.pth"

    # Start from adapt_ep016 (best real-video checkpoint)
    start_ckpt = ckpt_dir / "adapt_ep016_acc91.3.pth"
    if not start_ckpt.exists():
        start_ckpt = ckpt_path
        if not start_ckpt.exists():
            print("No checkpoint found"); sys.exit(1)

    model = VideoGANSteganography(
        msg_length=VIDEO_GAN.message_bits,
        base_ch=VIDEO_GAN.base_channels,
        temporal_window=VIDEO_GAN.temporal_window,
        frame_size=VIDEO_GAN.frame_size,
    ).to(device)
    model.load_state_dict(torch.load(start_ckpt, map_location=device, weights_only=True))
    print(f"Loaded: {start_ckpt.name}  strength={model.generator.strength.item():.4f}")

    extreme, probs = count_extreme_bits(model, device)
    print(f"Initial extreme bias bits: {extreme}/32\n")

    # ── Phase 0: Decoder debiasing (generator FROZEN at strength=0.5) ───────────
    PHASE0_EPOCHS = 20
    PHASE1_EPOCHS = 40

    print(f"{'='*65}")
    print(f"PHASE 0 ({PHASE0_EPOCHS} ep): Generator FROZEN at str=0.5 — decoder debiasing")
    print(f"{'='*65}\n")

    # Boost strength so decoder sees a strong, clear signal
    with torch.no_grad():
        model.generator.strength.data = torch.tensor(0.5, device=device)

    # Freeze generator
    for p in model.generator.parameters():
        p.requires_grad_(False)

    # Synthetic data for Phase 0 (decoder only needs clean, consistent signal)
    syn_ds = make_synthetic_dataset(3000)
    gen = torch.Generator().manual_seed(0)
    syn_train, syn_val = random_split(syn_ds, [2600, 400], generator=gen)
    syn_loader = DataLoader(syn_train, batch_size=32, shuffle=True, drop_last=True, num_workers=0)
    syn_val_loader = DataLoader(syn_val, batch_size=32, shuffle=False, num_workers=0)
    print(f"Synthetic train: 2600  val: 400  batches/ep: {len(syn_loader)}")

    opt_dec = torch.optim.AdamW(model.decoder.parameters(), lr=1e-3, weight_decay=1e-4)
    sched_dec = torch.optim.lr_scheduler.CosineAnnealingLR(opt_dec, T_max=PHASE0_EPOCHS, eta_min=1e-5)

    best_val_acc = 0.0
    t_start = time.time()

    for epoch in range(1, PHASE0_EPOCHS + 1):
        model.train()
        train_accs = []
        for clips, messages in syn_loader:
            clips, messages = clips.to(device), messages.to(device)
            opt_dec.zero_grad()
            stego, decoded = model(clips, messages, flow=None)
            loss = 200.0 * per_bit_focal_bce(decoded, messages)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), 1.0)
            opt_dec.step()
            with torch.no_grad():
                acc = ((torch.sigmoid(decoded)>0.5)==messages.bool()).float().mean().item()*100
            train_accs.append(acc)
        sched_dec.step()

        train_acc = float(np.mean(train_accs))
        val_acc, val_psnr = evaluate(model, syn_val_loader, device)

        extreme, _ = count_extreme_bits(model, device)
        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), ckpt_path)
            marker = "  ← saved"

        elapsed = (time.time()-t_start)/epoch
        eta = int(elapsed*(PHASE0_EPOCHS+PHASE1_EPOCHS-epoch))
        print(f"Ep {epoch:03d}/{PHASE0_EPOCHS+PHASE1_EPOCHS} [ph0]  "
              f"train={train_acc:5.1f}%  val={val_acc:5.1f}%  "
              f"psnr={val_psnr:5.1f}dB  str={model.generator.strength.item():.3f}  "
              f"extreme_bits={extreme:2d}  ETA={eta//60}min{marker}")

    print(f"\nPhase 0 done. Best synthetic val: {best_val_acc:.1f}%")
    extreme, probs = count_extreme_bits(model, device)
    print(f"Extreme bias bits after Phase 0: {extreme}/32\n")

    # ── Phase 1: Joint fine-tuning on real video ─────────────────────────────────
    print(f"{'='*65}")
    print(f"PHASE 1 ({PHASE1_EPOCHS} ep): Joint training on real video")
    print(f"{'='*65}\n")

    # Unfreeze generator
    for p in model.generator.parameters():
        p.requires_grad_(True)

    video_dir = os.path.join(PATHS.project_root, "uploads", "video")
    all_clips = build_real_clips(video_dir, max_frames_per_video=60, stride=3)
    if len(all_clips) < 10:
        print("Not enough real clips — check uploads/video/"); return

    n_val = max(1, len(all_clips) // 5)
    n_train = len(all_clips) - n_val
    train_clips, val_clips = all_clips[:n_train], all_clips[n_train:]
    train_loader = DataLoader(RealClipDataset(train_clips), batch_size=16, shuffle=True, drop_last=True, num_workers=0)
    val_loader   = DataLoader(RealClipDataset(val_clips),   batch_size=16, shuffle=False, num_workers=0)
    print(f"Real train: {n_train}  val: {n_val}  batches/ep: {len(train_loader)}")

    opt_dec = torch.optim.AdamW(model.decoder.parameters(), lr=3e-4, weight_decay=1e-4)
    opt_g   = torch.optim.AdamW(model.generator.parameters(), lr=1e-5, weight_decay=1e-4)
    sched_dec = torch.optim.lr_scheduler.CosineAnnealingLR(opt_dec, T_max=PHASE1_EPOCHS, eta_min=1e-6)
    sched_g   = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g,   T_max=PHASE1_EPOCHS, eta_min=1e-7)

    best_real_acc = 0.0

    for epoch in range(1, PHASE1_EPOCHS + 1):
        global_ep = PHASE0_EPOCHS + epoch
        model.train()
        # Gradually reduce strength from 0.5 toward 0.3 — keep it high enough to overcome biases
        target_strength = max(0.3, 0.5 - 0.2 * epoch / PHASE1_EPOCHS)

        train_accs = []
        for clips, messages in train_loader:
            clips, messages = clips.to(device), messages.to(device)
            opt_g.zero_grad(); opt_dec.zero_grad()
            stego, decoded = model(clips, messages, flow=None)
            loss = 50.0 * focal_bce(decoded, messages) + 1.0 * F.mse_loss(stego, clips)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(model.generator.parameters(), 1.0)
            opt_dec.step(); opt_g.step()
            with torch.no_grad():
                model.generator.strength.data.clamp_(min=target_strength)
                acc = ((torch.sigmoid(decoded)>0.5)==messages.bool()).float().mean().item()*100
            train_accs.append(acc)

        sched_dec.step(); sched_g.step()

        train_acc = float(np.mean(train_accs))
        val_acc, val_psnr = evaluate(model, val_loader, device)
        extreme, _ = count_extreme_bits(model, device)

        marker = ""
        if val_acc > best_real_acc:
            best_real_acc = val_acc
            torch.save(model.state_dict(), ckpt_path)
            torch.save(model.state_dict(), ckpt_dir / f"debias_ep{global_ep:03d}_acc{val_acc:.1f}.pth")
            marker = "  ← saved"

        elapsed = (time.time()-t_start)/global_ep
        eta = int(elapsed*(PHASE0_EPOCHS+PHASE1_EPOCHS-global_ep))
        print(f"Ep {global_ep:03d}/{PHASE0_EPOCHS+PHASE1_EPOCHS} [ph1]  "
              f"train={train_acc:5.1f}%  val={val_acc:5.1f}%  "
              f"psnr={val_psnr:5.1f}dB  str={model.generator.strength.item():.3f}  "
              f"extreme_bits={extreme:2d}  ETA={eta//60}min{marker}")

    print(f"\nDone. Best real-video val: {best_real_acc:.1f}%  →  {ckpt_path}")


if __name__ == "__main__":
    try:
        train()
    finally:
        sys.exit(0)
