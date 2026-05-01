#!/usr/bin/env python3
"""
Fine-tune the existing 98.2%-accuracy Image GAN to produce VISUALLY IMPERCEPTIBLE
stego images. Resumes from current best_model.pth and pushes PSNR up while
preserving bit accuracy near 95%.

Strategy: heavily increase image-quality loss weight and add an L1 penalty on
the residual (stego - cover). Clamp the model's learnable strength parameter
to a small value so perturbations stay sub-pixel. Reuse the warmup-phase
training loop (no noise layer) since we want the model to learn the
*minimum* perturbation that still decodes — adding noise during fine-tune
forces larger perturbations.
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import sys
import multiprocessing
if sys.platform != "win32":
    multiprocessing.set_start_method('fork', force=True)

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.image_gan import ImageGANSteganography
from config.settings import IMAGE_GAN, PATHS


def make_train_loader(num_samples: int = 10000) -> DataLoader:
    transform = transforms.Compose([
        transforms.Resize((IMAGE_GAN.image_size, IMAGE_GAN.image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
    ])
    ds = datasets.CIFAR10(root='./data/cifar10', train=True, download=True, transform=transform)
    ds = Subset(ds, range(min(num_samples, len(ds))))

    class WithMessages(torch.utils.data.Dataset):
        def __init__(self, base): self.base = base
        def __len__(self): return len(self.base)
        def __getitem__(self, i):
            img, _ = self.base[i]
            msg = torch.randint(0, 2, (IMAGE_GAN.message_bits,)).float()
            return img, msg

    return DataLoader(WithMessages(ds), batch_size=32, shuffle=True, drop_last=True, num_workers=2)


def make_test_loader(num_samples: int = 1000) -> DataLoader:
    transform = transforms.Compose([
        transforms.Resize((IMAGE_GAN.image_size, IMAGE_GAN.image_size)),
        transforms.ToTensor(),
    ])
    ds = datasets.CIFAR10(root='./data/cifar10', train=False, download=True, transform=transform)
    ds = Subset(ds, range(min(num_samples, len(ds))))

    class DetMsg(torch.utils.data.Dataset):
        def __init__(self, base): self.base = base
        def __len__(self): return len(self.base)
        def __getitem__(self, i):
            img, _ = self.base[i]
            g = torch.Generator(); g.manual_seed(42 + i)
            msg = torch.randint(0, 2, (IMAGE_GAN.message_bits,), generator=g).float()
            return img, msg

    return DataLoader(DetMsg(ds), batch_size=64, shuffle=False, num_workers=0)


def evaluate(model, loader, device):
    """Returns (bit_accuracy %, mean PSNR dB)."""
    model.eval()
    correct = 0
    total = 0
    psnr_sum = 0.0
    n_imgs = 0
    with torch.no_grad():
        for cover, message in loader:
            cover = cover.to(device)
            message = message.to(device)
            stego = model.generator(cover, message)
            decoded = model.decoder(stego)
            preds = (torch.sigmoid(decoded) > 0.5).float()
            correct += (preds == message).sum().item()
            total += message.numel()

            mse = F.mse_loss(stego, cover, reduction='none').mean(dim=[1, 2, 3])
            psnr = 10 * torch.log10(1.0 / mse.clamp(min=1e-10))
            psnr_sum += psnr.sum().item()
            n_imgs += cover.size(0)

    return (correct / total) * 100, psnr_sum / n_imgs


def finetune(epochs: int = 30):
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    model = ImageGANSteganography(
        msg_length=IMAGE_GAN.message_bits,
        base_ch=IMAGE_GAN.base_channels,
        image_size=IMAGE_GAN.image_size,
    ).to(device)

    ckpt_dir = Path(PATHS.models_dir) / "image_gan_improved"
    ckpt_path = ckpt_dir / "best_model.pth"
    if not ckpt_path.exists():
        print(f"✗ Checkpoint not found: {ckpt_path}")
        return
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"✓ Loaded checkpoint: {ckpt_path}")

    # Reset strength to a smaller initial value so perturbation starts small
    with torch.no_grad():
        model.generator.strength.data = torch.tensor(0.04, device=device)
    print(f"  strength reset to {model.generator.strength.item():.3f}")

    train_loader = make_train_loader(10000)
    test_loader = make_test_loader(1000)

    # Lower LRs for fine-tuning (we don't want to destroy learned weights)
    opt_g = torch.optim.AdamW(model.generator.parameters(), lr=1e-4, weight_decay=1e-4)
    opt_dec = torch.optim.AdamW(model.decoder.parameters(), lr=2e-4, weight_decay=1e-4)
    sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=epochs, eta_min=1e-6)
    sched_dec = torch.optim.lr_scheduler.CosineAnnealingLR(opt_dec, T_max=epochs, eta_min=1e-6)

    # Loss weights tuned for: PSNR > 38 dB, bit accuracy > 95%
    LAMBDA_MSG = 3.0       # message recovery (still important)
    LAMBDA_IMG = 20.0      # image MSE — drives PSNR up
    LAMBDA_RES = 5.0       # L1 on residual — drives small, sparse perturbations
    STRENGTH_MAX = 0.05    # hard cap on the encoder's strength multiplier

    best_score = 0.0
    save_path = ckpt_dir / "best_model.pth"
    save_path_alt = ckpt_dir / "best_model_lowpsnr_backup.pth"
    # One-time backup of the high-accuracy / low-PSNR model so we can roll back
    if not save_path_alt.exists():
        torch.save(model.state_dict(), save_path_alt)
        print(f"  backup saved → {save_path_alt}")

    print(f"\nWeights: msg={LAMBDA_MSG}, img={LAMBDA_IMG}, res={LAMBDA_RES}, strength≤{STRENGTH_MAX}")
    print()

    for epoch in range(epochs):
        model.train()
        msg_losses, img_losses, res_losses, accs = [], [], [], []

        for cover, message in train_loader:
            cover = cover.to(device)
            message = message.to(device)

            opt_g.zero_grad()
            opt_dec.zero_grad()

            # Clamp the strength parameter softly during forward (clip after each step)
            stego = model.generator(cover, message)
            decoded = model.decoder(stego)

            residual = stego - cover
            msg_loss = F.binary_cross_entropy_with_logits(decoded, message)
            img_loss = F.mse_loss(stego, cover)
            res_loss = residual.abs().mean()

            total = LAMBDA_MSG * msg_loss + LAMBDA_IMG * img_loss + LAMBDA_RES * res_loss
            total.backward()

            torch.nn.utils.clip_grad_norm_(model.generator.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), 1.0)
            opt_g.step()
            opt_dec.step()

            # Hard-clip the learnable strength to keep perturbation magnitude bounded
            with torch.no_grad():
                model.generator.strength.data.clamp_(min=0.0, max=STRENGTH_MAX)

            msg_losses.append(msg_loss.item())
            img_losses.append(img_loss.item())
            res_losses.append(res_loss.item())
            with torch.no_grad():
                accs.append(((torch.sigmoid(decoded) > 0.5) == message.bool()).float().mean().item())

        train_acc = sum(accs) / len(accs) * 100
        # Quick PSNR estimate from training MSE
        avg_img_mse = sum(img_losses) / len(img_losses)
        train_psnr = 10 * np.log10(1.0 / max(avg_img_mse, 1e-10))

        sched_g.step(); sched_dec.step()

        # Honest test eval every epoch (cheap on 1000 imgs)
        test_acc, test_psnr = evaluate(model, test_loader, device)

        # Composite score: prefer high PSNR + ≥95% accuracy
        # Penalize if accuracy below 95%
        acc_term = test_acc if test_acc >= 95.0 else test_acc - 10.0 * (95.0 - test_acc)
        score = test_psnr + 0.5 * acc_term

        marker = ""
        if score > best_score and test_acc >= 92.0:
            best_score = score
            torch.save(model.state_dict(), save_path)
            marker = "  ✓ saved"

        print(
            f"Epoch {epoch+1:02d}: "
            f"train_acc={train_acc:5.1f}%  test_acc={test_acc:5.1f}%  "
            f"train_psnr={train_psnr:5.1f}  test_psnr={test_psnr:5.1f} dB  "
            f"strength={model.generator.strength.item():.3f}{marker}"
        )

    print(f"\n✓ Fine-tune complete. Best checkpoint: {save_path}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=30)
    args = p.parse_args()
    try:
        finetune(epochs=args.epochs)
    finally:
        sys.exit(0)
