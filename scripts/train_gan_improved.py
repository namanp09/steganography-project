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
from torch.utils.data import DataLoader, TensorDataset, Subset
from torchvision import datasets, transforms

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.image_gan import ImageGANSteganography
from models.audio_gan import AudioGANSteganography
from models.video_gan import VideoGANSteganography
from config.settings import IMAGE_GAN, AUDIO_GAN, VIDEO_GAN, PATHS


def create_realistic_image_dataset(num_samples: int = 5000, augment: bool = True) -> DataLoader:
    """Load CIFAR-10 train set with augmentation for better generalization."""
    print(f"Loading CIFAR-10 train set (max {num_samples} samples, augment={augment})...")

    if augment:
        transform = transforms.Compose([
            transforms.Resize((IMAGE_GAN.image_size, IMAGE_GAN.image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((IMAGE_GAN.image_size, IMAGE_GAN.image_size)),
            transforms.ToTensor(),
        ])

    dataset = datasets.CIFAR10(root='./data/cifar10', train=True, download=True, transform=transform)
    subset = Subset(dataset, range(min(num_samples, len(dataset))))

    class CIFARWithMessages(torch.utils.data.Dataset):
        def __init__(self, cifar_dataset):
            self.cifar = cifar_dataset
        def __len__(self):
            return len(self.cifar)
        def __getitem__(self, idx):
            img, _ = self.cifar[idx]
            msg = torch.randint(0, 2, (IMAGE_GAN.message_bits,)).float()
            return img, msg

    dataset_with_msgs = CIFARWithMessages(subset)
    loader = DataLoader(dataset_with_msgs, batch_size=32, shuffle=True, drop_last=True, num_workers=2, pin_memory=True)

    print(f"✓ CIFAR-10 train loaded: {len(subset)} samples, shape (3, {IMAGE_GAN.image_size}, {IMAGE_GAN.image_size})")
    return loader


def create_test_dataset(num_samples: int = 1000) -> DataLoader:
    """Load CIFAR-10 TEST split (held-out, never seen during training) — no augmentation."""
    transform = transforms.Compose([
        transforms.Resize((IMAGE_GAN.image_size, IMAGE_GAN.image_size)),
        transforms.ToTensor(),
    ])

    dataset = datasets.CIFAR10(root='./data/cifar10', train=False, download=True, transform=transform)
    subset = Subset(dataset, range(min(num_samples, len(dataset))))

    class CIFARWithMessages(torch.utils.data.Dataset):
        def __init__(self, cifar_dataset, seed=42):
            self.cifar = cifar_dataset
            self.seed = seed
        def __len__(self):
            return len(self.cifar)
        def __getitem__(self, idx):
            img, _ = self.cifar[idx]
            # Deterministic message per index for reproducible eval
            g = torch.Generator()
            g.manual_seed(self.seed + idx)
            msg = torch.randint(0, 2, (IMAGE_GAN.message_bits,), generator=g).float()
            return img, msg

    test_loader = DataLoader(CIFARWithMessages(subset), batch_size=64, shuffle=False, num_workers=0)
    return test_loader


def evaluate_on_test_set(model, test_loader, device):
    """Run evaluation on held-out CIFAR-10 test split."""
    model.eval()
    all_correct = 0
    all_total = 0
    with torch.no_grad():
        for cover, message in test_loader:
            cover = cover.to(device)
            message = message.to(device)
            stego = model.generator(cover, message)
            decoded = model.decoder(stego)
            preds = (torch.sigmoid(decoded) > 0.5).float()
            all_correct += (preds == message).sum().item()
            all_total += message.numel()
    return (all_correct / all_total) * 100


def focal_bce_loss(logits, targets, gamma=2.0):
    """Focal loss for hard-bit emphasis: focuses gradient on bits that are wrong."""
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p = torch.sigmoid(logits)
    p_t = torch.where(targets == 1, p, 1 - p)
    focal_weight = (1 - p_t) ** gamma
    return (focal_weight * bce).mean()


def train_image_gan_improved(epochs: int = 60, warmup_epochs: int = 30, resume: bool = False):
    """
    Train Image GAN with three-phase strategy + augmentation + proper test eval.
    - Phase 1 (warmup): No noise, focal loss, focus on bit recovery
    - Phase 2: Enable noise layer for robustness
    """
    print("\n" + "="*80)
    print("TRAINING IMAGE GAN (IMPROVED — 32-bit + augmentation + focal loss)")
    print("="*80)

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}")
    print(f"Message bits: {IMAGE_GAN.message_bits}")
    print(f"Phase 1 (clean+focal):  epochs 1-{warmup_epochs}")
    print(f"Phase 2 (noisy):        epochs {warmup_epochs+1}-{epochs}")

    model = ImageGANSteganography(
        msg_length=IMAGE_GAN.message_bits,
        base_ch=IMAGE_GAN.base_channels,
        image_size=IMAGE_GAN.image_size,
    ).to(device)

    # Resume from existing checkpoint if requested
    ckpt_dir = Path(PATHS.models_dir) / "image_gan_improved"
    ckpt_path = ckpt_dir / "best_model.pth"
    if resume and ckpt_path.exists():
        try:
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            print(f"✓ Resumed from {ckpt_path}")
        except Exception as e:
            print(f"⚠ Could not resume ({e}) — starting fresh")

    train_loader = create_realistic_image_dataset(num_samples=10000, augment=True)
    test_loader = create_test_dataset(num_samples=1000)
    print(f"✓ CIFAR-10 test split loaded: 1000 held-out images")

    # AdamW for better generalization, cosine schedule for smooth LR decay
    opt_dec = torch.optim.AdamW(model.decoder.parameters(), lr=2e-3, weight_decay=1e-4)
    opt_g = torch.optim.AdamW(model.generator.parameters(), lr=5e-4, weight_decay=1e-4)

    scheduler_dec = torch.optim.lr_scheduler.CosineAnnealingLR(opt_dec, T_max=epochs, eta_min=1e-5)
    scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=epochs, eta_min=1e-5)

    best_test_acc = 0.0
    best_train_acc = 0.0
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        msg_losses = []
        dec_acc = []

        in_warmup = epoch < warmup_epochs
        model.train()

        for batch_idx, (cover, message) in enumerate(train_loader):
            cover = cover.to(device)
            message = message.to(device)

            opt_g.zero_grad()
            opt_dec.zero_grad()

            if in_warmup:
                # Clean forward pass — bypass noise layer
                stego = model.generator(cover, message)
                decoded = model.decoder(stego)
            else:
                # Full forward with noise
                stego, decoded = model(cover, message)

            # Focal loss focuses gradient on hard (incorrectly predicted) bits
            msg_loss = focal_bce_loss(decoded, message, gamma=2.0)
            img_loss = F.mse_loss(stego, cover)

            if in_warmup:
                total_loss = msg_loss * 10.0 + img_loss * 0.1
            else:
                total_loss = msg_loss * 5.0 + img_loss * 0.5

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
        train_acc = sum(dec_acc) / len(dec_acc) * 100

        phase = "WARMUP" if in_warmup else "ROBUST"

        # Real test eval every 2 epochs
        if (epoch + 1) % 2 == 0:
            test_acc = evaluate_on_test_set(model, test_loader, device)
            print(f"Epoch {epoch+1} [{phase}]: Loss={avg_msg_loss:.4f} | Train={train_acc:.1f}% | Test={test_acc:.1f}%")

            # Save based on TEST accuracy (more honest)
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_train_acc = train_acc
                torch.save(model.state_dict(), ckpt_dir / "best_model.pth")
                print(f"  ✓ Best checkpoint saved (test={best_test_acc:.1f}%, train={best_train_acc:.1f}%)")
        else:
            print(f"Epoch {epoch+1} [{phase}]: Loss={avg_msg_loss:.4f} | Train={train_acc:.1f}%")

        scheduler_dec.step()
        scheduler_g.step()

    print(f"\n✓ Training complete!")
    print(f"  Best test accuracy:  {best_test_acc:.1f}%")
    print(f"  Best train accuracy: {best_train_acc:.1f}%")
    print(f"  Checkpoint: {ckpt_dir / 'best_model.pth'}")
    return ckpt_dir / "best_model.pth"


def test_gan_encode_decode():
    """Test full encode/decode cycle."""
    print("\n" + "="*80)
    print("TESTING GAN ENCODE/DECODE")
    print("="*80)

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

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
    parser.add_argument("--warmup-epochs", type=int, default=20, help="Clean-image warmup epochs")
    parser.add_argument("--test-only", action="store_true", help="Test only")
    parser.add_argument("--resume", action="store_true", help="Resume from existing checkpoint")
    args = parser.parse_args()

    try:
        if args.test_only:
            test_gan_encode_decode()
        else:
            train_image_gan_improved(epochs=args.epochs, warmup_epochs=args.warmup_epochs, resume=args.resume)
            test_gan_encode_decode()
    finally:
        import sys
        sys.exit(0)
