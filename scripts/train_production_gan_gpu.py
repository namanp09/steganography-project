#!/usr/bin/env python3
"""
Train the **in-repo** Image / Video / Audio GANs on CUDA with AMP, train/val splits,
and checkpoint selection by **validation bit accuracy** (a better target than training
loss alone for generalization).

**Important:** `colab_train_gpu.py` at the project root uses a *different* architecture;
checkpoints from that file will **not** load into `api/` and `core/*/gan_stego.py`. Use
this script (or the `*_improved` trainers with matching `config.settings`) for
deployment-compatible models.

**90%+ bit accuracy** is a training target, not a guarantee. Use enough diverse data
(real images/frames where possible), sufficient epochs, and a held-out val set. If
decode is still short of the target, try lowering `message_bits` in
`config/settings.py` and retraining (often easier than 128 bits at fixed capacity).

**Device selection:** by default, uses `cuda` if available, else Apple `mps`, else `cpu`.  
Force NVIDIA: `--device cuda` · Apple GPU: `--device mps` · no GPU: `--device cpu`

Example (Image GAN, from project root, with GPU):
  python scripts/train_production_gan_gpu.py --modality image --epochs 200 \\
    --train-samples 8000 --batch 32

Explicit train/val counts (7000 train + 1000 val = 8000 total):
  python scripts/train_production_gan_gpu.py --modality image --epochs 200 \\
    --train-samples 7000 --val-samples 1000 --batch 32
"""

from __future__ import annotations

import argparse
import contextlib
import os
import random
import sys
from pathlib import Path

# Allow `python scripts/train_production_gan_gpu.py` from any cwd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm.auto import tqdm

from config.settings import GAN_TRAINING, IMAGE_GAN, PATHS, AUDIO_GAN, VIDEO_GAN
from models.audio_gan import AudioGANSteganography
from models.image_gan import ImageGANSteganography
from models.video_gan import VideoGANSteganography

from torch.cuda.amp import GradScaler, autocast

try:
    torch.set_float32_matmul_precision("high")
except AttributeError:
    pass


def _resolve_device(choice: str) -> torch.device:
    """NVIDIA (cuda) > Apple Metal (mps) > CPU when choice is *auto*."""
    c = (choice or "auto").strip().lower()
    if c == "cpu":
        return torch.device("cpu")
    if c == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit(
                "Requested --device cuda but CUDA is not available. Install a PyTorch build "
                "with CUDA and NVIDIA drivers (see https://pytorch.org/get-started/locally/)."
            )
        return torch.device("cuda")
    if c == "mps":
        if not _mps_ok():
            raise SystemExit(
                "Requested --device mps but MPS is not available (needs macOS with Apple GPU and supported PyTorch)."
            )
        return torch.device("mps")
    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if _mps_ok():
        return torch.device("mps")
    return torch.device("cpu")


def _mps_ok() -> bool:
    b = getattr(torch.backends, "mps", None)
    return b is not None and b.is_available()


def _epoch_range(epochs: int, no_tqdm: bool):
    """Outer progress: time remaining in `[elapsed<remaining]` is for the **whole** training run (all epochs)."""
    r = range(1, epochs + 1)
    if no_tqdm:
        return r
    return tqdm(
        r,
        desc="All epochs (run)",
        total=epochs,
        unit="ep",
        mininterval=0.3,
        position=0,
        leave=True,
        dynamic_ncols=True,
    )


def _batch_pbar(
    loader: DataLoader,
    *,
    no_tqdm: bool,
    epoch: int,
    epochs: int,
    name: str,
    nested: bool = False,
) -> DataLoader:
    if no_tqdm:
        return loader
    return tqdm(
        loader,
        desc=f"Batches {epoch}/{epochs} [{name}]",
        leave=bool(nested),
        mininterval=0.3,
        dynamic_ncols=True,
        position=1 if nested else 0,
    )


def _out(log_path: str | None, message: str) -> None:
    """Print to stdout and optionally append the same line to a file (for `tail` in another Colab cell)."""
    print(message, flush=True)
    if not log_path:
        return
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(message + "\n")
    except OSError as e:
        print(f"(log-file write failed: {e})", flush=True)


def _log_path_arg(args: argparse.Namespace) -> str | None:
    p = getattr(args, "log_file", None) or ""
    return p.strip() or None


def _autocast_ctx(device: torch.device, use_amp: bool):
    # AMP/GradScaler are CUDA-only in this script (MPS/CPU: full float32, stable)
    if use_amp and device.type == "cuda":
        return autocast()
    return contextlib.nullcontext()


def _scaler_if_cuda(device: torch.device, use_amp: bool) -> GradScaler | None:
    if use_amp and device.type == "cuda":
        return GradScaler()
    return None


def _bit_accuracy_pct(logits: torch.Tensor, message: torch.Tensor) -> float:
    with torch.no_grad():
        return (
            (torch.sigmoid(logits) > 0.5) == message.bool()
        ).float().mean().item() * 100.0


def _make_synthetic_image_tensors(
    n: int, image_size: int, message_bits: int
) -> TensorDataset:
    """Caller should set torch.manual_seed for reproducibility."""
    images, messages = [], []
    for _ in range(n):
        img = torch.zeros(3, image_size, image_size)
        for i in range(image_size):
            img[:, i, :] = torch.rand(3, image_size) * 0.8
        for j in range(0, image_size, 32):
            h = min(32, image_size - j)
            img[:, j : j + h, :] += torch.randn(3, h, image_size) * 0.3
        images.append(torch.clamp(img, 0, 1))
        messages.append(torch.randint(0, 2, (message_bits,)).float())
    return TensorDataset(torch.stack(images), torch.stack(messages))


def _make_synthetic_video_tensors(
    n: int, t: int, frame_size: int, message_bits: int
) -> TensorDataset:
    vids, messages = [], []
    for _ in range(n):
        vids.append(torch.rand(t, 3, frame_size, frame_size))
        messages.append(torch.randint(0, 2, (message_bits,)).float())
    return TensorDataset(torch.stack(vids), torch.stack(messages))


def _make_synthetic_audio_tensors(
    n: int, freq_bins: int, message_bits: int
) -> TensorDataset:
    mags, phases, messages = [], [], []
    for _ in range(n):
        mags.append(torch.exp(torch.randn(1, freq_bins, 64) * 0.5 - 1))
        phases.append(torch.randn(1, freq_bins, 64))
        messages.append(torch.randint(0, 2, (message_bits,)).float())
    return TensorDataset(
        torch.stack(mags), torch.stack(phases), torch.stack(messages)
    )


def _make_train_val_loaders(
    dataset: TensorDataset,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[DataLoader, DataLoader, int, int]:
    n = len(dataset)
    if args.val_samples and args.val_samples > 0:
        n_train, n_val = int(args.train_samples), int(args.val_samples)
        if n_train + n_val != n:
            raise SystemExit(
                f"Dataset size {n} != --train-samples {n_train} + --val-samples {n_val}"
            )
    else:
        n_val = max(1, int(n * args.val_ratio))
        n_train = n - n_val
    if n_train < 1:
        raise SystemExit("Train/val split empty: increase pool size or lower --val-ratio")

    gen = torch.Generator().manual_seed(42)
    train_d, val_d = random_split(dataset, [n_train, n_val], generator=gen)
    pin = device.type == "cuda"  # only CUDA benefits from pin_memory
    n_workers = args.num_workers
    train_l = DataLoader(
        train_d,
        batch_size=args.batch,
        shuffle=True,
        drop_last=True,
        num_workers=n_workers,
        pin_memory=pin,
    )
    val_l = DataLoader(
        val_d,
        batch_size=args.batch,
        shuffle=False,
        drop_last=False,
        num_workers=n_workers,
        pin_memory=pin,
    )
    if len(train_l) < 1:
        raise SystemExit(
            "Training DataLoader is empty: increase --train-samples, lower --batch, or increase pool size."
        )
    return train_l, val_l, n_train, n_val


@torch.no_grad()
def _eval_image(
    model: ImageGANSteganography, val_loader: DataLoader, device: torch.device
) -> float:
    model.eval()
    accs: list[float] = []
    for cover, message in val_loader:
        cover = cover.to(device, non_blocking=True)
        message = message.to(device, non_blocking=True)
        _, decoded = model(cover, message)
        accs.append(_bit_accuracy_pct(decoded, message))
    model.train()
    return float(sum(accs) / max(len(accs), 1))


@torch.no_grad()
def _eval_video(
    model: VideoGANSteganography, val_loader: DataLoader, device: torch.device
) -> float:
    model.eval()
    accs: list[float] = []
    for video, message in val_loader:
        video = video.to(device, non_blocking=True)
        message = message.to(device, non_blocking=True)
        _, decoded = model(video, message, flow=None)
        accs.append(_bit_accuracy_pct(decoded, message))
    model.train()
    return float(sum(accs) / max(len(accs), 1))


@torch.no_grad()
def _eval_audio(
    model: AudioGANSteganography, val_loader: DataLoader, device: torch.device
) -> float:
    model.eval()
    accs: list[float] = []
    for magnitude, phase, message in val_loader:
        magnitude = magnitude.to(device, non_blocking=True)
        phase = phase.to(device, non_blocking=True)
        message = message.to(device, non_blocking=True)
        _, decoded = model(magnitude, phase, message)
        accs.append(_bit_accuracy_pct(decoded, message))
    model.train()
    return float(sum(accs) / max(len(accs), 1))


def _pool_size_for_args(args: argparse.Namespace) -> int:
    if args.val_samples and args.val_samples > 0:
        return int(args.train_samples) + int(args.val_samples)
    return int(args.train_samples)


def _train_image(
    args: argparse.Namespace, device: torch.device
) -> Path:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    log = _log_path_arg(args)

    total = _pool_size_for_args(args)
    _out(
        log,
        f"Building {total} synthetic image samples (can take a few minutes; no epoch lines until this finishes)…",
    )
    ds = _make_synthetic_image_tensors(
        total, IMAGE_GAN.image_size, IMAGE_GAN.message_bits
    )
    train_l, val_l, n_t, n_v = _make_train_val_loaders(ds, args, device)
    _out(
        log,
        f"[image] size={IMAGE_GAN.image_size} base_ch={IMAGE_GAN.base_channels} "
        f"train={n_t} val={n_v} batch={args.batch}",
    )

    model = ImageGANSteganography(
        msg_length=IMAGE_GAN.message_bits,
        base_ch=IMAGE_GAN.base_channels,
        image_size=IMAGE_GAN.image_size,
    ).to(device)

    opt_g = torch.optim.Adam(
        model.generator.parameters(),
        lr=args.lr_g,
        betas=(0.9, 0.999),
        weight_decay=GAN_TRAINING.weight_decay,
    )
    opt_dec = torch.optim.Adam(
        model.decoder.parameters(), lr=args.lr_dec, betas=(0.9, 0.999)
    )

    use_amp = bool(args.amp) and device.type == "cuda"
    scaler = _scaler_if_cuda(device, use_amp)  # None on MPS/CPU

    subdir = "image_gan_improved"
    if args.output_subdir:
        subdir = args.output_subdir
    ckpt_dir = Path(PATHS.models_dir) / subdir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "best_model.pth"

    best_val = -1.0
    for epoch in _epoch_range(args.epochs, args.no_tqdm):
        model.train()
        train_accs: list[float] = []
        pbar = _batch_pbar(
            train_l,
            no_tqdm=args.no_tqdm,
            epoch=epoch,
            epochs=args.epochs,
            name="image",
            nested=not args.no_tqdm,
        )
        for cover, message in pbar:
            cover = cover.to(device, non_blocking=True)
            message = message.to(device, non_blocking=True)
            opt_g.zero_grad()
            opt_dec.zero_grad()
            with _autocast_ctx(device, use_amp):
                stego, decoded = model(cover, message)
                msg_loss = F.binary_cross_entropy_with_logits(decoded, message)
                img_loss = F.mse_loss(stego, cover)
                loss = msg_loss * args.msg_w + img_loss * args.recon_w
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(opt_g)
                scaler.unscale_(opt_dec)
                torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(model.generator.parameters(), 1.0)
                scaler.step(opt_g)
                scaler.step(opt_dec)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(model.generator.parameters(), 1.0)
                opt_g.step()
                opt_dec.step()
            bit_now = _bit_accuracy_pct(decoded, message)
            train_accs.append(bit_now)
            if not args.no_tqdm and isinstance(pbar, tqdm):
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    msg=f"{msg_loss.item():.4f}",
                    bit=f"{bit_now:.1f}",
                )

        val_bit = _eval_image(model, val_l, device)
        train_bit = float(sum(train_accs) / max(len(train_accs), 1))
        if epoch == 1 or epoch % max(1, args.log_every) == 0 or epoch == args.epochs:
            _out(
                log,
                f"Epoch {epoch:4d}/{args.epochs}  train_bit%={train_bit:.2f}  val_bit%={val_bit:.2f}",
            )
        if val_bit > best_val:
            best_val = val_bit
            torch.save(model.state_dict(), best_path)
            _out(log, f"  -> saved (best val bit acc {best_val:.2f}%)  {best_path}")
    return best_path


def _train_video(
    args: argparse.Namespace, device: torch.device
) -> Path:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    log = _log_path_arg(args)
    total = _pool_size_for_args(args)
    _out(log, f"Building {total} synthetic video sequences (can take a few minutes)…")
    ds = _make_synthetic_video_tensors(
        total,
        VIDEO_GAN.temporal_window,
        VIDEO_GAN.frame_size,
        VIDEO_GAN.message_bits,
    )
    train_l, val_l, n_t, n_v = _make_train_val_loaders(ds, args, device)
    _out(
        log,
        f"[video] T={VIDEO_GAN.temporal_window} frame={VIDEO_GAN.frame_size} "
        f"base_ch={VIDEO_GAN.base_channels} train={n_t} val={n_v}",
    )
    model = VideoGANSteganography(
        msg_length=VIDEO_GAN.message_bits,
        base_ch=VIDEO_GAN.base_channels,
        temporal_window=VIDEO_GAN.temporal_window,
        frame_size=VIDEO_GAN.frame_size,
    ).to(device)
    opt_g = torch.optim.Adam(
        model.generator.parameters(), lr=args.lr_g, betas=(0.9, 0.999)
    )
    opt_dec = torch.optim.Adam(
        model.decoder.parameters(), lr=args.lr_dec, betas=(0.9, 0.999)
    )
    use_amp = bool(args.amp) and device.type == "cuda"
    scaler = _scaler_if_cuda(device, use_amp)
    subdir = "video_gan_improved"
    if args.output_subdir:
        subdir = args.output_subdir
    ckpt_dir = Path(PATHS.models_dir) / subdir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "best_model.pth"
    best_val = -1.0
    for epoch in _epoch_range(args.epochs, args.no_tqdm):
        model.train()
        train_accs: list[float] = []
        pbar = _batch_pbar(
            train_l,
            no_tqdm=args.no_tqdm,
            epoch=epoch,
            epochs=args.epochs,
            name="video",
            nested=not args.no_tqdm,
        )
        for video, message in pbar:
            video = video.to(device, non_blocking=True)
            message = message.to(device, non_blocking=True)
            opt_g.zero_grad()
            opt_dec.zero_grad()
            with _autocast_ctx(device, use_amp):
                stego, decoded = model(video, message, flow=None)
                msg_loss = F.binary_cross_entropy_with_logits(decoded, message)
                video_loss = F.mse_loss(stego, video)
                loss = msg_loss * args.msg_w + video_loss * args.recon_w
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(opt_g)
                scaler.unscale_(opt_dec)
                torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(model.generator.parameters(), 1.0)
                scaler.step(opt_g)
                scaler.step(opt_dec)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(model.generator.parameters(), 1.0)
                opt_g.step()
                opt_dec.step()
            bit_now = _bit_accuracy_pct(decoded, message)
            train_accs.append(bit_now)
            if not args.no_tqdm and isinstance(pbar, tqdm):
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    msg=f"{msg_loss.item():.4f}",
                    bit=f"{bit_now:.1f}",
                )
        val_bit = _eval_video(model, val_l, device)
        train_bit = float(sum(train_accs) / max(len(train_accs), 1))
        if epoch == 1 or epoch % max(1, args.log_every) == 0 or epoch == args.epochs:
            _out(
                log,
                f"Epoch {epoch:4d}/{args.epochs}  train_bit%={train_bit:.2f}  val_bit%={val_bit:.2f}",
            )
        if val_bit > best_val:
            best_val = val_bit
            torch.save(model.state_dict(), best_path)
            _out(log, f"  -> saved (best val bit acc {best_val:.2f}%)  {best_path}")
    return best_path


def _train_audio(
    args: argparse.Namespace, device: torch.device
) -> Path:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    log = _log_path_arg(args)
    total = _pool_size_for_args(args)
    _out(log, f"Building {total} synthetic audio samples (can take a few minutes)…")
    ds = _make_synthetic_audio_tensors(
        total, AUDIO_GAN.freq_bins, AUDIO_GAN.message_bits
    )
    train_l, val_l, _, _ = _make_train_val_loaders(ds, args, device)
    _out(log, f"[audio] freq_bins={AUDIO_GAN.freq_bins} base_ch={AUDIO_GAN.base_channels}")
    model = AudioGANSteganography(
        msg_length=AUDIO_GAN.message_bits,
        freq_bins=AUDIO_GAN.freq_bins,
        base_ch=AUDIO_GAN.base_channels,
    ).to(device)
    opt_g = torch.optim.Adam(
        model.generator.parameters(), lr=args.lr_g, betas=(0.9, 0.999)
    )
    opt_dec = torch.optim.Adam(
        model.decoder.parameters(), lr=args.lr_dec, betas=(0.9, 0.999)
    )
    use_amp = bool(args.amp) and device.type == "cuda"
    scaler = _scaler_if_cuda(device, use_amp)
    subdir = "audio_gan_improved"
    if args.output_subdir:
        subdir = args.output_subdir
    ckpt_dir = Path(PATHS.models_dir) / subdir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "best_model.pth"
    best_val = -1.0
    for epoch in _epoch_range(args.epochs, args.no_tqdm):
        model.train()
        train_accs: list[float] = []
        pbar = _batch_pbar(
            train_l,
            no_tqdm=args.no_tqdm,
            epoch=epoch,
            epochs=args.epochs,
            name="audio",
            nested=not args.no_tqdm,
        )
        for mag, ph, message in pbar:
            mag = mag.to(device, non_blocking=True)
            ph = ph.to(device, non_blocking=True)
            message = message.to(device, non_blocking=True)
            opt_g.zero_grad()
            opt_dec.zero_grad()
            with _autocast_ctx(device, use_amp):
                stego_mag, decoded = model(mag, ph, message)
                msg_loss = F.binary_cross_entropy_with_logits(decoded, message)
                audio_loss = F.mse_loss(stego_mag, mag)
                loss = msg_loss * args.msg_w + audio_loss * args.recon_w
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(opt_g)
                scaler.unscale_(opt_dec)
                torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(model.generator.parameters(), 1.0)
                scaler.step(opt_g)
                scaler.step(opt_dec)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(model.generator.parameters(), 1.0)
                opt_g.step()
                opt_dec.step()
            bit_now = _bit_accuracy_pct(decoded, message)
            train_accs.append(bit_now)
            if not args.no_tqdm and isinstance(pbar, tqdm):
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    msg=f"{msg_loss.item():.4f}",
                    bit=f"{bit_now:.1f}",
                )
        val_bit = _eval_audio(model, val_l, device)
        train_bit = float(sum(train_accs) / max(len(train_accs), 1))
        if epoch == 1 or epoch % max(1, args.log_every) == 0 or epoch == args.epochs:
            _out(
                log,
                f"Epoch {epoch:4d}/{args.epochs}  train_bit%={train_bit:.2f}  val_bit%={val_bit:.2f}",
            )
        if val_bit > best_val:
            best_val = val_bit
            torch.save(model.state_dict(), best_path)
            _out(log, f"  -> saved (best val bit acc {best_val:.2f}%)  {best_path}")
    return best_path


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--modality", choices=["image", "video", "audio"], required=True
    )
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument(
        "--train-samples",
        type=int,
        default=8_000,
        help="With --val-samples 0: total pool size, split by --val-ratio. "
        "With --val-samples N: number of *training* samples (val is separate).",
    )
    p.add_argument(
        "--val-samples",
        type=int,
        default=0,
        help="If 0, take val as val_ratio of the pool. Else pool = train+val, split exactly.",
    )
    p.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Val fraction of pool (only if --val-samples is 0).",
    )
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--msg-w",
        type=float,
        default=10.0,
        help="BCE on message logits weight (vs MSE stego).",
    )
    p.add_argument(
        "--recon-w", type=float, default=0.3, help="MSE stego/cover (or spec) weight."
    )
    p.add_argument("--lr-g", type=float, default=2e-4)
    p.add_argument("--lr-d", type=float, default=2e-4)
    p.add_argument("--lr-dec", type=float, default=1e-2)
    p.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable automatic mixed precision (e.g. on CPU or if you see NaNs).",
    )
    p.add_argument(
        "--output-subdir",
        type=str,
        default="",
        help="If set, save to models/checkpoints/<subdir>/best_model.pth",
    )
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="auto: use CUDA if present, else Apple MPS, else CPU. "
        "Use cuda on NVIDIA; mps on Apple Silicon.",
    )
    p.add_argument(
        "--cpu", action="store_true", help="Shorthand for --device cpu (slow; for quick tests only)"
    )
    p.add_argument(
        "--log-file",
        type=str,
        default="",
        help="Also append the same log lines to this file (for `!tail` in a second Colab cell).",
    )
    p.add_argument(
        "--no-tqdm",
        action="store_true",
        help="Disable per-batch progress bars (plain log lines only; useful in CI).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse()
    args.amp = not args.no_amp
    if args.val_samples and args.val_samples < 0:
        raise SystemExit("--val-samples must be non-negative")
    if args.val_samples and args.val_samples > 0 and args.train_samples < 1:
        raise SystemExit("--train-samples must be positive when using --val-samples")

    if args.cpu:
        dev_choice = "cpu"
    else:
        dev_choice = args.device
    device = _resolve_device(dev_choice)
    lp = _log_path_arg(args)
    _out(lp, f"Device: {device}")
    if device.type == "cuda" and args.amp:
        _out(lp, "  (CUDA + automatic mixed precision)")
    elif device.type == "mps":
        _out(lp, "  (Apple Metal — full float32; use smaller --batch if you hit OOM)")
    if dev_choice == "auto" and device.type == "cpu":
        _out(
            lp,
            "WARNING: No CUDA or MPS; training on CPU is very slow. "
            "For NVIDIA: install CUDA PyTorch on a GPU machine; for Mac: use Apple Silicon with --device mps (default picks MPS if available).",
        )
    if args.modality == "image":
        path = _train_image(args, device)
    elif args.modality == "video":
        path = _train_video(args, device)
    else:
        path = _train_audio(args, device)
    _out(lp, f"Done. Checkpoint: {path}")


if __name__ == "__main__":
    main()
