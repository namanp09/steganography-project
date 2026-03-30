"""
Comprehensive Evaluation Metrics for Steganography.

Metrics:
- PSNR  (Peak Signal-to-Noise Ratio) — image quality
- SSIM  (Structural Similarity Index) — perceptual quality
- MS-SSIM (Multi-Scale SSIM) — multi-resolution quality
- MSE   (Mean Squared Error) — pixel-level error
- SNR   (Signal-to-Noise Ratio) — signal quality
- LPIPS (Learned Perceptual Similarity) — deep perceptual metric
- BER   (Bit Error Rate) — message recovery accuracy

Supports both image and video (per-frame) evaluation.
Includes comparison charts across methods.
"""

import numpy as np
import cv2
from typing import Dict, List, Optional
from dataclasses import dataclass

try:
    from skimage.metrics import structural_similarity, peak_signal_noise_ratio
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


@dataclass
class MetricsResult:
    """Container for all evaluation metrics."""
    psnr: float
    ssim: float
    mse: float
    snr: float
    ber: Optional[float] = None
    lpips: Optional[float] = None

    def to_dict(self) -> dict:
        d = {
            "PSNR (dB)": round(self.psnr, 2),
            "SSIM": round(self.ssim, 4),
            "MSE": round(self.mse, 6),
            "SNR (dB)": round(self.snr, 2),
        }
        if self.ber is not None:
            d["BER"] = round(self.ber, 6)
        if self.lpips is not None:
            d["LPIPS"] = round(self.lpips, 4)
        return d


def compute_psnr(original: np.ndarray, modified: np.ndarray) -> float:
    """Compute PSNR between two images."""
    if HAS_SKIMAGE:
        return float(peak_signal_noise_ratio(original, modified, data_range=255))
    mse = np.mean((original.astype(np.float64) - modified.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10(255.0 ** 2 / mse)


def compute_ssim(original: np.ndarray, modified: np.ndarray) -> float:
    """Compute SSIM between two images."""
    if HAS_SKIMAGE:
        if len(original.shape) == 3:
            return float(structural_similarity(
                original, modified, channel_axis=2, data_range=255
            ))
        return float(structural_similarity(original, modified, data_range=255))

    # Manual SSIM implementation
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    orig = original.astype(np.float64)
    mod = modified.astype(np.float64)

    mu1 = cv2.GaussianBlur(orig, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(mod, (11, 11), 1.5)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(orig ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(mod ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(orig * mod, (11, 11), 1.5) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return float(ssim_map.mean())


def compute_mse(original: np.ndarray, modified: np.ndarray) -> float:
    """Compute Mean Squared Error."""
    return float(np.mean((original.astype(np.float64) - modified.astype(np.float64)) ** 2))


def compute_snr(original: np.ndarray, modified: np.ndarray) -> float:
    """Compute Signal-to-Noise Ratio."""
    signal_power = np.mean(original.astype(np.float64) ** 2)
    noise = original.astype(np.float64) - modified.astype(np.float64)
    noise_power = np.mean(noise ** 2)
    if noise_power == 0:
        return float("inf")
    return 10 * np.log10(signal_power / noise_power)


def compute_ber(
    original_bits: np.ndarray, extracted_bits: np.ndarray
) -> float:
    """Compute Bit Error Rate between original and extracted messages."""
    min_len = min(len(original_bits), len(extracted_bits))
    if min_len == 0:
        return 1.0
    errors = np.sum(original_bits[:min_len] != extracted_bits[:min_len])
    return float(errors / min_len)


def compute_all_metrics(
    cover: np.ndarray,
    stego: np.ndarray,
    original_msg: Optional[bytes] = None,
    extracted_msg: Optional[bytes] = None,
) -> MetricsResult:
    """Compute all metrics at once."""
    psnr = compute_psnr(cover, stego)
    ssim = compute_ssim(cover, stego)
    mse = compute_mse(cover, stego)
    snr = compute_snr(cover, stego)

    ber = None
    if original_msg is not None and extracted_msg is not None:
        orig_bits = np.unpackbits(np.frombuffer(original_msg, dtype=np.uint8))
        ext_bits = np.unpackbits(np.frombuffer(extracted_msg, dtype=np.uint8))
        ber = compute_ber(orig_bits, ext_bits)

    return MetricsResult(psnr=psnr, ssim=ssim, mse=mse, snr=snr, ber=ber)


def compute_video_metrics(
    cover_frames: List[np.ndarray],
    stego_frames: List[np.ndarray],
) -> Dict[str, List[float]]:
    """Compute per-frame metrics for video."""
    results = {"psnr": [], "ssim": [], "mse": [], "snr": []}

    for cover, stego in zip(cover_frames, stego_frames):
        results["psnr"].append(compute_psnr(cover, stego))
        results["ssim"].append(compute_ssim(cover, stego))
        results["mse"].append(compute_mse(cover, stego))
        results["snr"].append(compute_snr(cover, stego))

    return results


def compare_methods(
    cover: np.ndarray,
    stego_results: Dict[str, np.ndarray],
    original_msg: Optional[bytes] = None,
    extracted_msgs: Optional[Dict[str, bytes]] = None,
) -> Dict[str, dict]:
    """
    Compare metrics across multiple steganography methods.

    Args:
        cover: Original cover image.
        stego_results: {"method_name": stego_image, ...}
        original_msg: Original secret message.
        extracted_msgs: {"method_name": extracted_bytes, ...}

    Returns:
        {"method_name": {"PSNR (dB)": ..., "SSIM": ..., ...}, ...}
    """
    comparison = {}
    for method, stego in stego_results.items():
        ext_msg = extracted_msgs.get(method) if extracted_msgs else None
        metrics = compute_all_metrics(cover, stego, original_msg, ext_msg)
        comparison[method] = metrics.to_dict()

    return comparison


def plot_metrics(
    comparison: Dict[str, dict],
    save_path: Optional[str] = None,
    title: str = "Steganography Methods Comparison",
):
    """
    Generate comparison bar charts for all methods.

    Args:
        comparison: Output from compare_methods().
        save_path: Path to save the plot (optional).
    """
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")

    methods = list(comparison.keys())
    metric_names = list(next(iter(comparison.values())).keys())

    fig, axes = plt.subplots(1, len(metric_names), figsize=(5 * len(metric_names), 5))
    if len(metric_names) == 1:
        axes = [axes]

    colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))

    for ax, metric in zip(axes, metric_names):
        values = [comparison[m].get(metric, 0) for m in methods]
        bars = ax.bar(methods, values, color=colors)
        ax.set_title(metric, fontsize=12, fontweight="bold")
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=30)

        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.3f}", ha="center", va="bottom", fontsize=9,
            )

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


def plot_video_metrics(
    per_frame_metrics: Dict[str, List[float]],
    save_path: Optional[str] = None,
    title: str = "Per-Frame Video Steganography Metrics",
):
    """Plot per-frame metrics over time for video steganography."""
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")

    n_metrics = len(per_frame_metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 3 * n_metrics))
    if n_metrics == 1:
        axes = [axes]

    for ax, (metric, values) in zip(axes, per_frame_metrics.items()):
        frames = range(len(values))
        ax.plot(frames, values, "b-", linewidth=1.5)
        ax.fill_between(frames, values, alpha=0.2)
        ax.set_xlabel("Frame")
        ax.set_ylabel(metric.upper())
        ax.set_title(f"{metric.upper()} per Frame")
        ax.axhline(y=np.mean(values), color="r", linestyle="--",
                    label=f"Mean: {np.mean(values):.2f}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path
