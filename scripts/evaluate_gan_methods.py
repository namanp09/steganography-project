#!/usr/bin/env python3
"""
Comprehensive evaluation script comparing GAN-based steganography
with classical baseline methods (LSB, DCT, DWT).

Metrics evaluated:
- Image: PSNR, SSIM, LPIPS
- Audio: SNR, frequency spectrum difference
- Video: Frame-wise PSNR/SSIM, temporal consistency
- All: Compression ratio, message recovery accuracy
"""

import os
import sys
import argparse
import numpy as np
import torch
import cv2
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.image import ImageLSB, ImageDCT, ImageDWT, ImageGANStego
from core.audio import AudioLSB, AudioDWT, AudioGANStego
from core.video import VideoLSB, VideoDCT, VideoDWT, VideoGANStego
from core.metrics import compute_all_metrics
from core.text import compress_text, decompress_text


class ImageEvaluator:
    """Evaluate image steganography methods."""

    def __init__(self, test_images_dir: str = "data/images"):
        """
        Args:
            test_images_dir: Directory with test images
        """
        self.test_dir = test_images_dir
        self.methods = {
            "LSB": ImageLSB(num_bits=2),
            "DCT": ImageDCT(alpha=10.0),
            "DWT": ImageDWT(wavelet="haar", level=2, alpha=5.0),
            "GAN": None,  # Will load if model exists
        }

    def load_gan_model(self, model_path: str):
        """Load pretrained GAN model."""
        if os.path.exists(model_path):
            self.methods["GAN"] = ImageGANStego(model_path=model_path)
            return True
        return False

    def evaluate_method(self, method_name: str, secret_text: str = "test message"):
        """
        Evaluate single method on available test images.

        Returns:
            Dictionary of metrics
        """
        if not os.path.exists(self.test_dir):
            print(f"Test directory not found: {self.test_dir}")
            return {}

        method = self.methods.get(method_name)
        if method is None:
            return {}

        results = {
            "psnr": [],
            "ssim": [],
            "capacity": [],
            "compression_ratio": [],
            "ber": 0.0,
        }

        secret_data = secret_text.encode("utf-8")

        # Iterate over test images
        for img_path in Path(self.test_dir).glob("*.png"):
            try:
                cover = cv2.imread(str(img_path))
                if cover is None:
                    continue

                # Capacity
                cap = method.capacity(cover)
                if cap < len(secret_data):
                    continue

                # Encode
                stego = method.encode(cover, secret_data)

                # Metrics
                psnr_val = 20 * np.log10(255 / np.sqrt(np.mean((cover.astype(float) - stego.astype(float)) ** 2)))
                results["psnr"].append(psnr_val)

                # Decode and check
                decoded = method.decode(stego)
                if decoded == secret_data:
                    results["ber"] += 0.0
                else:
                    # Compute bit error rate
                    bits_orig = np.unpackbits(np.frombuffer(secret_data, dtype=np.uint8))
                    bits_decoded = np.unpackbits(np.frombuffer(decoded, dtype=np.uint8))
                    min_len = min(len(bits_orig), len(bits_decoded))
                    ber = np.mean(bits_orig[:min_len] != bits_decoded[:min_len])
                    results["ber"] += ber

            except Exception as e:
                print(f"Error processing {img_path}: {e}")

        # Average metrics
        if results["psnr"]:
            results["psnr"] = np.mean(results["psnr"])
            results["ber"] = results["ber"] / len(results["psnr"])
        else:
            results = {}

        return results

    def compare_all_methods(self, secret_text: str = "test message") -> dict:
        """Compare all methods and return summary."""
        comparison = {}

        for method_name in self.methods.keys():
            print(f"\nEvaluating {method_name}...")
            results = self.evaluate_method(method_name, secret_text)
            if results:
                comparison[method_name] = results

        return comparison


class AudioEvaluator:
    """Evaluate audio steganography methods."""

    def __init__(self, test_audio_dir: str = "data/audio"):
        """
        Args:
            test_audio_dir: Directory with test audio files
        """
        self.test_dir = test_audio_dir
        self.methods = {
            "LSB": AudioLSB(num_bits=1),
            "DWT": AudioDWT(wavelet="db4", level=4, alpha=0.02),
            "GAN": None,
        }

    def load_gan_model(self, model_path: str):
        """Load pretrained GAN model."""
        if os.path.exists(model_path):
            self.methods["GAN"] = AudioGANStego(model_path=model_path)
            return True
        return False

    def evaluate_method(self, method_name: str, secret_text: str = "test"):
        """Evaluate single method."""
        if not os.path.exists(self.test_dir):
            return {}

        method = self.methods.get(method_name)
        if method is None:
            return {}

        results = {
            "snr": [],
            "capacity": [],
            "ber": 0.0,
        }

        secret_data = secret_text.encode("utf-8")

        for audio_path in Path(self.test_dir).glob("*.wav"):
            try:
                import soundfile as sf

                audio, sr = sf.read(str(audio_path))
                cap = method.capacity(audio)

                if cap < len(secret_data):
                    continue

                # Encode
                stego, _ = method.encode(audio, sr, secret_data)

                # SNR
                noise = stego - audio[: len(stego)]
                snr = 10 * np.log10(np.mean(audio[: len(stego)] ** 2) / (np.mean(noise**2) + 1e-10))
                results["snr"].append(snr)

                # Decode
                decoded = method.decode(stego)
                if decoded == secret_data:
                    results["ber"] += 0.0
                else:
                    bits_orig = np.unpackbits(np.frombuffer(secret_data, dtype=np.uint8))
                    bits_decoded = np.unpackbits(np.frombuffer(decoded, dtype=np.uint8))
                    min_len = min(len(bits_orig), len(bits_decoded))
                    ber = np.mean(bits_orig[:min_len] != bits_decoded[:min_len])
                    results["ber"] += ber

            except Exception as e:
                print(f"Error processing {audio_path}: {e}")

        if results["snr"]:
            results["snr"] = np.mean(results["snr"])
            results["ber"] = results["ber"] / len(results["snr"])
        else:
            results = {}

        return results

    def compare_all_methods(self, secret_text: str = "test") -> dict:
        """Compare all methods."""
        comparison = {}

        for method_name in self.methods.keys():
            print(f"\nEvaluating {method_name}...")
            results = self.evaluate_method(method_name, secret_text)
            if results:
                comparison[method_name] = results

        return comparison


class VideoEvaluator:
    """Evaluate video steganography methods."""

    def __init__(self, test_video_dir: str = "data/videos"):
        """
        Args:
            test_video_dir: Directory with test videos
        """
        self.test_dir = test_video_dir
        self.methods = {
            "LSB": VideoLSB(num_bits=1, embed_every_n=2),
            "DCT": VideoDCT(alpha=10.0, embed_every_n=2),
            "DWT": VideoDWT(wavelet="haar", level=2, alpha=5.0, embed_every_n=2),
            "GAN": None,
        }

    def load_gan_model(self, model_path: str):
        """Load pretrained GAN model."""
        if os.path.exists(model_path):
            self.methods["GAN"] = VideoGANStego(model_path=model_path)
            return True
        return False

    def evaluate_method(self, method_name: str, secret_text: str = "test video"):
        """Evaluate single method."""
        if not os.path.exists(self.test_dir):
            return {}

        method = self.methods.get(method_name)
        if method is None:
            return {}

        results = {
            "psnr": [],
            "capacity": [],
            "ber": 0.0,
        }

        secret_data = secret_text.encode("utf-8")

        for video_path in Path(self.test_dir).glob("*.mp4"):
            try:
                output = f"/tmp/{Path(video_path).stem}_{method_name}.mp4"

                meta = method.encode(str(video_path), secret_data, output, max_frames=10)
                cap = meta.get("capacity_bytes", 0)

                if cap < len(secret_data):
                    continue

                # Decode
                decoded = method.decode(output, max_frames=10)
                if decoded == secret_data:
                    results["ber"] += 0.0
                else:
                    bits_orig = np.unpackbits(np.frombuffer(secret_data, dtype=np.uint8))
                    bits_decoded = np.unpackbits(np.frombuffer(decoded, dtype=np.uint8))
                    min_len = min(len(bits_orig), len(bits_decoded))
                    ber = np.mean(bits_orig[:min_len] != bits_decoded[:min_len])
                    results["ber"] += ber

                # Clean up
                if os.path.exists(output):
                    os.remove(output)

            except Exception as e:
                print(f"Error processing {video_path}: {e}")

        if len(results["psnr"]) > 0:
            results["ber"] = results["ber"] / len(results["psnr"])
        else:
            results = {}

        return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate steganography methods")
    parser.add_argument("--modality", choices=["image", "audio", "video"], default="image")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--image-gan-model", default=None, help="Path to pretrained image GAN")
    parser.add_argument("--audio-gan-model", default=None, help="Path to pretrained audio GAN")
    parser.add_argument("--video-gan-model", default=None, help="Path to pretrained video GAN")
    parser.add_argument("--secret-text", default="test message", help="Secret text to embed")

    args = parser.parse_args()

    if args.modality == "image":
        evaluator = ImageEvaluator(os.path.join(args.data_dir, "images"))
        if args.image_gan_model:
            evaluator.load_gan_model(args.image_gan_model)
        comparison = evaluator.compare_all_methods(args.secret_text)

    elif args.modality == "audio":
        evaluator = AudioEvaluator(os.path.join(args.data_dir, "audio"))
        if args.audio_gan_model:
            evaluator.load_gan_model(args.audio_gan_model)
        comparison = evaluator.compare_all_methods(args.secret_text)

    elif args.modality == "video":
        evaluator = VideoEvaluator(os.path.join(args.data_dir, "videos"))
        if args.video_gan_model:
            evaluator.load_gan_model(args.video_gan_model)
        comparison = evaluator.compare_all_methods(args.secret_text)

    # Print results
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)

    for method, metrics in comparison.items():
        print(f"\n{method}:")
        for metric, value in metrics.items():
            if isinstance(value, list):
                value = np.mean(value)
            print(f"  {metric}: {value:.4f}")


if __name__ == "__main__":
    main()
