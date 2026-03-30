"""
Download standard test datasets for steganography evaluation.

Downloads:
- Standard test images (Lenna, Baboon, Peppers, etc.)
- Sample WAV audio files
- Sample MP4 video clips

Usage: python scripts/download_test_data.py
"""

import os
import requests
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def download_file(url: str, save_path: str, desc: str = ""):
    """Download file with progress bar."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if os.path.exists(save_path):
        print(f"  [SKIP] {desc} already exists")
        return

    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))

        with open(save_path, "wb") as f:
            with tqdm(total=total, unit="B", unit_scale=True, desc=desc) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        print(f"  [OK] {desc}")
    except Exception as e:
        print(f"  [FAIL] {desc}: {e}")


def download_test_images():
    """Download standard steganography test images."""
    print("\n=== Downloading Test Images ===")
    img_dir = DATA_DIR / "images"

    # Standard test images from public sources
    images = {
        "lenna.png": "https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png",
        "baboon.png": "https://www.hlevkin.com/hlevkin/TestImages/baboon.bmp",
        "peppers.png": "https://www.hlevkin.com/hlevkin/TestImages/peppers.bmp",
    }

    for name, url in images.items():
        download_file(url, str(img_dir / name), name)


def create_test_audio():
    """Create synthetic test audio files for testing."""
    print("\n=== Creating Test Audio Files ===")
    audio_dir = DATA_DIR / "audio"
    os.makedirs(audio_dir, exist_ok=True)

    try:
        import numpy as np
        import soundfile as sf

        sr = 44100
        duration = 5  # seconds

        # Sine wave
        t = np.linspace(0, duration, sr * duration)
        sine = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float64)
        sf.write(str(audio_dir / "test_sine.wav"), sine, sr)
        print("  [OK] test_sine.wav (440Hz sine, 5s)")

        # White noise
        noise = (np.random.randn(sr * duration) * 0.3).astype(np.float64)
        sf.write(str(audio_dir / "test_noise.wav"), noise, sr)
        print("  [OK] test_noise.wav (white noise, 5s)")

        # Music-like (multiple harmonics)
        music = np.zeros(sr * duration)
        for freq in [261.63, 329.63, 392.00, 523.25]:  # C major chord
            music += 0.2 * np.sin(2 * np.pi * freq * t)
        sf.write(str(audio_dir / "test_music.wav"), music.astype(np.float64), sr)
        print("  [OK] test_music.wav (C major chord, 5s)")

    except ImportError:
        print("  [SKIP] soundfile not installed, skipping audio generation")


def create_test_video():
    """Create synthetic test video for testing."""
    print("\n=== Creating Test Video Files ===")
    video_dir = DATA_DIR / "video"
    os.makedirs(video_dir, exist_ok=True)

    try:
        import cv2
        import numpy as np

        # Create a simple animated test video
        fps = 30
        duration = 5
        width, height = 512, 512
        n_frames = fps * duration

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        path = str(video_dir / "test_pattern.mp4")
        writer = cv2.VideoWriter(path, fourcc, fps, (width, height))

        for i in range(n_frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            # Moving gradient
            offset = int((i / n_frames) * width)
            for x in range(width):
                val = int(((x + offset) % width) / width * 255)
                frame[:, x, :] = val
            # Add some shapes
            cv2.circle(frame, (256 + int(100 * np.sin(i * 0.1)), 256), 50, (0, 0, 255), -1)
            cv2.rectangle(frame, (100, 100 + i % 100), (200, 200 + i % 100), (0, 255, 0), 2)
            writer.write(frame)

        writer.release()
        print(f"  [OK] test_pattern.mp4 ({width}x{height}, {fps}fps, {duration}s)")

        # Static scene video (better for steganography testing)
        path2 = str(video_dir / "test_static.mp4")
        writer2 = cv2.VideoWriter(path2, fourcc, fps, (width, height))

        # Generate a static complex image
        base = np.random.randint(100, 200, (height, width, 3), dtype=np.uint8)
        base = cv2.GaussianBlur(base, (21, 21), 5)

        for i in range(n_frames):
            # Small random variations (simulates camera noise)
            noise = np.random.randint(-2, 3, (height, width, 3), dtype=np.int16)
            frame = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            writer2.write(frame)

        writer2.release()
        print(f"  [OK] test_static.mp4 (static scene with noise, {duration}s)")

    except ImportError:
        print("  [SKIP] opencv-python not installed, skipping video generation")


if __name__ == "__main__":
    print("=" * 60)
    print("  Downloading/Creating Test Data for Steganography System")
    print("=" * 60)

    download_test_images()
    create_test_audio()
    create_test_video()

    print("\n" + "=" * 60)
    print("  Done! Test data is in:", DATA_DIR)
    print("=" * 60)
