"""
Video attack functions for steganography robustness evaluation.
Each function takes a list of BGR uint8 frames and returns the attacked frame list.
"""

import numpy as np
import cv2


def no_attack(frames):
    return [f.copy() for f in frames]


def gaussian_noise_10(frames):
    out = []
    for f in frames:
        noise = np.random.normal(0, 10, f.shape).astype(np.float32)
        out.append(np.clip(f.astype(np.float32) + noise, 0, 255).astype(np.uint8))
    return out


def jpeg_q75(frames):
    out = []
    for f in frames:
        _, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 75])
        out.append(cv2.imdecode(buf, cv2.IMREAD_COLOR))
    return out


def jpeg_q50(frames):
    out = []
    for f in frames:
        _, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 50])
        out.append(cv2.imdecode(buf, cv2.IMREAD_COLOR))
    return out


def resize_half(frames):
    out = []
    for f in frames:
        h, w = f.shape[:2]
        small = cv2.resize(f, (w // 2, h // 2))
        out.append(cv2.resize(small, (w, h)))
    return out


def frame_drop(frames):
    # Drop every other frame, duplicate preceding frame to maintain count
    out = []
    for i, f in enumerate(frames):
        if i % 2 == 0:
            out.append(f.copy())
        else:
            out.append(frames[i - 1].copy())
    return out


def brightness_plus20(frames):
    out = []
    for f in frames:
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] + 20, 0, 255)
        out.append(cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR))
    return out


def gaussian_blur(frames):
    return [cv2.GaussianBlur(f, (3, 3), 0) for f in frames]


def salt_pepper(frames):
    out = []
    for f in frames:
        attacked = f.copy()
        rnd = np.random.random(f.shape[:2])
        attacked[rnd < 0.005] = 0
        attacked[rnd > 0.995] = 255
        out.append(attacked)
    return out


VIDEO_ATTACKS = [
    ("No Attack (Baseline)",        no_attack),
    ("Gaussian Noise (σ=10)",       gaussian_noise_10),
    ("JPEG Frames Q=75",            jpeg_q75),
    ("JPEG Frames Q=50",            jpeg_q50),
    ("Resize 50% + Rescale",        resize_half),
    ("Frame Dropping (50%)",        frame_drop),
    ("Brightness +20",              brightness_plus20),
    ("Gaussian Blur (3×3)",         gaussian_blur),
    ("Salt & Pepper (p=0.01)",      salt_pepper),
]
