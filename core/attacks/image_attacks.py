"""
Image attack functions for steganography robustness evaluation.
Each function takes a BGR uint8 numpy array and returns the attacked version.
"""

import numpy as np
import cv2
from io import BytesIO


def no_attack(img):
    return img.copy()


def jpeg_q90(img):
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def jpeg_q75(img):
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def jpeg_q50(img):
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 50])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def gaussian_noise_5(img):
    noise = np.random.normal(0, 5, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def gaussian_noise_15(img):
    noise = np.random.normal(0, 15, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def salt_pepper(img):
    out = img.copy()
    prob = 0.01
    rnd = np.random.random(img.shape[:2])
    out[rnd < prob / 2] = 0
    out[rnd > 1 - prob / 2] = 255
    return out


def gaussian_blur(img):
    return cv2.GaussianBlur(img, (3, 3), 0)


def median_filter(img):
    return cv2.medianBlur(img, 3)


def resize_half(img):
    h, w = img.shape[:2]
    small = cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def brightness_plus20(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] + 20, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def rotation_1deg(img):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 1, 1)
    return cv2.warpAffine(img, M, (w, h))


def crop_5pct(img):
    h, w = img.shape[:2]
    cy, cx = int(h * 0.05), int(w * 0.05)
    cropped = img[cy: h - cy, cx: w - cx]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)


# Registry — (label, function)
IMAGE_ATTACKS = [
    ("No Attack (Baseline)",      no_attack),
    ("JPEG Compression Q=90",     jpeg_q90),
    ("JPEG Compression Q=75",     jpeg_q75),
    ("JPEG Compression Q=50",     jpeg_q50),
    ("Gaussian Noise (σ=5)",      gaussian_noise_5),
    ("Gaussian Noise (σ=15)",     gaussian_noise_15),
    ("Salt & Pepper (p=0.01)",    salt_pepper),
    ("Gaussian Blur (3×3)",       gaussian_blur),
    ("Median Filter (3×3)",       median_filter),
    ("Resize 50% + Rescale",      resize_half),
    ("Brightness +20",            brightness_plus20),
    ("Rotation 1°",               rotation_1deg),
    ("Crop 5% + Rescale",         crop_5pct),
]
