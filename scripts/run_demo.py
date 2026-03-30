"""
Demo script — Demonstrates all steganography methods with metrics comparison.

Usage: python scripts/run_demo.py
"""

import os
import sys
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.encryption import AESCipher, compute_hash, verify_hash
from core.image import ImageLSB, ImageDCT, ImageDWT
from core.audio import AudioLSB, AudioDWT
from core.metrics import compute_all_metrics, compare_methods, plot_metrics
from config.settings import PATHS


def demo_image_steganography():
    """Run image steganography demo with all methods."""
    print("\n" + "=" * 60)
    print("  IMAGE STEGANOGRAPHY DEMO")
    print("=" * 60)

    # Create a test image if none exists
    img_path = os.path.join(PATHS.data_dir, "images", "lenna.png")
    if not os.path.exists(img_path):
        print("Creating synthetic test image...")
        cover = np.random.randint(100, 200, (512, 512, 3), dtype=np.uint8)
        cover = cv2.GaussianBlur(cover, (5, 5), 2)
    else:
        cover = cv2.imread(img_path)
        cover = cv2.resize(cover, (512, 512))

    print(f"Cover image: {cover.shape}")

    # Secret message
    secret_message = "This is a secret message for the steganography demo! 🔒"
    password = "my_secure_password_123"

    # Encrypt
    cipher = AESCipher(password)
    encrypted = cipher.encrypt_message(secret_message)
    msg_hash = compute_hash(encrypted)
    print(f"Message encrypted: {len(encrypted)} bytes, SHA-256: {msg_hash[:16]}...")

    # Methods to test
    methods = {
        "LSB (2-bit)": ImageLSB(num_bits=2, seed=42),
        "DCT (α=10)": ImageDCT(alpha=10.0, seed=42),
        "DWT (Haar, L2)": ImageDWT(wavelet="haar", level=2, alpha=5.0, seed=42),
    }

    stego_results = {}
    extracted_msgs = {}

    for name, method in methods.items():
        print(f"\n--- {name} ---")
        try:
            # Encode
            stego = method.encode(cover, encrypted)
            stego_results[name] = stego

            # Decode
            extracted = method.decode(stego)
            decrypted = cipher.decrypt_message(extracted)
            extracted_msgs[name] = extracted

            # Verify
            success = decrypted == secret_message
            print(f"  Extraction: {'✓ SUCCESS' if success else '✗ FAILED'}")
            if success:
                print(f"  Decoded: \"{decrypted[:50]}...\"")

            # Save stego image
            out_path = os.path.join(PATHS.output_dir, f"demo_stego_{name.replace(' ', '_')}.png")
            cv2.imwrite(out_path, stego)

        except Exception as e:
            print(f"  ERROR: {e}")

    # Compare metrics
    print("\n\n" + "=" * 60)
    print("  METRICS COMPARISON")
    print("=" * 60)

    comparison = compare_methods(cover, stego_results, encrypted, extracted_msgs)

    # Print table
    header = f"{'Method':<20}"
    metrics_names = list(next(iter(comparison.values())).keys())
    for m in metrics_names:
        header += f"{m:>12}"
    print(header)
    print("-" * len(header))

    for method, metrics in comparison.items():
        row = f"{method:<20}"
        for m in metrics_names:
            row += f"{metrics.get(m, 'N/A'):>12}"
        print(row)

    # Save comparison chart
    chart_path = os.path.join(PATHS.output_dir, "demo_comparison.png")
    plot_metrics(comparison, save_path=chart_path)
    print(f"\nComparison chart saved: {chart_path}")


def demo_encryption():
    """Demo the encryption module."""
    print("\n" + "=" * 60)
    print("  AES-256-GCM ENCRYPTION DEMO")
    print("=" * 60)

    password = "secure_password_2025"
    cipher = AESCipher(password)

    message = "Top secret information that needs to be hidden!"
    print(f"Original: \"{message}\"")

    encrypted = cipher.encrypt_message(message)
    print(f"Encrypted: {len(encrypted)} bytes")
    print(f"SHA-256: {compute_hash(encrypted)}")

    decrypted = cipher.decrypt_message(encrypted)
    print(f"Decrypted: \"{decrypted}\"")
    print(f"Match: {'✓' if message == decrypted else '✗'}")

    # Test tamper detection
    tampered = bytearray(encrypted)
    tampered[-1] ^= 0xFF
    try:
        cipher.decrypt(bytes(tampered))
        print("Tamper detection: ✗ FAILED")
    except ValueError:
        print("Tamper detection: ✓ GCM authentication correctly detected tampering")


if __name__ == "__main__":
    os.makedirs(PATHS.output_dir, exist_ok=True)
    demo_encryption()
    demo_image_steganography()
    print("\n\nAll demos completed! Outputs in:", PATHS.output_dir)
