#!/usr/bin/env python3
"""
Complete GAN encode/decode test with multiple messages and images.
Verifies full functionality.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path.home() / "Desktop" / "steganography-project"))

import cv2
import numpy as np
from core.encryption import AESCipher
from core.image import ImageGANStego

def test_image_gan():
    """Test Image GAN with multiple scenarios."""
    print("\n" + "="*80)
    print("IMAGE GAN ENCODE/DECODE TEST")
    print("="*80)

    device = "cpu"

    # Try improved model first, fallback to quickstart
    model_path = "models/checkpoints/image_gan_improved/best_model.pth"
    if not Path(model_path).exists():
        print(f"⚠ Improved model not found, using quickstart")
        model_path = "models/checkpoints/image_gan_quickstart/best_model.pth"

    gan = ImageGANStego(model_path=model_path, device=device)
    print(f"✓ Model loaded: {model_path}")

    # Test messages
    test_messages = [
        "Hi",
        "Hello World!",
        "Secret123",
        "This is a longer message to test capacity",
    ]

    results = []

    for i, msg in enumerate(test_messages, 1):
        print(f"\n[Test {i}] Message: '{msg}'")

        # Create test image
        img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)

        # Encrypt
        cipher = AESCipher("testpass123")
        encrypted = cipher.encrypt_message(msg)
        print(f"  Encrypted: {len(encrypted)} bytes")

        # Encode
        try:
            stego = gan.encode(img, encrypted)
            print(f"  ✓ Encoded: {stego.shape}")
        except Exception as e:
            print(f"  ✗ Encode failed: {e}")
            results.append((msg, False, str(e)))
            continue

        # Decode
        try:
            decrypted_enc = gan.decode(stego)
            decrypted = cipher.decrypt_message(decrypted_enc)

            if decrypted == msg:
                print(f"  ✓ Decoded: '{decrypted}'")
                results.append((msg, True, "Success"))
            else:
                print(f"  ✗ Mismatch: expected '{msg}', got '{decrypted}'")
                results.append((msg, False, f"Decoded: {decrypted}"))
        except Exception as e:
            print(f"  ✗ Decode failed: {e}")
            results.append((msg, False, str(e)))

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    passed = sum(1 for _, success, _ in results if success)
    total = len(results)

    for msg, success, detail in results:
        status = "✓" if success else "✗"
        print(f"{status} '{msg[:30]}': {detail}")

    print(f"\nPassed: {passed}/{total}")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED! GAN is working perfectly!")
        return True
    else:
        print(f"\n⚠ {total - passed} tests failed")
        return False


if __name__ == "__main__":
    success = test_image_gan()
    sys.exit(0 if success else 1)
