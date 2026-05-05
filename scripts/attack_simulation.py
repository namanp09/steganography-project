"""
Attack Simulation Script — Steganography Robustness Evaluation

For each medium (image, audio, video) and each method (LSB, DCT, DWT, GAN):
  1. Encode a test message into synthetic cover media.
  2. Apply each attack to the stego file.
  3. Compute PSNR between stego and attacked stego.
  4. Attempt to decode the message.
  5. Compute Bit Error Rate (BER) and recovery status.

Results are printed as tables and saved to:
  outputs/attack_simulation/attack_results_image.csv
  outputs/attack_simulation/attack_results_audio.csv
  outputs/attack_simulation/attack_results_video.csv
  outputs/attack_simulation/attack_simulation_report.txt
"""

import sys
import os
import csv
import time
import numpy as np
import cv2
import soundfile as sf
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.image import ImageLSB, ImageDCT, ImageDWT
from core.audio import AudioLSB, AudioDWT
from core.video import VideoLSB, VideoDCT, VideoDWT
from core.video.frame_utils import extract_frames, reconstruct_video
from core.encryption import AESCipher
from core.attacks.image_attacks import IMAGE_ATTACKS
from core.attacks.audio_attacks import AUDIO_ATTACKS
from core.attacks.video_attacks import VIDEO_ATTACKS
from config.settings import PATHS

# ── Config ─────────────────────────────────────────────────────────────────────
TEST_MESSAGE     = "Steganography robustness test: Hello World 1234567890!@#"
TEST_PASSWORD    = "testpassword123"
# GAN capacity is limited (image GAN ≈2 bytes with ecc=5, video GAN ≈4 bytes with ecc=1)
GAN_TEST_MESSAGE = "Hi"  # 2 bytes — fits within all GAN method capacities
OUT_DIR       = Path(__file__).parent.parent / "outputs" / "attack_simulation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(42)

# ── Helpers ────────────────────────────────────────────────────────────────────

def compute_psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse < 1e-10:
        return 99.99
    import math
    v = 10 * math.log10(255.0 ** 2 / mse)
    return round(min(v, 99.99), 2)


def compute_audio_psnr(a, b):
    a, b = a.astype(np.float64).flatten(), b.astype(np.float64).flatten()
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    mse = np.mean((a - b) ** 2)
    if mse < 1e-12:
        return 99.99
    import math
    sig = np.mean(a ** 2)
    if sig < 1e-12:
        return 0.0
    v = 10 * math.log10(sig / mse)
    return round(min(v, 99.99), 2)


def compute_ber(original: bytes, decoded: bytes) -> float:
    orig_bits = np.unpackbits(np.frombuffer(original[:64], dtype=np.uint8))
    n = min(len(orig_bits), len(np.unpackbits(np.frombuffer(decoded[:64], dtype=np.uint8))))
    dec_bits  = np.unpackbits(np.frombuffer(decoded[:64],    dtype=np.uint8))[:n]
    orig_bits = orig_bits[:n]
    return float(np.sum(orig_bits != dec_bits) / n) if n > 0 else 1.0


def col(text, width):
    return str(text)[:width].ljust(width)


def print_table(title, headers, rows):
    widths = [max(len(h), max((len(str(r[i])) for r in rows), default=0)) + 2
              for i, h in enumerate(headers)]
    sep   = "+" + "+".join("-" * w for w in widths) + "+"
    hdr   = "|" + "|".join(col(h, w) for h, w in zip(headers, widths)) + "|"
    print(f"\n{'─'*len(sep)}")
    print(f"  {title}")
    print(f"{'─'*len(sep)}")
    print(sep); print(hdr); print(sep)
    for row in rows:
        print("|" + "|".join(col(v, w) for v, w in zip(row, widths)) + "|")
    print(sep)


def save_csv(path, headers, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)
    print(f"  → Saved: {path}")


# ── Synthetic test media ────────────────────────────────────────────────────────

def make_test_image(size=256):
    """Natural-looking synthetic image: gradient + texture + noise."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    for c, base in enumerate([60, 90, 120]):
        grad = np.tile(np.linspace(base, base + 120, size), (size, 1)).astype(np.uint8)
        img[:, :, c] = grad
    noise = np.random.randint(0, 30, (size, size, 3), dtype=np.uint8)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    # add some structure (circles / rectangles)
    cv2.rectangle(img, (30, 30), (100, 100), (200, 100, 50), -1)
    cv2.circle(img, (180, 80), 40, (50, 180, 200), -1)
    cv2.rectangle(img, (60, 150), (200, 220), (100, 200, 80), -1)
    return img


def make_test_audio(sr=44100, duration=3.0):
    """Multi-frequency audio signal."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = (
        0.4 * np.sin(2 * np.pi * 440  * t) +
        0.3 * np.sin(2 * np.pi * 880  * t) +
        0.2 * np.sin(2 * np.pi * 1760 * t) +
        0.1 * np.sin(2 * np.pi * 3520 * t)
    ).astype(np.float32)
    return audio, sr


def make_test_video(frames=30, h=128, w=128):
    """Synthetic video with colour gradients changing over time."""
    all_frames = []
    for i in range(frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        for c, base in enumerate([40 + i, 80 + i // 2, 120 - i // 3]):
            frame[:, :, c] = np.clip(base + np.tile(
                np.linspace(0, 80, w), (h, 1)), 0, 255).astype(np.uint8)
        noise = np.random.randint(0, 15, (h, w, 3), dtype=np.uint8)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        all_frames.append(frame)
    return all_frames


# ── Per-method encode helpers ───────────────────────────────────────────────────

def encode_image(method_obj, cover, message, password):
    cipher = AESCipher(password)
    payload = cipher.encrypt_message(message)
    return method_obj.encode(cover, payload), payload


def decode_image(method_obj, stego, password):
    raw = method_obj.decode(stego)
    cipher = AESCipher(password)
    try:
        msg = cipher.decrypt_message(raw)
        return msg, raw
    except Exception:
        return None, raw


def encode_audio(method_obj, audio, sr, message, password):
    cipher = AESCipher(password)
    payload = cipher.encrypt_message(message)
    stego_audio, _ = method_obj.encode(audio, sr, payload)
    return stego_audio, payload


def decode_audio(method_obj, audio, password):
    raw = method_obj.decode(audio)
    cipher = AESCipher(password)
    try:
        msg = cipher.decrypt_message(raw)
        return msg, raw
    except Exception:
        return None, raw


def encode_video_frames(method_obj, frames, message, password, fps=30):
    cipher = AESCipher(password)
    payload = cipher.encrypt_message(message)
    cover_path = str(OUT_DIR / "_tmp_cover.mp4")
    stego_path = str(OUT_DIR / "_tmp_stego.mp4")
    reconstruct_video(frames, cover_path, fps=fps)
    method_obj.encode(cover_path, payload, stego_path)
    stego_frames, _ = extract_frames(stego_path, max_frames=len(frames) + 5)
    return stego_frames[:len(frames)], payload


def decode_video_frames(method_obj, frames, password, fps=30):
    stego_path = str(OUT_DIR / "_tmp_stego_attack.mp4")
    reconstruct_video(frames, stego_path, fps=fps)
    raw = method_obj.decode(stego_path)
    cipher = AESCipher(password)
    try:
        msg = cipher.decrypt_message(raw)
        return msg, raw
    except Exception:
        return None, raw


# ── Simulation runners ──────────────────────────────────────────────────────────

IMAGE_METHODS = [
    ("LSB", ImageLSB(num_bits=2, seed=42),                                         "aes"),
    ("DCT", ImageDCT(alpha=10.0, seed=42),                                         "aes"),
    ("DWT", ImageDWT(wavelet="haar", level=2, alpha=5.0, seed=42),                 "aes"),
]

AUDIO_METHODS = [
    ("LSB", AudioLSB(num_bits=2, seed=42),                                         "aes"),
    ("DWT", AudioDWT(wavelet="db4", level=4, alpha=0.02, seed=42),                 "aes"),
]

VIDEO_METHODS = [
    ("LSB", VideoLSB(num_bits=2, embed_every_n=2, use_motion_comp=False, seed=42), "aes"),
    ("DCT", VideoDCT(alpha=30.0, embed_every_n=2, use_motion_comp=False, seed=42), "aes"),
    ("DWT", VideoDWT(wavelet="haar", level=2, alpha=10.0, embed_every_n=2,
                     use_motion_comp=False, seed=42),                               "aes"),
]

HEADERS = ["Method", "Attack", "PSNR Stego→Attack (dB)", "PSNR Cover→Attack (dB)", "BER (Robustness ↓)", "Time (ms)"]


# ── GAN loader helpers ──────────────────────────────────────────────────────────

def _ckpt(name):
    p = Path(PATHS.models_dir) / name / "best_model.pth"
    return str(p) if p.exists() else None


def load_image_gan():
    try:
        from core.image.gan_stego import ImageGANStego
        path = _ckpt("image_gan_improved")
        obj = ImageGANStego(model_path=path, device="cpu", ecc_factor=5)
        print(f"    [GAN] Image model loaded — checkpoint: {'found' if path else 'MISSING (random weights)'}")
        return obj
    except Exception as e:
        print(f"    [GAN] Image model FAILED to load: {e}")
        return None


def load_audio_gan():
    try:
        from core.audio.gan_stego import AudioGANStego
        path = _ckpt("audio_gan_improved")
        obj = AudioGANStego(model_path=path, device="cpu", ecc_factor=1)
        print(f"    [GAN] Audio model loaded — checkpoint: {'found' if path else 'MISSING (random weights)'}")
        return obj
    except Exception as e:
        print(f"    [GAN] Audio model FAILED to load: {e}")
        return None


def load_video_gan():
    try:
        from core.video.gan_stego import VideoGANStego
        path = _ckpt("video_gan_improved")
        obj = VideoGANStego(model_path=path, device="cpu", ecc_factor=1)
        print(f"    [GAN] Video model loaded — checkpoint: {'found' if path else 'MISSING (random weights)'}")
        return obj
    except Exception as e:
        print(f"    [GAN] Video model FAILED to load: {e}")
        return None


# ── GAN-specific encode/decode (no AES — raw UTF-8) ────────────────────────────

def encode_image_gan(gan, cover, message):
    payload = message.encode("utf-8")
    stego = gan.encode(cover, payload)
    return stego, payload


def decode_image_gan(gan, stego, original_msg):
    try:
        raw = gan.decode(stego)
        decoded = raw.rstrip(b"\x00").decode("utf-8", errors="replace")
        recovered = "YES" if decoded == original_msg else "NO"
        ber = compute_ber(original_msg.encode("utf-8"), raw)
        return recovered, ber
    except Exception:
        return "ERROR", 1.0


def encode_audio_gan(gan, audio, sr, message):
    payload = message.encode("utf-8")
    stego_audio, _ = gan.encode(audio, sr, payload)
    return stego_audio, payload


def decode_audio_gan(gan, audio, original_msg):
    try:
        raw = gan.decode(audio)
        decoded = raw.rstrip(b"\x00").decode("utf-8", errors="replace")
        recovered = "YES" if decoded == original_msg else "NO"
        ber = compute_ber(original_msg.encode("utf-8"), raw)
        return recovered, ber
    except Exception:
        return "ERROR", 1.0


def encode_video_gan(gan, frames, message, fps=30):
    payload = message.encode("utf-8")
    cover_path = str(OUT_DIR / "_tmp_gan_cover.mp4")
    stego_path = str(OUT_DIR / "_tmp_gan_stego.mp4")
    reconstruct_video(frames, cover_path, fps=fps)
    gan.encode(cover_path, payload, stego_path, max_frames=len(frames))
    stego_frames, _ = extract_frames(stego_path, max_frames=len(frames) + 5)
    return stego_frames[:len(frames)], payload


def decode_video_gan(gan, frames, original_msg, fps=30):
    stego_path = str(OUT_DIR / "_tmp_gan_attack.mp4")
    reconstruct_video(frames, stego_path, fps=fps)
    try:
        raw = gan.decode(stego_path)
        decoded = raw.rstrip(b"\x00").decode("utf-8", errors="replace")
        recovered = "YES" if decoded == original_msg else "NO"
        ber = compute_ber(original_msg.encode("utf-8"), raw)
        return recovered, ber
    except Exception:
        return "ERROR", 1.0


def run_image_simulation():
    print("\n" + "="*70)
    print("  IMAGE ATTACK SIMULATION")
    print("="*70)
    cover = make_test_image(256)
    all_rows = []

    # ── Traditional methods ───────────────────────────────────────────────────
    for method_name, method_obj, mode in IMAGE_METHODS:
        print(f"\n  Method: {method_name}")
        try:
            stego, payload = encode_image(method_obj, cover, TEST_MESSAGE, TEST_PASSWORD)
        except Exception as e:
            print(f"    [ENCODE FAILED] {e}")
            continue

        embed_psnr = compute_psnr(cover, stego)
        rows = []
        for attack_name, attack_fn in IMAGE_ATTACKS:
            t0 = time.time()
            try:
                attacked = attack_fn(stego)
                psnr_s2a = compute_psnr(stego, attacked)
                psnr_c2a = compute_psnr(cover, attacked)
                decoded_msg, raw = decode_image(method_obj, attacked, TEST_PASSWORD)
                ber = 0.0 if decoded_msg == TEST_MESSAGE else (compute_ber(payload, raw) if raw else 1.0)
            except Exception:
                psnr_s2a, psnr_c2a, ber = 0.0, 0.0, 1.0
            elapsed = round((time.time() - t0) * 1000, 1)
            row = [method_name, attack_name, psnr_s2a, psnr_c2a, round(ber, 4), elapsed]
            rows.append(row); all_rows.append(row)
        print_table(f"Image — {method_name}  [embed PSNR: {embed_psnr} dB]", HEADERS, rows)

    # ── GAN ───────────────────────────────────────────────────────────────────
    print("\n  Method: GAN")
    gan = load_image_gan()
    if gan is not None:
        try:
            stego, payload = encode_image_gan(gan, cover, GAN_TEST_MESSAGE)
            embed_psnr = compute_psnr(cover, stego)
            rows = []
            for attack_name, attack_fn in IMAGE_ATTACKS:
                t0 = time.time()
                try:
                    attacked = attack_fn(stego)
                    psnr_s2a = compute_psnr(stego, attacked)
                    psnr_c2a = compute_psnr(cover, attacked)
                    _, ber = decode_image_gan(gan, attacked, GAN_TEST_MESSAGE)
                except Exception:
                    psnr_s2a, psnr_c2a, ber = 0.0, 0.0, 1.0
                elapsed = round((time.time() - t0) * 1000, 1)
                row = ["GAN", attack_name, psnr_s2a, psnr_c2a, round(ber, 4), elapsed]
                rows.append(row); all_rows.append(row)
            print_table(f"Image — GAN  [embed PSNR: {embed_psnr} dB]", HEADERS, rows)
        except Exception as e:
            print(f"    [GAN ENCODE FAILED] {e}")
        finally:
            del gan
            import gc; gc.collect()
    else:
        print("    [GAN SKIPPED — model unavailable]")

    save_csv(OUT_DIR / "attack_results_image.csv", HEADERS, all_rows)
    return all_rows


def run_audio_simulation():
    print("\n" + "="*70)
    print("  AUDIO ATTACK SIMULATION")
    print("="*70)
    cover_audio, sr = make_test_audio()
    all_rows = []

    # ── Traditional methods ───────────────────────────────────────────────────
    for method_name, method_obj, mode in AUDIO_METHODS:
        print(f"\n  Method: {method_name}")
        try:
            stego_audio, payload = encode_audio(method_obj, cover_audio, sr, TEST_MESSAGE, TEST_PASSWORD)
        except Exception as e:
            print(f"    [ENCODE FAILED] {e}")
            continue

        embed_psnr = compute_audio_psnr(cover_audio, stego_audio)
        rows = []
        for attack_name, attack_fn in AUDIO_ATTACKS:
            t0 = time.time()
            try:
                attacked = attack_fn(stego_audio, sr)
                psnr_s2a = compute_audio_psnr(stego_audio, attacked)
                psnr_c2a = compute_audio_psnr(cover_audio, attacked)
                decoded_msg, raw = decode_audio(method_obj, attacked, TEST_PASSWORD)
                ber = 0.0 if decoded_msg == TEST_MESSAGE else (compute_ber(payload, raw) if raw else 1.0)
            except Exception:
                psnr_s2a, psnr_c2a, ber = 0.0, 0.0, 1.0
            elapsed = round((time.time() - t0) * 1000, 1)
            row = [method_name, attack_name, psnr_s2a, psnr_c2a, round(ber, 4), elapsed]
            rows.append(row); all_rows.append(row)
        print_table(f"Audio — {method_name}  [embed PSNR: {embed_psnr} dB]", HEADERS, rows)

    # ── GAN ───────────────────────────────────────────────────────────────────
    print("\n  Method: GAN")
    gan = load_audio_gan()
    if gan is not None:
        try:
            stego_audio, payload = encode_audio_gan(gan, cover_audio, sr, GAN_TEST_MESSAGE)
            embed_psnr = compute_audio_psnr(cover_audio, stego_audio)
            rows = []
            for attack_name, attack_fn in AUDIO_ATTACKS:
                t0 = time.time()
                try:
                    attacked = attack_fn(stego_audio, sr)
                    psnr_s2a = compute_audio_psnr(stego_audio, attacked)
                    psnr_c2a = compute_audio_psnr(cover_audio, attacked)
                    _, ber = decode_audio_gan(gan, attacked, GAN_TEST_MESSAGE)
                except Exception:
                    psnr_s2a, psnr_c2a, ber = 0.0, 0.0, 1.0
                elapsed = round((time.time() - t0) * 1000, 1)
                row = ["GAN", attack_name, psnr_s2a, psnr_c2a, round(ber, 4), elapsed]
                rows.append(row); all_rows.append(row)
            print_table(f"Audio — GAN  [embed PSNR: {embed_psnr} dB]", HEADERS, rows)
        except Exception as e:
            print(f"    [GAN ENCODE FAILED] {e}")
        finally:
            del gan
            import gc; gc.collect()
    else:
        print("    [GAN SKIPPED — model unavailable]")

    save_csv(OUT_DIR / "attack_results_audio.csv", HEADERS, all_rows)
    return all_rows


def run_video_simulation():
    print("\n" + "="*70)
    print("  VIDEO ATTACK SIMULATION")
    print("="*70)
    cover_frames = make_test_video(frames=30, h=128, w=128)
    all_rows = []

    # ── Traditional methods ───────────────────────────────────────────────────
    for method_name, method_obj, mode in VIDEO_METHODS:
        print(f"\n  Method: {method_name}")
        try:
            stego_frames, payload = encode_video_frames(method_obj, cover_frames, TEST_MESSAGE, TEST_PASSWORD)
        except Exception as e:
            print(f"    [ENCODE FAILED] {e}")
            continue

        n = min(len(cover_frames), len(stego_frames))
        embed_psnr = round(np.mean([compute_psnr(cover_frames[i], stego_frames[i]) for i in range(n)]), 2)
        rows = []
        for attack_name, attack_fn in VIDEO_ATTACKS:
            t0 = time.time()
            try:
                attacked_frames = attack_fn(stego_frames)
                psnr_s2a = round(np.mean([compute_psnr(stego_frames[i], attacked_frames[i]) for i in range(n)]), 2)
                psnr_c2a = round(np.mean([compute_psnr(cover_frames[i], attacked_frames[i]) for i in range(n)]), 2)
                decoded_msg, raw = decode_video_frames(method_obj, attacked_frames, TEST_PASSWORD)
                ber = 0.0 if decoded_msg == TEST_MESSAGE else (compute_ber(payload, raw) if raw else 1.0)
            except Exception:
                psnr_s2a, psnr_c2a, ber = 0.0, 0.0, 1.0
            elapsed = round((time.time() - t0) * 1000, 1)
            row = [method_name, attack_name, psnr_s2a, psnr_c2a, round(ber, 4), elapsed]
            rows.append(row); all_rows.append(row)
        print_table(f"Video — {method_name}  [embed PSNR: {embed_psnr} dB]", HEADERS, rows)

    # ── GAN ───────────────────────────────────────────────────────────────────
    print("\n  Method: GAN")
    gan = load_video_gan()
    if gan is not None:
        try:
            stego_frames, payload = encode_video_gan(gan, cover_frames, GAN_TEST_MESSAGE)
            n = min(len(cover_frames), len(stego_frames))
            embed_psnr = round(np.mean([compute_psnr(cover_frames[i], stego_frames[i]) for i in range(n)]), 2)
            rows = []
            for attack_name, attack_fn in VIDEO_ATTACKS:
                t0 = time.time()
                try:
                    attacked_frames = attack_fn(stego_frames)
                    psnr_s2a = round(np.mean([compute_psnr(stego_frames[i], attacked_frames[i]) for i in range(n)]), 2)
                    psnr_c2a = round(np.mean([compute_psnr(cover_frames[i], attacked_frames[i]) for i in range(n)]), 2)
                    _, ber = decode_video_gan(gan, attacked_frames, GAN_TEST_MESSAGE)
                except Exception:
                    psnr_s2a, psnr_c2a, ber = 0.0, 0.0, 1.0
                elapsed = round((time.time() - t0) * 1000, 1)
                row = ["GAN", attack_name, psnr_s2a, psnr_c2a, round(ber, 4), elapsed]
                rows.append(row); all_rows.append(row)
            print_table(f"Video — GAN  [embed PSNR: {embed_psnr} dB]", HEADERS, rows)
        except Exception as e:
            print(f"    [GAN ENCODE FAILED] {e}")
        finally:
            del gan
            import gc; gc.collect()
    else:
        print("    [GAN SKIPPED — model unavailable]")

    save_csv(OUT_DIR / "attack_results_video.csv", HEADERS, all_rows)
    return all_rows


# ── Charts ──────────────────────────────────────────────────────────────────────

def generate_charts(img_rows, audio_rows, video_rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        def plot_group(rows, title, out_file):
            if not rows:
                return
            methods = list(dict.fromkeys(r[0] for r in rows))
            attacks = list(dict.fromkeys(r[1] for r in rows if r[1] != "No Attack (Baseline)"))

            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            fig.suptitle(title, fontsize=14, fontweight="bold")
            colors = ["#3b82f6", "#8b5cf6", "#f59e0b", "#10b981"]

            # PSNR chart
            ax = axes[0]
            x = np.arange(len(attacks))
            w = 0.25
            for i, method in enumerate(methods):
                psnrs = []
                for atk in attacks:
                    match = [r[2] for r in rows if r[0] == method and r[1] == atk]
                    psnrs.append(match[0] if match else 0)
                ax.bar(x + i * w, psnrs, w, label=method, color=colors[i % len(colors)], alpha=0.85)
            ax.set_xticks(x + w)
            ax.set_xticklabels(attacks, rotation=35, ha="right", fontsize=8)
            ax.set_ylabel("PSNR (dB)")
            ax.set_title("PSNR After Attack (stego → attacked)")
            ax.legend()
            ax.grid(axis="y", alpha=0.3)

            # Robustness chart (BER — lower is better)
            ax2 = axes[1]
            for i, method in enumerate(methods):
                bers = []
                for atk in attacks:
                    match = [r[4] for r in rows if r[0] == method and r[1] == atk]
                    bers.append(match[0] if match else 0.5)
                ax2.bar(x + i * w, bers, w, label=method, color=colors[i % len(colors)], alpha=0.85)
            ax2.set_xticks(x + w)
            ax2.set_xticklabels(attacks, rotation=35, ha="right", fontsize=8)
            ax2.set_ylabel("BER (lower = more robust)")
            ax2.set_ylim(0, 0.6)
            ax2.set_title("Bit Error Rate After Attack (BER ↓ better)")
            ax2.axhline(0.5, color="red", linestyle="--", linewidth=0.8, alpha=0.6, label="Random (BER=0.5)")
            ax2.legend()
            ax2.grid(axis="y", alpha=0.3)

            plt.tight_layout()
            plt.savefig(out_file, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  → Chart saved: {out_file}")

        plot_group(img_rows,   "Image — Attack Simulation",  OUT_DIR / "chart_image.png")
        plot_group(audio_rows, "Audio — Attack Simulation",  OUT_DIR / "chart_audio.png")
        plot_group(video_rows, "Video — Attack Simulation",  OUT_DIR / "chart_video.png")

    except Exception as e:
        print(f"  [Chart generation failed: {e}]")


# ── Summary report ──────────────────────────────────────────────────────────────

def write_report(img_rows, audio_rows, video_rows):
    lines = [
        "=" * 70,
        "ATTACK SIMULATION SUMMARY REPORT",
        f"Metrics: PSNR (higher = less distortion)  |  BER (lower = more robust)",
        "=" * 70,
    ]

    def section(name, rows):
        lines.append(f"\n── {name} ──")
        lines.append(f"  {'Method':<8}  {'Embed PSNR':>12}  {'Avg PSNR after attack':>22}  {'Avg BER under attack':>21}")
        lines.append(f"  {'-'*8}  {'-'*12}  {'-'*22}  {'-'*21}")
        methods = list(dict.fromkeys(r[0] for r in rows))
        for method in methods:
            m_rows = [r for r in rows if r[0] == method]
            baseline = [r for r in m_rows if "Baseline" in r[1]]
            embed_psnr = baseline[0][3] if baseline else "—"
            attacked = [r for r in m_rows if "Baseline" not in r[1]]
            avg_psnr = round(np.mean([r[2] for r in attacked if isinstance(r[2], (int, float))]), 2) if attacked else "—"
            avg_ber  = round(np.mean([r[4] for r in attacked if isinstance(r[4], (int, float))]), 4) if attacked else "—"
            lines.append(f"  {method:<8}  {str(embed_psnr)+' dB':>12}  {str(avg_psnr)+' dB':>22}  {str(avg_ber):>21}")

    section("IMAGE",  img_rows)
    section("AUDIO",  audio_rows)
    section("VIDEO",  video_rows)

    lines.append("\nInterpretation:")
    lines.append("  Embed PSNR  — quality of stego vs cover (higher = less visible change)")
    lines.append("  PSNR after attack — signal quality surviving the attack (higher = better)")
    lines.append("  BER under attack  — bit error rate (lower = more robust; 0.5 = random noise)")
    lines.append("")
    lines.append("  LSB  : high embed PSNR but fragile — BER spikes near 0.5 under most attacks.")
    lines.append("  DCT  : more robust to JPEG/brightness attacks due to frequency-domain embedding.")
    lines.append("  DWT  : similar to DCT; survives resampling and mild compression attacks.")
    lines.append("  GAN  : adversarially trained with noise layers; lower BER under mild attacks")

    report_path = OUT_DIR / "attack_simulation_report.txt"
    report_path.write_text("\n".join(lines))
    print(f"\n  → Report saved: {report_path}")
    print("\n".join(lines))


# ── Main ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "█"*70)
    print("  STEGANOGRAPHY ATTACK SIMULATION")
    print("  Test message:", TEST_MESSAGE)
    print("█"*70)

    img_rows   = run_image_simulation()
    audio_rows = run_audio_simulation()
    video_rows = run_video_simulation()

    print("\n" + "="*70)
    print("  GENERATING CHARTS...")
    print("="*70)
    generate_charts(img_rows, audio_rows, video_rows)

    write_report(img_rows, audio_rows, video_rows)

    print("\n✓ Simulation complete. Results in:", OUT_DIR)
