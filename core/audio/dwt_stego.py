"""
Audio DWT Steganography — Transform Domain Method.

Embeds data in DWT detail coefficients of audio signals.
Uses multi-level Daubechies wavelet decomposition for robustness.
Implements adaptive quantization based on coefficient energy.
"""

import numpy as np
import pywt
from typing import Optional, Tuple


class AudioDWT:
    """DWT-based audio steganography with adaptive embedding."""

    def __init__(
        self,
        wavelet: str = "db4",
        level: int = 4,
        alpha: float = 0.02,
        seed: Optional[int] = None,
    ):
        """
        Args:
            wavelet: Wavelet family (db4 has good freq. localization for audio).
            level: Decomposition levels.
            alpha: Embedding strength factor.
            seed: Random seed for coefficient ordering.
        """
        self.wavelet = wavelet
        self.level = level
        self.alpha = alpha
        self.seed = seed

    def capacity(self, audio: np.ndarray) -> int:
        """Return capacity in bytes."""
        coeffs = pywt.wavedec(audio.flatten(), self.wavelet, level=self.level)
        # Use detail coefficients at level 1 (coarsest details)
        n_coeffs = len(coeffs[1])
        return (n_coeffs // 8) - 4

    def encode(
        self, audio: np.ndarray, sr: int, secret_data: bytes
    ) -> Tuple[np.ndarray, int]:
        """Embed data in DWT detail coefficients of audio."""
        original_shape = audio.shape
        signal = audio.flatten().astype(np.float64)

        # Multi-level DWT
        coeffs = pywt.wavedec(signal, self.wavelet, level=self.level)

        # Embed in level-1 detail coefficients (most robust)
        detail = coeffs[1].copy()

        # Prepare payload
        length_header = len(secret_data).to_bytes(4, "big")
        payload = length_header + secret_data
        bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))

        if len(bits) > len(detail):
            raise ValueError(
                f"Data too large: {len(bits)} bits, {len(detail)} coefficients available"
            )

        indices = np.arange(len(detail))
        if self.seed is not None:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(indices)

        # Compute local energy for adaptive strength
        window = 32
        energy = np.array([
            np.mean(detail[max(0, i - window):i + window] ** 2)
            for i in range(len(detail))
        ])
        energy = np.maximum(energy, 1e-10)

        for i, bit in enumerate(bits):
            idx = indices[i]
            strength = self.alpha * np.sqrt(energy[idx])

            coeff = detail[idx]
            q = np.round(coeff / strength) * strength

            if bit == 1:
                detail[idx] = q + strength / 2
            else:
                detail[idx] = q - strength / 2

        coeffs[1] = detail

        # Inverse DWT
        reconstructed = pywt.waverec(coeffs, self.wavelet)
        reconstructed = reconstructed[: len(signal)]

        stego = reconstructed.reshape(original_shape)
        return stego, sr

    def decode(self, audio: np.ndarray) -> bytes:
        """Extract data from DWT detail coefficients."""
        signal = audio.flatten().astype(np.float64)

        coeffs = pywt.wavedec(signal, self.wavelet, level=self.level)
        detail = coeffs[1]

        indices = np.arange(len(detail))
        if self.seed is not None:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(indices)

        window = 32
        energy = np.array([
            np.mean(detail[max(0, i - window):i + window] ** 2)
            for i in range(len(detail))
        ])
        energy = np.maximum(energy, 1e-10)

        all_bits = []
        for i in range(len(detail)):
            idx = indices[i]
            strength = self.alpha * np.sqrt(energy[idx])
            coeff = detail[idx]
            q = np.round(coeff / strength) * strength
            diff = coeff - q
            all_bits.append(1 if diff >= 0 else 0)

        all_bits = np.array(all_bits, dtype=np.uint8)

        header = np.packbits(all_bits[:32]).tobytes()
        length = int.from_bytes(header[:4], "big")

        msg_bits = all_bits[32 : 32 + length * 8]
        pad_len = (8 - len(msg_bits) % 8) % 8
        if pad_len:
            msg_bits = np.concatenate([msg_bits, np.zeros(pad_len, dtype=np.uint8)])

        return np.packbits(msg_bits).tobytes()[:length]
