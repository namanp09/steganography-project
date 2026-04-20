"""
Audio DWT Steganography — Transform Domain Method.

Embeds data in DWT detail coefficients of audio signals using
fixed-strength QIM (Quantization Index Modulation) for reliable extraction.
"""

import numpy as np
import pywt
from typing import Optional, Tuple


class AudioDWT:
    """DWT-based audio steganography with QIM embedding."""

    def __init__(
        self,
        wavelet: str = "db4",
        level: int = 4,
        alpha: float = 0.02,
        seed: Optional[int] = None,
    ):
        self.wavelet = wavelet
        self.level = level
        self.alpha = alpha
        self.seed = seed

    def capacity(self, audio: np.ndarray) -> int:
        coeffs = pywt.wavedec(audio.flatten(), self.wavelet, level=self.level, mode="periodization")
        n_coeffs = len(coeffs[1])
        return (n_coeffs // 8) - 4

    def encode(
        self, audio: np.ndarray, sr: int, secret_data: bytes
    ) -> Tuple[np.ndarray, int]:
        original_shape = audio.shape
        signal = audio.flatten().astype(np.float64)

        coeffs = pywt.wavedec(signal, self.wavelet, level=self.level, mode="periodization")
        detail = coeffs[1].copy()

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

        delta = self.alpha

        for i, bit in enumerate(bits):
            idx = indices[i]
            coeff = detail[idx]
            bit = int(bit)
            q = int(np.floor(coeff / delta))
            if (q % 2) != bit:
                if ((q + 1) % 2) == bit:
                    detail[idx] = (q + 1) * delta + delta / 2
                else:
                    detail[idx] = q * delta + delta / 2
            else:
                detail[idx] = q * delta + delta / 2

        coeffs[1] = detail
        reconstructed = pywt.waverec(coeffs, self.wavelet, mode="periodization")
        reconstructed = reconstructed[: len(signal)]

        return reconstructed.reshape(original_shape), sr

    def decode(self, audio: np.ndarray) -> bytes:
        signal = audio.flatten().astype(np.float64)
        coeffs = pywt.wavedec(signal, self.wavelet, level=self.level, mode="periodization")
        detail = coeffs[1]

        indices = np.arange(len(detail))
        if self.seed is not None:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(indices)

        delta = self.alpha
        all_bits = []

        for i in range(len(detail)):
            idx = indices[i]
            coeff = detail[idx]
            q = int(np.floor(coeff / delta))
            all_bits.append(q % 2)

        all_bits = np.array(all_bits, dtype=np.uint8)
        header = np.packbits(all_bits[:32]).tobytes()
        length = int.from_bytes(header[:4], "big")

        msg_bits = all_bits[32 : 32 + length * 8]
        pad_len = (8 - len(msg_bits) % 8) % 8
        if pad_len:
            msg_bits = np.concatenate([msg_bits, np.zeros(pad_len, dtype=np.uint8)])

        return np.packbits(msg_bits).tobytes()[:length]
