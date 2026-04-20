"""
Image DWT Steganography — Transform Domain Method.

Embeds data in Discrete Wavelet Transform detail coefficients.
Uses multi-level decomposition with fixed-strength QIM for reliable extraction.
Skips coefficients whose underlying pixel regions would clip during inverse DWT.
"""

import numpy as np
import cv2
import pywt
from typing import Optional


class ImageDWT:
    """DWT-based image steganography with multi-level decomposition."""

    def __init__(
        self,
        wavelet: str = "haar",
        level: int = 2,
        alpha: float = 0.5,
        subband: str = "LH",
        seed: Optional[int] = None,
    ):
        self.wavelet = wavelet
        self.level = level
        self.alpha = alpha
        self.subband = subband
        self.seed = seed
        self._subband_idx = {"LH": 0, "HL": 1, "HH": 2}[subband]

    def capacity(self, image: np.ndarray) -> int:
        h, w = image.shape[:2]
        coeff_h = h // (2 ** self.level)
        coeff_w = w // (2 ** self.level)
        total_bits = coeff_h * coeff_w
        return (total_bits // 8) - 4

    def _safe_indices_mask(self, channel: np.ndarray, subband_shape: tuple) -> np.ndarray:
        """Mask of subband coefficients whose underlying pixel block has embedding headroom."""
        step = 2 ** self.level
        sh, sw = subband_shape
        mask = np.zeros(sh * sw, dtype=bool)
        margin = max(self.alpha * 2, 20.0)
        lo, hi = margin, 255.0 - margin
        for i in range(sh):
            y0 = i * step
            y1 = min(y0 + step, channel.shape[0])
            for j in range(sw):
                x0 = j * step
                x1 = min(x0 + step, channel.shape[1])
                region = channel[y0:y1, x0:x1]
                if region.size == 0:
                    continue
                m = float(region.mean())
                if lo < m < hi:
                    mask[i * sw + j] = True
        return mask

    def encode(self, cover_image: np.ndarray, secret_data: bytes) -> np.ndarray:
        """Embed data in DWT detail coefficients using QIM."""
        image = cover_image.copy()

        if len(image.shape) == 3:
            ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            channel = ycrcb[:, :, 0].astype(np.float64)
        else:
            channel = image.astype(np.float64)
            ycrcb = None

        coeffs = pywt.wavedec2(channel, self.wavelet, level=self.level)
        detail_coeffs = list(coeffs[1])
        target = detail_coeffs[self._subband_idx].copy()
        flat = target.flatten()

        length_header = len(secret_data).to_bytes(4, "big")
        payload = length_header + secret_data
        bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))

        safe_mask = self._safe_indices_mask(channel, target.shape)

        indices = np.arange(len(flat))
        if self.seed is not None:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(indices)

        delta = self.alpha
        bit_idx = 0

        for pos in indices:
            if bit_idx >= len(bits):
                break
            if not safe_mask[pos]:
                continue
            coeff = flat[pos]
            bit = int(bits[bit_idx])
            q = int(np.floor(coeff / delta))
            if (q % 2) != bit:
                if ((q + 1) % 2) == bit:
                    flat[pos] = (q + 1) * delta + delta / 2
                else:
                    flat[pos] = q * delta + delta / 2
            else:
                flat[pos] = q * delta + delta / 2
            bit_idx += 1

        if bit_idx < len(bits):
            raise ValueError(
                f"Not enough safe coefficients: needed {len(bits)}, got {bit_idx}"
            )

        detail_coeffs[self._subband_idx] = flat.reshape(target.shape)
        coeffs[1] = tuple(detail_coeffs)

        reconstructed = pywt.waverec2(coeffs, self.wavelet)
        reconstructed = reconstructed[:channel.shape[0], :channel.shape[1]]

        if ycrcb is not None:
            ycrcb[:, :, 0] = np.clip(reconstructed, 0, 255).astype(np.uint8)
            return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        return np.clip(reconstructed, 0, 255).astype(np.uint8)

    def decode(self, stego_image: np.ndarray) -> bytes:
        """Extract data from DWT detail coefficients using QIM."""
        if len(stego_image.shape) == 3:
            ycrcb = cv2.cvtColor(stego_image, cv2.COLOR_BGR2YCrCb)
            channel = ycrcb[:, :, 0].astype(np.float64)
        else:
            channel = stego_image.astype(np.float64)

        coeffs = pywt.wavedec2(channel, self.wavelet, level=self.level)
        detail_coeffs = coeffs[1]
        target = detail_coeffs[self._subband_idx]
        flat = target.flatten()

        safe_mask = self._safe_indices_mask(channel, target.shape)

        indices = np.arange(len(flat))
        if self.seed is not None:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(indices)

        delta = self.alpha
        all_bits = []

        for pos in indices:
            if not safe_mask[pos]:
                continue
            coeff = flat[pos]
            q = int(np.floor(coeff / delta))
            all_bits.append(q % 2)

        all_bits = np.array(all_bits, dtype=np.uint8)

        header_bytes = np.packbits(all_bits[:32]).tobytes()
        length = int.from_bytes(header_bytes[:4], "big")

        msg_bits = all_bits[32 : 32 + length * 8]
        pad_len = (8 - len(msg_bits) % 8) % 8
        if pad_len:
            msg_bits = np.concatenate([msg_bits, np.zeros(pad_len, dtype=np.uint8)])

        return np.packbits(msg_bits).tobytes()[:length]
