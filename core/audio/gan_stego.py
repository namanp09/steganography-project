"""
GAN-based Audio Steganography API wrapper.
Spectrogram domain embedding with psychoacoustic masking.
"""
from __future__ import annotations

try:
    import torch
    from models.audio_gan import AudioGANSteganography
    _TORCH_AVAILABLE = True
    _TORCH_ERROR = None
except Exception as _e:
    _TORCH_AVAILABLE = False
    _TORCH_ERROR = f"{type(_e).__name__}: {_e}"

import numpy as np
import librosa
from typing import Tuple

from config.settings import AUDIO_GAN
from core.error_correction import BitRepetitionECC, bits_to_bytes


class AudioGANStego:
    """High-level API for audio GAN steganography."""

    def __init__(self, model_path: str = None, device: str = "cuda", ecc_factor: int = 3):
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not installed. GAN method is unavailable in this deployment.")
        self.device = device
        # Checkpoint was trained with msg_length=128, freq_bins=128, base_ch=32.
        # Hardcode these so config changes don't break checkpoint loading.
        self._msg_length = 128
        self.ecc = BitRepetitionECC(factor=ecc_factor) if ecc_factor > 1 else None
        self.effective_bits = self._msg_length // ecc_factor

        self.model = AudioGANSteganography(
            msg_length=self._msg_length,
            freq_bins=128,
            base_ch=32,
        ).to(device)
        self.model.eval()

        if model_path:
            state_dict = torch.load(model_path, map_location=device, weights_only=True)
            self.model.load_state_dict(state_dict)
            del state_dict
            import gc; gc.collect()

        # STFT parameters
        self.n_fft = AUDIO_GAN.n_fft
        self.hop_length = AUDIO_GAN.hop_length

    def capacity(self, audio: np.ndarray) -> int:
        """Effective capacity in bytes after ECC."""
        return self.effective_bits // 8

    def _text_to_bits(self, data: bytes) -> torch.Tensor:
        """Convert bytes → bits → repetition encode → tensor."""
        bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
        bits = bits[: self.effective_bits]
        if len(bits) < self.effective_bits:
            bits = np.pad(bits, (0, self.effective_bits - len(bits)))

        if self.ecc is not None:
            bits = self.ecc.encode(bits)

        if len(bits) < self._msg_length:
            bits = np.pad(bits, (0, self._msg_length - len(bits)))
        bits = bits[: self._msg_length]

        return torch.from_numpy(bits.astype(np.float32))

    def _bits_to_text(self, bits: torch.Tensor) -> bytes:
        """Decode model logits → sigmoid → majority vote → bytes."""
        with torch.no_grad():
            bits_np = (torch.sigmoid(bits) > 0.5).cpu().numpy().astype(np.uint8)

        if self.ecc is not None:
            bits_np = self.ecc.decode(bits_np)

        return bits_to_bytes(bits_np)

    def encode(self, audio: np.ndarray, sr: int, secret_data: bytes) -> Tuple[np.ndarray, int]:
        """
        Embed secret data into audio.

        Args:
            audio: Audio samples (float32 or int16)
            sr: Sample rate
            secret_data: Binary data to embed

        Returns:
            (stego_audio, sr)
        """
        # Convert to float32
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # STFT — produces (n_fft//2+1, T) = (513, T) with n_fft=1024
        D = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(D)
        phase = np.angle(D)

        # Checkpoint was trained with freq_bins=128: slice lower 128 bins only.
        # Upper bins (128:) are left untouched during embedding.
        mag_slice = magnitude[:self._msg_length, :]   # (128, T)
        phase_slice = phase[:self._msg_length, :]

        mag_max = magnitude.max() + 1e-8
        mag_norm = mag_slice / mag_max
        mag_tensor = torch.from_numpy(mag_norm[np.newaxis, np.newaxis]).float().to(self.device)
        phase_tensor = torch.from_numpy(phase_slice[np.newaxis, np.newaxis]).float().to(self.device)

        # Message
        message_bits = self._text_to_bits(secret_data).unsqueeze(0).to(self.device)

        # Embed
        with torch.no_grad():
            stego_mag_tensor, _ = self.model(mag_tensor, phase_tensor, message_bits)

        # Merge stego slice back into full spectrogram
        stego_slice = stego_mag_tensor[0, 0].cpu().numpy() * mag_max
        stego_mag = magnitude.copy()
        stego_mag[:self._msg_length, :] = stego_slice

        # Reconstruct with original phase
        D_stego = stego_mag * np.exp(1j * phase)

        # ISTFT
        stego_audio = librosa.istft(D_stego, hop_length=self.hop_length)

        return stego_audio, sr

    def decode(self, audio: np.ndarray, sr: int = None) -> bytes:
        """
        Extract secret data from audio.

        Args:
            audio: Audio samples
            sr: Sample rate (optional, assumed to match training)

        Returns:
            Extracted binary data
        """
        # Convert to float32
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0

        # STFT
        D = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(D)

        # Slice to the 128 freq bins the model was trained on
        mag_slice = magnitude[:self._msg_length, :]
        mag_norm = mag_slice / (mag_slice.max() + 1e-8)
        mag_tensor = torch.from_numpy(mag_norm[np.newaxis, np.newaxis]).float().to(self.device)

        # Run ONLY the decoder — do not re-encode
        with torch.no_grad():
            decoded_bits = self.model.decoder(mag_tensor)

        return self._bits_to_text(decoded_bits[0])
