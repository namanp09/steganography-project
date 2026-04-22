"""
GAN-based Audio Steganography API wrapper.
Spectrogram domain embedding with psychoacoustic masking.
"""

import torch
import numpy as np
import librosa
from typing import Tuple

from models.audio_gan import AudioGANSteganography
from config.settings import AUDIO_GAN


class AudioGANStego:
    """High-level API for audio GAN steganography."""

    def __init__(self, model_path: str = None, device: str = "cuda"):
        """
        Initialize Audio GAN Stego.

        Args:
            model_path: Path to pretrained model
            device: cuda or cpu
        """
        self.device = device
        self.model = AudioGANSteganography(
            msg_length=AUDIO_GAN.message_bits,
            freq_bins=AUDIO_GAN.freq_bins,
            base_ch=32,
        ).to(device)
        self.model.eval()

        if model_path:
            state_dict = torch.load(model_path, map_location=device)
            self.model.load_state_dict(state_dict)

        # STFT parameters
        self.n_fft = AUDIO_GAN.n_fft
        self.hop_length = AUDIO_GAN.hop_length

    def capacity(self, audio: np.ndarray) -> int:
        """Estimate capacity in bytes."""
        return AUDIO_GAN.message_bits // 8

    def _text_to_bits(self, data: bytes) -> torch.Tensor:
        """Convert bytes to binary tensor."""
        bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
        bits = bits[: AUDIO_GAN.message_bits]

        if len(bits) < AUDIO_GAN.message_bits:
            bits = np.pad(bits, (0, AUDIO_GAN.message_bits - len(bits)))

        return torch.from_numpy(bits.astype(np.float32))

    def _bits_to_text(self, bits: torch.Tensor) -> bytes:
        """Convert binary tensor to bytes."""
        with torch.no_grad():
            bits_np = (bits > 0.5).cpu().numpy().astype(np.uint8)

        num_bytes = len(bits_np) // 8
        bits_reshaped = bits_np[: num_bytes * 8].reshape(-1, 8)
        byte_array = np.packbits(bits_reshaped.reshape(-1, 8), axis=1, bitorder="big").flatten()

        return byte_array.tobytes()

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

        # Resample if needed
        if sr != AUDIO_GAN.n_fft:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=AUDIO_GAN.n_fft)
            sr = AUDIO_GAN.n_fft

        # STFT
        D = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(D)
        phase = np.angle(D)

        # Normalize and convert to tensor
        mag_norm = magnitude / (magnitude.max() + 1e-8)
        mag_tensor = torch.from_numpy(mag_norm[np.newaxis, np.newaxis, :, :]).float().to(self.device)
        phase_tensor = torch.from_numpy(phase[np.newaxis, np.newaxis, :, :]).float().to(self.device)

        # Message
        message_bits = self._text_to_bits(secret_data).unsqueeze(0).to(self.device)

        # Embed
        with torch.no_grad():
            stego_mag_tensor, _ = self.model(mag_tensor, phase_tensor, message_bits)

        # Convert back
        stego_mag = stego_mag_tensor[0, 0].cpu().numpy()
        stego_mag = stego_mag * magnitude.max()

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

        # Normalize
        mag_norm = magnitude / (magnitude.max() + 1e-8)
        mag_tensor = torch.from_numpy(mag_norm[np.newaxis, np.newaxis, :, :]).float().to(self.device)

        # Decode (phase not needed for message extraction)
        with torch.no_grad():
            _, decoded_bits = self.model(
                mag_tensor,
                torch.zeros_like(mag_tensor),
                torch.zeros(1, AUDIO_GAN.message_bits).to(self.device),
            )

        return self._bits_to_text(decoded_bits[0])
