"""
GAN-based Video Steganography API wrapper.
Spatio-temporal embedding with motion-aware masking.
"""

import torch
import numpy as np
from typing import Dict
from pathlib import Path

from models.video_gan import VideoGANSteganography
from core.video.frame_utils import extract_frames, reconstruct_video, compute_optical_flow
from config.settings import VIDEO_GAN
from core.error_correction import BitRepetitionECC, bits_to_bytes


class VideoGANStego:
    """High-level API for video GAN steganography."""

    def __init__(self, model_path: str = None, device: str = "cuda", ecc_factor: int = 3):
        """
        Initialize Video GAN Stego with bit-repetition ECC.

        Args:
            model_path: Path to pretrained model
            device: cuda or cpu
            ecc_factor: Bit-repetition factor (1=no ECC, 3=balanced, 5=strong).
        """
        self.device = device
        self.ecc = BitRepetitionECC(factor=ecc_factor) if ecc_factor > 1 else None
        self.effective_bits = VIDEO_GAN.message_bits // ecc_factor

        self.model = VideoGANSteganography(
            msg_length=VIDEO_GAN.message_bits,
            base_ch=VIDEO_GAN.base_channels,
            temporal_window=VIDEO_GAN.temporal_window,
            frame_size=VIDEO_GAN.frame_size,
        ).to(device)
        self.model.eval()

        if model_path:
            state_dict = torch.load(model_path, map_location=device)
            self.model.load_state_dict(state_dict)

    def capacity(self, video_path: str) -> int:
        """Effective capacity in bytes after ECC, across all temporal windows."""
        frames, _ = extract_frames(video_path, max_frames=1, resize=(VIDEO_GAN.frame_size, VIDEO_GAN.frame_size))
        if len(frames) > 0:
            n_frames = min(100, 300)
            return (self.effective_bits // 8) * (n_frames // VIDEO_GAN.temporal_window)
        return 0

    def _text_to_bits(self, data: bytes) -> torch.Tensor:
        """Convert bytes → bits → repetition encode → tensor."""
        bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
        bits = bits[: self.effective_bits]
        if len(bits) < self.effective_bits:
            bits = np.pad(bits, (0, self.effective_bits - len(bits)))

        if self.ecc is not None:
            bits = self.ecc.encode(bits)

        if len(bits) < VIDEO_GAN.message_bits:
            bits = np.pad(bits, (0, VIDEO_GAN.message_bits - len(bits)))
        bits = bits[: VIDEO_GAN.message_bits]

        return torch.from_numpy(bits.astype(np.float32))

    def _bits_to_text(self, bits: torch.Tensor) -> bytes:
        """Decode model logits → sigmoid → majority vote → bytes."""
        with torch.no_grad():
            bits_np = (torch.sigmoid(bits) > 0.5).cpu().numpy().astype(np.uint8)

        if self.ecc is not None:
            bits_np = self.ecc.decode(bits_np)

        return bits_to_bytes(bits_np)

    def encode(
        self,
        video_path: str,
        secret_data: bytes,
        output_path: str,
        max_frames: int = 300,
    ) -> Dict:
        """
        Embed secret data into video.

        Args:
            video_path: Path to input video
            secret_data: Binary data to embed
            output_path: Path to save stego video
            max_frames: Maximum frames to process

        Returns:
            Dictionary with metadata
        """
        # Extract frames
        frames, metadata = extract_frames(video_path, max_frames=max_frames, resize=(VIDEO_GAN.frame_size, VIDEO_GAN.frame_size))

        if len(frames) == 0:
            raise ValueError(f"Could not extract frames from {video_path}")

        fps = metadata.get("fps", 30)

        # Convert frames to tensor (skip optical flow - optional feature)
        frames_tensor = torch.from_numpy(np.stack(frames)).float().permute(0, 3, 1, 2)  # (T, 3, H, W)
        frames_tensor = frames_tensor.unsqueeze(0) / 255.0  # (1, T, 3, H, W), normalize

        # Message
        message_bits = self._text_to_bits(secret_data).unsqueeze(0).to(self.device)

        flow_tensor = None

        # Embed in temporal windows
        frames_tensor = frames_tensor.to(self.device)
        T = frames_tensor.shape[1]
        tw = VIDEO_GAN.temporal_window
        stego_windows = []
        with torch.no_grad():
            for start in range(0, T, tw):
                window = frames_tensor[:, start:start + tw]
                if window.shape[1] < tw:
                    pad = torch.zeros(1, tw - window.shape[1], 3, VIDEO_GAN.frame_size, VIDEO_GAN.frame_size, device=self.device)
                    window = torch.cat([window, pad], dim=1)
                stego_win, _ = self.model(window, message_bits, None)
                stego_windows.append(stego_win[0, :min(tw, T - start)])
        stego_tensor = torch.cat(stego_windows, dim=0).unsqueeze(0)  # (1, T, 3, H, W)

        # Convert back to uint8
        stego_frames = (stego_tensor[0].permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)

        # Reconstruct video
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        reconstruct_video(stego_frames, output_path, fps=fps)

        return {
            "frames_used": len(frames),
            "total_frames": len(frames),
            "capacity_bytes": (VIDEO_GAN.message_bits // 8) * (len(frames) // VIDEO_GAN.temporal_window),
            "data_size_bytes": len(secret_data),
            "fps": fps,
        }

    def decode(self, video_path: str, max_frames: int = 300) -> bytes:
        """
        Extract secret data from video.

        Args:
            video_path: Path to stego video
            max_frames: Maximum frames to process

        Returns:
            Extracted binary data
        """
        # Extract frames
        frames, _ = extract_frames(video_path, max_frames=max_frames, resize=(VIDEO_GAN.frame_size, VIDEO_GAN.frame_size))

        if len(frames) == 0:
            raise ValueError(f"Could not extract frames from {video_path}")

        # Convert to tensor
        frames_tensor = torch.from_numpy(np.stack(frames)).float().permute(0, 3, 1, 2).unsqueeze(0) / 255.0
        frames_tensor = frames_tensor.to(self.device)

        # Decode using first temporal window only
        T = frames_tensor.shape[1]
        tw = VIDEO_GAN.temporal_window
        window = frames_tensor[:, :tw]
        if window.shape[1] < tw:
            pad = torch.zeros(1, tw - window.shape[1], 3, VIDEO_GAN.frame_size, VIDEO_GAN.frame_size, device=self.device)
            window = torch.cat([window, pad], dim=1)
        # Run ONLY the decoder — do not re-encode
        # decoder expects (B, 3, T, H, W); window is (B, T, 3, H, W)
        with torch.no_grad():
            decoded_bits = self.model.decoder(window.permute(0, 2, 1, 3, 4))

        return self._bits_to_text(decoded_bits[0])
