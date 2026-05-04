"""
GAN-based Video Steganography using VideoGANSteganography (3D CNN).

Encode: embed message into the first temporal window (5 frames) of the video.
Decode: extract from the first temporal window.
All other frames are passed through unchanged.

Message layout per window (ECC factor=3 for error correction):
    [HEADER(4B) | CRC32(4B) | DATA | TERMINATOR(2B)]
"""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    import numpy as np
    import binascii
    from models.video_gan import VideoGANSteganography
    from core.error_correction import BitRepetitionECC, bytes_to_bits, bits_to_bytes
    _TORCH_AVAILABLE = True
    _TORCH_ERROR = None
except Exception as _e:
    _TORCH_AVAILABLE = False
    _TORCH_ERROR = f"{type(_e).__name__}: {_e}"

from pathlib import Path
from typing import Dict, Tuple

from core.video.frame_utils import extract_frames, reconstruct_video
from config.settings import PATHS, VIDEO_GAN


class VideoGANStego:
    """
    Video steganography using the trained VideoGANSteganography (3D CNN).
    Embeds message into the first 5-frame temporal window.

    With message_bits=32 there is no room for header/CRC overhead, so we use
    raw bit packing: the first N bits carry the message, the rest are zero-padded.
    Decode strips trailing null bytes and returns whatever the model recovered.
    """

    def __init__(self, model_path: str = None, device: str = "cpu", ecc_factor: int = 1):
        if not _TORCH_AVAILABLE:
            raise RuntimeError(
                f"PyTorch not available: {_TORCH_ERROR}. Video GAN method unavailable."
            )

        self.device = device
        self.ecc = BitRepetitionECC(factor=ecc_factor) if ecc_factor > 1 else None
        self.ecc_factor = ecc_factor
        self.bits_per_window = VIDEO_GAN.message_bits // ecc_factor

        self.model = VideoGANSteganography(
            msg_length=VIDEO_GAN.message_bits,
            base_ch=VIDEO_GAN.base_channels,
            temporal_window=VIDEO_GAN.temporal_window,
            frame_size=VIDEO_GAN.frame_size,
        ).to(device)
        self.model.eval()

        if model_path and Path(model_path).exists():
            state = torch.load(model_path, map_location=device, weights_only=True)
            self.model.load_state_dict(state)
            del state

    # ── Capacity ───────────────────────────────────────────────────────────────

    def capacity(self, video_path: str) -> int:
        """Message capacity in bytes (raw, no overhead with this encoding)."""
        return self.bits_per_window // 8

    # ── Encode helpers ─────────────────────────────────────────────────────────

    def _payload_to_bits(self, secret_data: bytes) -> torch.Tensor:
        """
        Pack secret_data into VIDEO_GAN.message_bits bits (raw, no header overhead).
        Truncates to capacity; zero-pads shorter messages.
        Returns float32 tensor of shape (message_bits,).
        """
        max_bytes = self.bits_per_window // 8
        payload = secret_data[:max_bytes]
        bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))

        target = self.bits_per_window
        if len(bits) < target:
            bits = np.pad(bits, (0, target - len(bits)))
        bits = bits[:target]

        if self.ecc is not None:
            bits = self.ecc.encode(bits)

        full = VIDEO_GAN.message_bits
        if len(bits) < full:
            bits = np.pad(bits, (0, full - len(bits)))
        bits = bits[:full]

        return torch.from_numpy(bits.astype(np.float32))

    # ── Decode helpers ─────────────────────────────────────────────────────────

    def _logits_to_payload(self, logits: torch.Tensor) -> tuple[bytes, bool]:
        """
        Decode message bits from logits tensor (shape: message_bits,).
        Returns (message_bytes, verified: bool).
        Verification is best-effort: returns True if the decoded bytes are valid UTF-8.
        """
        with torch.no_grad():
            bits_np = (torch.sigmoid(logits) > 0.5).cpu().numpy().astype(np.uint8)

        if self.ecc is not None:
            bits_np = self.ecc.decode(bits_np)

        all_bytes = bits_to_bytes(bits_np[:self.bits_per_window])
        # Strip null padding
        message = all_bytes.rstrip(b"\x00")
        try:
            message.decode("utf-8")
            verified = True
        except UnicodeDecodeError:
            verified = False
        return message, verified

    # ── Frames ↔ tensor ────────────────────────────────────────────────────────

    def _frames_to_tensor(self, frames) -> torch.Tensor:
        """List of BGR numpy frames → (1, T, 3, H, W) float32 tensor in [0,1]."""
        import cv2
        T = VIDEO_GAN.temporal_window
        fs = VIDEO_GAN.frame_size
        arr = []
        for f in frames[:T]:
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (fs, fs))
            arr.append(torch.from_numpy(resized).float() / 255.0)
        # (T, H, W, 3) → (T, 3, H, W)
        t = torch.stack(arr).permute(0, 3, 1, 2)
        return t.unsqueeze(0).to(self.device)  # (1, T, 3, H, W)

    def _tensor_to_frames(self, tensor: torch.Tensor, orig_frames) -> list:
        """
        (1, T, 3, H, W) stego tensor → list of BGR frames at original resolution.
        Applies the GAN residual (stego - cover at 64px) to the original full-res
        frame so the original quality is preserved and only the signal is added.
        """
        import cv2
        T = VIDEO_GAN.temporal_window
        fs = VIDEO_GAN.frame_size
        stego_np = (tensor[0].permute(0, 2, 3, 1).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        result = []
        for i in range(T):
            orig_h, orig_w = orig_frames[i].shape[:2]
            # Cover frame at 64×64 (same as model input)
            cover_rgb = cv2.cvtColor(orig_frames[i], cv2.COLOR_BGR2RGB)
            cover_small = cv2.resize(cover_rgb, (fs, fs)).astype(np.int16)
            # Residual at 64×64
            residual_small = stego_np[i].astype(np.int16) - cover_small
            # Upscale residual to original resolution and add to original
            residual_full = cv2.resize(
                residual_small.astype(np.float32), (orig_w, orig_h),
                interpolation=cv2.INTER_LINEAR,
            )
            stego_full = np.clip(
                cover_rgb.astype(np.float32) + residual_full, 0, 255
            ).astype(np.uint8)
            result.append(cv2.cvtColor(stego_full, cv2.COLOR_RGB2BGR))
        return result

    # ── Public API ─────────────────────────────────────────────────────────────

    def encode(self, video_path: str, secret_data: bytes, output_path: str, max_frames: int = 300) -> Dict:
        frames, metadata = extract_frames(video_path, max_frames=max_frames)
        if not frames:
            raise ValueError(f"Could not extract frames from {video_path}")

        fps = metadata.get("fps", 30)
        T = VIDEO_GAN.temporal_window

        if len(frames) < T:
            raise ValueError(f"Video has only {len(frames)} frames, need at least {T}")

        # Prepare message bits
        msg_bits = self._payload_to_bits(secret_data).unsqueeze(0).to(self.device)  # (1, bits)

        # Cap strength to 0.05 for visual quality — keeps PSNR ~32dB
        original_strength = self.model.generator.strength.item()
        with torch.no_grad():
            self.model.generator.strength.data.clamp_(max=0.05)

        # Embed in ALL non-overlapping windows for multi-window redundancy
        output_frames = list(frames)
        windows_used = 0
        for start in range(0, len(frames) - T + 1, T):
            window_tensor = self._frames_to_tensor(frames[start:start + T])
            with torch.no_grad():
                stego_tensor, _ = self.model(window_tensor, msg_bits, flow=None)
            stego_window = self._tensor_to_frames(stego_tensor, frames[start:start + T])
            output_frames[start:start + T] = stego_window
            windows_used += 1

        # Restore original strength for decode path
        with torch.no_grad():
            self.model.generator.strength.data.fill_(original_strength)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        reconstruct_video(output_frames, output_path, fps=fps)

        return {
            "frames_used": windows_used * T,
            "total_frames": len(frames),
            "windows_embedded": windows_used,
            "capacity_bytes": self.capacity(video_path),
            "data_size_bytes": len(secret_data),
            "fps": fps,
        }

    def decode(self, video_path: str, max_frames: int = 300) -> bytes:
        message, _ = self.decode_with_verification(video_path, max_frames)
        return message

    def decode_with_verification(self, video_path: str, max_frames: int = 300) -> Tuple[bytes, bool]:
        T = VIDEO_GAN.temporal_window
        frames, _ = extract_frames(video_path, max_frames=max_frames)
        if not frames:
            raise ValueError(f"Could not extract frames from {video_path}")
        if len(frames) < T:
            raise ValueError(f"Video has only {len(frames)} frames, need {T}")

        # Aggregate logits across all windows — majority vote makes recovery robust
        all_logits = []
        for start in range(0, len(frames) - T + 1, T):
            window_tensor = self._frames_to_tensor(frames[start:start + T])
            with torch.no_grad():
                stego_for_decoder = window_tensor.permute(0, 2, 1, 3, 4)
                logits = self.model.decoder(stego_for_decoder)
            all_logits.append(logits[0])

        # Average logits across all windows before thresholding
        avg_logits = torch.stack(all_logits).mean(dim=0)
        return self._logits_to_payload(avg_logits)
