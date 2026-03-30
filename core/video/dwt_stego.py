"""
Video DWT Steganography — Transform Domain.

Applies DWT-based embedding on selected video frames.
Uses motion-compensated frame selection.
"""

import numpy as np
from typing import Optional
from core.image.dwt_stego import ImageDWT
from core.video.frame_utils import (
    extract_frames, reconstruct_video, compute_optical_flow, select_embedding_regions
)


class VideoDWT:
    """DWT-based video steganography with temporal awareness."""

    def __init__(
        self,
        wavelet: str = "haar",
        level: int = 2,
        alpha: float = 0.1,
        embed_every_n: int = 2,
        use_motion_comp: bool = True,
        seed: Optional[int] = None,
    ):
        self.embed_every_n = embed_every_n
        self.use_motion_comp = use_motion_comp
        self.image_dwt = ImageDWT(
            wavelet=wavelet, level=level, alpha=alpha, seed=seed
        )

    def encode(
        self,
        video_path: str,
        secret_data: bytes,
        output_path: str,
        max_frames: int = 300,
    ) -> dict:
        """Embed data across video frames using DWT."""
        frames, meta = extract_frames(video_path, max_frames=max_frames)
        fps = meta["fps"]

        embed_indices = list(range(0, len(frames), self.embed_every_n))
        sample_cap = self.image_dwt.capacity(frames[0])
        total_capacity = sample_cap * len(embed_indices)

        length_header = len(secret_data).to_bytes(4, "big")
        payload = length_header + secret_data

        if len(payload) > total_capacity:
            raise ValueError(
                f"Data too large: {len(payload)} bytes, capacity: {total_capacity} bytes"
            )

        chunk_size = sample_cap
        data_offset = 0

        for frame_idx in embed_indices:
            if data_offset >= len(payload):
                break

            chunk = payload[data_offset : data_offset + chunk_size]
            if not chunk:
                break

            if self.use_motion_comp and frame_idx > 0:
                flow_mag = compute_optical_flow(frames[frame_idx - 1], frames[frame_idx])
                stable_ratio = np.mean(select_embedding_regions(flow_mag))
                if stable_ratio < 0.4:
                    continue

            frames[frame_idx] = self.image_dwt.encode(frames[frame_idx], chunk)
            data_offset += len(chunk)

        reconstruct_video(frames, output_path, fps=fps)

        return {
            "frames_used": len(embed_indices),
            "total_frames": len(frames),
            "capacity_bytes": total_capacity,
            "data_size_bytes": len(secret_data),
        }

    def decode(self, video_path: str, max_frames: int = 300) -> bytes:
        """Extract data from DWT stego video."""
        frames, _ = extract_frames(video_path, max_frames=max_frames)
        embed_indices = list(range(0, len(frames), self.embed_every_n))

        first_chunk = self.image_dwt.decode(frames[embed_indices[0]])
        total_length = int.from_bytes(first_chunk[:4], "big")
        all_data = first_chunk[4:]

        for i in range(1, len(embed_indices)):
            if len(all_data) >= total_length:
                break
            chunk = self.image_dwt.decode(frames[embed_indices[i]])
            all_data += chunk

        return all_data[:total_length]
