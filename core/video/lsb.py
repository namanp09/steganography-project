"""
Video LSB Steganography — Baseline Method.

Embeds data across selected video frames using LSB.
Supports motion-compensated frame selection for robustness.
"""

import numpy as np
import cv2
from typing import List, Optional
from core.image.lsb import ImageLSB
from core.video.frame_utils import (
    extract_frames, reconstruct_video, compute_optical_flow, select_embedding_regions
)


class VideoLSB:
    """Video LSB steganography with frame selection strategies."""

    def __init__(
        self,
        num_bits: int = 1,
        embed_every_n: int = 2,
        use_motion_comp: bool = True,
        seed: Optional[int] = None,
    ):
        self.num_bits = num_bits
        self.embed_every_n = embed_every_n
        self.use_motion_comp = use_motion_comp
        self.seed = seed
        self.image_lsb = ImageLSB(num_bits=num_bits, seed=seed)

    def encode(
        self,
        video_path: str,
        secret_data: bytes,
        output_path: str,
        max_frames: int = 300,
    ) -> dict:
        """
        Embed data across video frames.

        Returns:
            Dict with metadata (frames used, capacity, etc.)
        """
        frames, meta = extract_frames(video_path, max_frames=max_frames)
        fps = meta["fps"]

        # Select frames for embedding
        embed_indices = list(range(0, len(frames), self.embed_every_n))

        # Calculate capacity per frame and split data
        sample_cap = self.image_lsb.capacity(frames[0])
        total_capacity = sample_cap * len(embed_indices)

        # Add global length header
        length_header = len(secret_data).to_bytes(4, "big")
        payload = length_header + secret_data

        if len(payload) > total_capacity:
            raise ValueError(
                f"Data too large: {len(payload)} bytes, capacity: {total_capacity} bytes"
            )

        # Distribute data chunks across frames
        chunk_size = sample_cap
        data_offset = 0

        for i, frame_idx in enumerate(embed_indices):
            if data_offset >= len(payload):
                break

            chunk = payload[data_offset : data_offset + chunk_size]
            if not chunk:
                break

            frame = frames[frame_idx]

            if self.use_motion_comp and frame_idx > 0:
                flow_mag = compute_optical_flow(frames[frame_idx - 1], frame)
                stable_mask = select_embedding_regions(flow_mag)
                # Apply mask: only embed in stable regions
                # For simplicity, embed full frame if >60% stable
                stable_ratio = np.mean(stable_mask)
                if stable_ratio < 0.4:
                    # Skip highly dynamic frame
                    continue

            frames[frame_idx] = self.image_lsb.encode(frame, chunk)
            data_offset += len(chunk)

        reconstruct_video(frames, output_path, fps=fps)

        return {
            "frames_used": len(embed_indices),
            "total_frames": len(frames),
            "capacity_bytes": total_capacity,
            "data_size_bytes": len(secret_data),
            "fps": fps,
        }

    def decode(self, video_path: str, max_frames: int = 300) -> bytes:
        """Extract data from stego video."""
        frames, _ = extract_frames(video_path, max_frames=max_frames)
        embed_indices = list(range(0, len(frames), self.embed_every_n))

        # Extract from first frame to get global length
        first_chunk = self.image_lsb.decode(frames[embed_indices[0]])

        # The first 4 bytes of the first chunk contain the total data length
        total_length = int.from_bytes(first_chunk[:4], "big")
        all_data = first_chunk[4:]  # Data portion from first frame

        for i in range(1, len(embed_indices)):
            if len(all_data) >= total_length:
                break
            frame_idx = embed_indices[i]
            chunk = self.image_lsb.decode(frames[frame_idx])
            all_data += chunk

        return all_data[:total_length]
