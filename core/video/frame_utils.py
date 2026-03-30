"""
Video Frame Utilities — Modern FFmpeg-based I/O.

Uses PyAV for fast, codec-aware video processing.
Includes optical flow computation for motion-compensated embedding.
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional
from pathlib import Path

try:
    import av
    HAS_AV = True
except ImportError:
    HAS_AV = False


def extract_frames(
    video_path: str,
    max_frames: Optional[int] = None,
    resize: Optional[Tuple[int, int]] = None,
) -> Tuple[List[np.ndarray], dict]:
    """
    Extract frames from video using PyAV (FFmpeg-based, fast).

    Args:
        video_path: Path to video file.
        max_frames: Maximum frames to extract.
        resize: Optional (width, height) to resize frames.

    Returns:
        (frames_list, metadata_dict)
    """
    if not HAS_AV:
        return _extract_frames_cv2(video_path, max_frames, resize)

    container = av.open(video_path)
    stream = container.streams.video[0]

    metadata = {
        "fps": float(stream.average_rate),
        "width": stream.width,
        "height": stream.height,
        "codec": stream.codec_context.name,
        "total_frames": stream.frames or 0,
        "duration": float(stream.duration * stream.time_base) if stream.duration else 0,
    }

    frames = []
    for i, frame in enumerate(container.decode(video=0)):
        if max_frames and i >= max_frames:
            break
        img = frame.to_ndarray(format="bgr24")
        if resize:
            img = cv2.resize(img, resize)
        frames.append(img)

    container.close()
    return frames, metadata


def _extract_frames_cv2(
    video_path: str,
    max_frames: Optional[int],
    resize: Optional[Tuple[int, int]],
) -> Tuple[List[np.ndarray], dict]:
    """Fallback using OpenCV."""
    cap = cv2.VideoCapture(video_path)
    metadata = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "codec": "unknown",
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames and len(frames) >= max_frames:
            break
        if resize:
            frame = cv2.resize(frame, resize)
        frames.append(frame)

    cap.release()
    return frames, metadata


def reconstruct_video(
    frames: List[np.ndarray],
    output_path: str,
    fps: float = 30.0,
    codec: str = "libx264",
    crf: int = 18,
) -> str:
    """
    Reconstruct video from frames using PyAV with H.264 encoding.

    Args:
        frames: List of BGR numpy arrays.
        output_path: Output video file path.
        fps: Frames per second.
        codec: Video codec (libx264 for H.264, libx265 for H.265).
        crf: Constant Rate Factor (lower = higher quality, 18 is visually lossless).

    Returns:
        Output path.
    """
    if not HAS_AV:
        return _reconstruct_video_cv2(frames, output_path, fps)

    h, w = frames[0].shape[:2]
    container = av.open(output_path, mode="w")
    stream = container.add_stream(codec, rate=int(fps))
    stream.width = w
    stream.height = h
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": str(crf)}

    for frame_np in frames:
        frame = av.VideoFrame.from_ndarray(frame_np, format="bgr24")
        for packet in stream.encode(frame):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)

    container.close()
    return output_path


def _reconstruct_video_cv2(
    frames: List[np.ndarray], output_path: str, fps: float
) -> str:
    """Fallback using OpenCV."""
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()
    return output_path


def compute_optical_flow(
    prev_frame: np.ndarray, curr_frame: np.ndarray
) -> np.ndarray:
    """
    Compute dense optical flow using Farneback method.
    Used for motion-compensated embedding — identify stable regions.

    Returns:
        Flow magnitude map (H, W) — low values = stable (good for embedding).
    """
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray,
        flow=None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )

    magnitude = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)
    return magnitude


def select_embedding_regions(
    flow_magnitude: np.ndarray, threshold: float = 2.0
) -> np.ndarray:
    """
    Select stable regions for embedding based on optical flow.
    Low motion = stable = better for embedding.

    Returns:
        Binary mask (H, W) — True where embedding is safe.
    """
    return flow_magnitude < threshold
