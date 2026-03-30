"""
Global configuration for the AI-Enhanced Steganography System.
Uses modern 2025 state-of-the-art parameters.
"""

import os
from dataclasses import dataclass, field

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class PathConfig:
    project_root: str = PROJECT_ROOT
    data_dir: str = os.path.join(PROJECT_ROOT, "data")
    models_dir: str = os.path.join(PROJECT_ROOT, "models", "checkpoints")
    upload_dir: str = os.path.join(PROJECT_ROOT, "uploads")
    output_dir: str = os.path.join(PROJECT_ROOT, "outputs")

    def __post_init__(self):
        for d in [self.data_dir, self.models_dir, self.upload_dir, self.output_dir]:
            os.makedirs(d, exist_ok=True)


@dataclass
class EncryptionConfig:
    algorithm: str = "AES-256-GCM"  # Authenticated encryption (modern standard)
    key_size: int = 32              # 256 bits
    nonce_size: int = 12            # 96-bit nonce for GCM
    tag_size: int = 16              # 128-bit auth tag
    pbkdf2_iterations: int = 600_000  # OWASP 2024 recommendation
    salt_size: int = 16


@dataclass
class ImageStegoConfig:
    supported_formats: list = field(default_factory=lambda: [".png", ".bmp", ".tiff"])
    dct_block_size: int = 8
    dwt_wavelet: str = "haar"
    dwt_level: int = 2
    # Deep learning
    dl_image_size: int = 256
    dl_message_bits: int = 128       # Higher capacity than older methods


@dataclass
class AudioStegoConfig:
    supported_formats: list = field(default_factory=lambda: [".wav", ".flac"])
    sample_rate: int = 44100
    dwt_wavelet: str = "db4"         # Daubechies-4 (better freq. localization)
    dwt_level: int = 4
    frame_length: int = 2048
    hop_length: int = 512


@dataclass
class VideoStegoConfig:
    """Video steganography — PRIMARY FOCUS of this project."""
    supported_formats: list = field(default_factory=lambda: [".mp4", ".avi", ".mkv"])
    frame_size: int = 256            # DL model input size
    message_bits: int = 128          # Bits per frame
    # Frame selection
    embed_every_n_frames: int = 2    # Skip frames for temporal consistency
    max_embed_frames: int = 300
    # Codec-robust training
    simulate_h264: bool = True       # Train against H.264 compression artifacts
    jpeg_quality_range: tuple = (50, 95)  # JPEG quality range for noise layer
    # Invertible Neural Network
    inn_num_blocks: int = 8          # Number of coupling blocks
    inn_subnet_channels: int = 64
    # 3D CNN temporal
    temporal_window: int = 5         # Frames for temporal attention
    # Motion compensation
    use_motion_vectors: bool = True  # Use optical flow for adaptive embedding


@dataclass
class TrainingConfig:
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    epochs: int = 100
    # Loss weights
    lambda_image: float = 1.0        # Reconstruction / image quality loss
    lambda_message: float = 10.0     # Message recovery loss
    lambda_perceptual: float = 0.5   # LPIPS perceptual loss
    lambda_adversarial: float = 0.01 # GAN loss (WGAN-GP)
    lambda_frequency: float = 0.1    # Frequency domain loss
    # Scheduler
    scheduler: str = "cosine"        # Cosine annealing with warm restarts
    warmup_epochs: int = 5
    # Mixed precision
    use_amp: bool = True             # Automatic Mixed Precision for speed


@dataclass
class MetricsConfig:
    min_psnr: float = 35.0           # Modern target: >35 dB
    min_ssim: float = 0.97           # Modern target: >0.97
    target_ber: float = 0.0          # Bit Error Rate target: 0%
    target_lpips: float = 0.05       # Perceptual similarity target: <0.05


# Singleton instances
PATHS = PathConfig()
ENCRYPTION = EncryptionConfig()
IMAGE_STEGO = ImageStegoConfig()
AUDIO_STEGO = AudioStegoConfig()
VIDEO_STEGO = VideoStegoConfig()
TRAINING = TrainingConfig()
METRICS = MetricsConfig()
