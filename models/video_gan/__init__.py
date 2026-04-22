"""GAN-based video steganography with spatio-temporal architecture."""

from .model import (
    VideoGANGenerator,
    TemporalDiscriminator,
    VideoGANSteganography,
)

__all__ = [
    "VideoGANGenerator",
    "TemporalDiscriminator",
    "VideoGANSteganography",
]
