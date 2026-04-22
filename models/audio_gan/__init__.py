"""GAN-based audio steganography with spectrogram domain embedding."""

from .model import (
    AudioGANGenerator,
    AudioGANDiscriminator,
    AudioGANSteganography,
)

__all__ = [
    "AudioGANGenerator",
    "AudioGANDiscriminator",
    "AudioGANSteganography",
]
