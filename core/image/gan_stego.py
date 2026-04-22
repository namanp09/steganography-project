"""
GAN-based Image Steganography API wrapper.
Provides encode/decode interface compatible with the existing API.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple
from torchvision import transforms
from PIL import Image

from models.image_gan import ImageGANSteganography
from config.settings import IMAGE_GAN


class ImageGANStego:
    """
    High-level API for image GAN steganography.
    Compatible with existing encode/decode interface.
    """

    def __init__(self, model_path: Optional[str] = None, device: str = "cuda"):
        """
        Initialize Image GAN Stego.

        Args:
            model_path: Path to pretrained model checkpoint
            device: Device to use (cuda/cpu)
        """
        self.device = device
        self.model = ImageGANSteganography(
            msg_length=IMAGE_GAN.message_bits,
            base_ch=IMAGE_GAN.base_channels,
            image_size=IMAGE_GAN.image_size,
        ).to(device)
        self.model.eval()

        if model_path:
            state_dict = torch.load(model_path, map_location=device)
            self.model.load_state_dict(state_dict)

        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_GAN.image_size, IMAGE_GAN.image_size)),
            transforms.ToTensor(),
        ])

    def capacity(self, image: np.ndarray) -> int:
        """
        Estimate capacity in bytes.
        GAN-based methods can embed the full message_bits.

        Args:
            image: Cover image (H, W, 3) BGR uint8

        Returns:
            Capacity in bytes
        """
        # Typically: message_bits / 8 bytes
        return IMAGE_GAN.message_bits // 8

    def _text_to_bits(self, data: bytes) -> torch.Tensor:
        """Convert bytes to binary tensor."""
        bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
        bits = bits[: IMAGE_GAN.message_bits]  # Truncate to message length

        # Pad if necessary
        if len(bits) < IMAGE_GAN.message_bits:
            bits = np.pad(bits, (0, IMAGE_GAN.message_bits - len(bits)))

        return torch.from_numpy(bits.astype(np.float32))

    def _bits_to_text(self, bits: torch.Tensor) -> bytes:
        """Convert binary tensor to bytes."""
        with torch.no_grad():
            bits_np = (bits > 0.5).cpu().numpy().astype(np.uint8)

        # Reshape to bytes
        num_bytes = len(bits_np) // 8
        bits_reshaped = bits_np[: num_bytes * 8].reshape(-1, 8)
        byte_array = np.packbits(bits_reshaped.reshape(-1, 8), axis=1, bitorder="big").flatten()

        return byte_array.tobytes()

    def encode(self, cover_image: np.ndarray, secret_data: bytes) -> np.ndarray:
        """
        Embed secret data into cover image.

        Args:
            cover_image: (H, W, 3) BGR uint8 image
            secret_data: Binary data to embed

        Returns:
            stego_image: (H, W, 3) BGR uint8 image
        """
        # Convert BGR to RGB and normalize
        cover_rgb = cv2.cvtColor(cover_image, cv2.COLOR_BGR2RGB)
        cover_tensor = self.transform(Image.fromarray(cover_rgb)).unsqueeze(0).to(self.device)

        # Convert data to bits
        message_bits = self._text_to_bits(secret_data).unsqueeze(0).to(self.device)

        # Embed
        with torch.no_grad():
            stego_tensor, _ = self.model(cover_tensor, message_bits)

        # Convert back to uint8 BGR
        stego_np = (stego_tensor[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        stego_bgr = cv2.cvtColor(stego_np, cv2.COLOR_RGB2BGR)

        return stego_bgr

    def decode(self, stego_image: np.ndarray) -> bytes:
        """
        Extract secret data from stego image.

        Args:
            stego_image: (H, W, 3) BGR uint8 image

        Returns:
            Extracted binary data
        """
        # Convert BGR to RGB and normalize
        stego_rgb = cv2.cvtColor(stego_image, cv2.COLOR_BGR2RGB)
        stego_tensor = self.transform(Image.fromarray(stego_rgb)).unsqueeze(0).to(self.device)

        # Decode
        with torch.no_grad():
            _, decoded_bits = self.model(stego_tensor, torch.zeros(1, IMAGE_GAN.message_bits).to(self.device))

        # Convert bits to bytes
        return self._bits_to_text(decoded_bits[0])


# Import cv2 only when needed
import cv2
