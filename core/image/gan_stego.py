"""
GAN-based Image Steganography API wrapper.
Provides encode/decode interface compatible with the existing API.
"""
from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    from torchvision import transforms
    from models.image_gan import ImageGANSteganography
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

import numpy as np
import binascii
from typing import Optional, Tuple
from PIL import Image

from config.settings import IMAGE_GAN
from core.error_correction import BitRepetitionECC, bytes_to_bits, bits_to_bytes


class ImageGANStego:
    """
    High-level API for image GAN steganography.
    Compatible with existing encode/decode interface.
    """

    # Magic header marks payload start
    _HEADER = b"\xC0DE"
    _TERMINATOR = b"\x00\x00"  # Sentinel marking end of (CRC + message)
    _CRC_SIZE = 4              # CRC32, 4 bytes

    def __init__(self, model_path: Optional[str] = None, device: str = "cuda", ecc_factor: int = 3):
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not installed. GAN method is unavailable in this deployment.")
        self.device = device
        self.ecc = BitRepetitionECC(factor=ecc_factor) if ecc_factor > 1 else None
        self.bits_per_tile = IMAGE_GAN.message_bits // ecc_factor
        self.tile_size = IMAGE_GAN.image_size

        self.model = ImageGANSteganography(
            msg_length=IMAGE_GAN.message_bits,
            base_ch=IMAGE_GAN.base_channels,
            image_size=IMAGE_GAN.image_size,
        ).to(device)
        self.model.eval()

        if model_path:
            state_dict = torch.load(model_path, map_location=device)
            self.model.load_state_dict(state_dict)

    def _grid_dims(self, h: int, w: int) -> tuple:
        """Number of non-overlapping tiles that fit in (h, w) at tile_size."""
        return h // self.tile_size, w // self.tile_size

    def capacity(self, image: np.ndarray) -> int:
        """Effective payload capacity in bytes (after header + CRC + terminator)."""
        h, w = image.shape[:2]
        n_tiles = (h // self.tile_size) * (w // self.tile_size)
        total_bits = n_tiles * self.bits_per_tile
        # Subtract 4-byte header + 4-byte CRC + 2-byte terminator = 10 bytes overhead
        return max(0, (total_bits // 8) - 10)

    def _payload_to_tile_bits(self, secret_data: bytes, n_tiles: int) -> tuple:
        """
        Returns (tile_bits, n_used_tiles).
        Layout: [HEADER(4B) | CRC32(4B) | DATA | TERMINATOR(2B)]
        CRC is computed over DATA only and lets decode verify integrity.
        """
        clean = secret_data.replace(b"\x00", b" ")
        crc = binascii.crc32(clean).to_bytes(self._CRC_SIZE, "big")
        payload = self._HEADER + crc + clean + self._TERMINATOR

        max_bytes = (n_tiles * self.bits_per_tile) // 8
        if len(payload) > max_bytes:
            payload = payload[:max_bytes]

        bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))

        # Compute how many tiles are actually needed
        bits_after_ecc = len(bits) * (self.ecc.factor if self.ecc else 1)
        n_used = min(n_tiles, max(1, (bits_after_ecc + IMAGE_GAN.message_bits - 1) // IMAGE_GAN.message_bits))

        # Pad data bits to exactly n_used × bits_per_tile (so ECC fills n_used × message_bits)
        target = n_used * self.bits_per_tile
        if len(bits) < target:
            bits = np.pad(bits, (0, target - len(bits)))

        if self.ecc is not None:
            bits = self.ecc.encode(bits)

        full_used = n_used * IMAGE_GAN.message_bits
        if len(bits) < full_used:
            bits = np.pad(bits, (0, full_used - len(bits)))
        bits = bits[:full_used]

        # Build full tile array with unused tiles as zeros (will be skipped during embed)
        tile_bits = np.zeros((n_tiles, IMAGE_GAN.message_bits), dtype=np.float32)
        tile_bits[:n_used] = bits.reshape(n_used, IMAGE_GAN.message_bits).astype(np.float32)
        return tile_bits, n_used

    def _tile_logits_to_payload(self, logits: torch.Tensor) -> tuple:
        """
        Returns (message_bytes, verified) where verified is True iff CRC32 matches.
        Layout decoded: HEADER + CRC(4) + DATA + TERMINATOR
        """
        with torch.no_grad():
            bits_np = (torch.sigmoid(logits) > 0.5).cpu().numpy().astype(np.uint8)

        flat = bits_np.flatten()
        if self.ecc is not None:
            flat = self.ecc.decode(flat)

        all_bytes = bits_to_bytes(flat)

        idx = all_bytes.find(self._HEADER)
        if idx < 0:
            # Header not found — return best-effort raw bytes, unverified
            return all_bytes.split(b"\x00", 1)[0], False

        body = all_bytes[idx + len(self._HEADER):]
        if len(body) < self._CRC_SIZE:
            return body, False

        expected_crc = body[: self._CRC_SIZE]
        rest = body[self._CRC_SIZE :]

        # Read until first null (terminator)
        end = rest.find(b"\x00")
        message = rest if end < 0 else rest[:end]

        actual_crc = binascii.crc32(message).to_bytes(self._CRC_SIZE, "big")
        verified = actual_crc == expected_crc

        return message, verified

    def encode(self, cover_image: np.ndarray, secret_data: bytes) -> np.ndarray:
        """
        Embed secret data via spatial tiling.
        Image is cropped to multiple of tile_size, split into tiles, each tile
        embeds bits_per_tile bits. Image resolution is preserved (no global resize).
        """
        orig_h, orig_w = cover_image.shape[:2]
        n_h, n_w = self._grid_dims(orig_h, orig_w)

        if n_h == 0 or n_w == 0:
            # Image smaller than one tile — fall back to single-tile resize-based encode
            n_h, n_w = 1, 1
            cover_rgb = cv2.cvtColor(cover_image, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(cover_rgb, (self.tile_size, self.tile_size))
            tile_grid = resized[None, :, :, :]
            crop_h = crop_w = self.tile_size
            single_tile_path = True
        else:
            crop_h, crop_w = n_h * self.tile_size, n_w * self.tile_size
            cover_rgb = cv2.cvtColor(cover_image[:crop_h, :crop_w], cv2.COLOR_BGR2RGB)
            # (n_h, tile, n_w, tile, 3) → (n_h*n_w, tile, tile, 3)
            tile_grid = cover_rgb.reshape(n_h, self.tile_size, n_w, self.tile_size, 3)\
                                 .swapaxes(1, 2)\
                                 .reshape(n_h * n_w, self.tile_size, self.tile_size, 3)
            single_tile_path = False

        n_tiles = n_h * n_w
        tile_bits, n_used = self._payload_to_tile_bits(secret_data, n_tiles)

        # Only run the model on the tiles we actually need to embed in
        used_tiles = tile_grid[:n_used]
        used_tensor = torch.from_numpy(used_tiles).float().permute(0, 3, 1, 2) / 255.0
        used_tensor = used_tensor.to(self.device)
        msg_tensor = torch.from_numpy(tile_bits[:n_used]).to(self.device)

        with torch.no_grad():
            stego_used, _ = self.model(used_tensor, msg_tensor)

        stego_used_np = (stego_used.permute(0, 2, 3, 1).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

        # Build full stego_np: replace only the used tiles, leave rest unchanged
        stego_np = tile_grid.copy()
        stego_np[:n_used] = stego_used_np

        if single_tile_path:
            stego_rgb = cv2.resize(stego_np[0], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        else:
            grid = stego_np.reshape(n_h, n_w, self.tile_size, self.tile_size, 3)\
                           .swapaxes(1, 2)\
                           .reshape(crop_h, crop_w, 3)
            stego_rgb = cv2.cvtColor(cover_image, cv2.COLOR_BGR2RGB).copy()
            stego_rgb[:crop_h, :crop_w] = grid

        return cv2.cvtColor(stego_rgb, cv2.COLOR_RGB2BGR)

    def decode(self, stego_image: np.ndarray) -> bytes:
        """Extract payload from a tile-encoded stego image. Returns message bytes only."""
        message, _ = self.decode_with_verification(stego_image)
        return message

    def decode_with_verification(self, stego_image: np.ndarray) -> tuple:
        """
        Extract payload AND verify CRC32. Returns (message_bytes, verified: bool).
        verified == True means the message recovered exactly as encoded.
        """
        h, w = stego_image.shape[:2]
        n_h, n_w = self._grid_dims(h, w)

        if n_h == 0 or n_w == 0:
            stego_rgb = cv2.cvtColor(stego_image, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(stego_rgb, (self.tile_size, self.tile_size))
            tile_grid = resized[None, :, :, :]
        else:
            crop_h, crop_w = n_h * self.tile_size, n_w * self.tile_size
            stego_rgb = cv2.cvtColor(stego_image[:crop_h, :crop_w], cv2.COLOR_BGR2RGB)
            tile_grid = stego_rgb.reshape(n_h, self.tile_size, n_w, self.tile_size, 3)\
                                 .swapaxes(1, 2)\
                                 .reshape(n_h * n_w, self.tile_size, self.tile_size, 3)

        tiles_tensor = torch.from_numpy(tile_grid).float().permute(0, 3, 1, 2) / 255.0
        tiles_tensor = tiles_tensor.to(self.device)

        with torch.no_grad():
            logits = self.model.decoder(tiles_tensor)

        return self._tile_logits_to_payload(logits)


# Import cv2 only when needed
import cv2
