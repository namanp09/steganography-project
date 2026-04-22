"""Text compression for steganography payload reduction."""

from .compression import (
    TextCompressor,
    compress_text,
    decompress_text,
)

__all__ = [
    "TextCompressor",
    "compress_text",
    "decompress_text",
]
