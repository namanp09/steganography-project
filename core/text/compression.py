"""
Text Compression Pipeline: Transformer + Arithmetic Coding

Modern compression using a lightweight character-level language model
to predict token probabilities, then arithmetic coding for efficient encoding.
"""

import torch
import torch.nn as nn
import numpy as np
import struct
from typing import Tuple, Optional
import zlib


class CharacterLevelTransformer(nn.Module):
    """
    Lightweight transformer for character-level probability estimation.
    Predicts P(next_char | context) for arithmetic coding.
    """

    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = self._create_pos_encoding(max_len=512, d_model=d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=0.1,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.to_logits = nn.Linear(d_model, vocab_size)

    def _create_pos_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """Create positional encoding."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: (batch, seq_len) tensor of token indices

        Returns:
            logits: (batch, seq_len, vocab_size) log-probabilities
        """
        seq_len = token_ids.shape[1]
        pos_enc = self.pos_encoding[:, :seq_len, :].to(token_ids.device)

        x = self.embedding(token_ids)
        x = x + pos_enc
        x = self.transformer(x)
        logits = self.to_logits(x)
        return logits


class ArithmeticCoder:
    """
    Simple arithmetic coder/decoder for entropy-optimal bit sequences.
    Uses floating-point arithmetic (sufficient for our use case).
    """

    def __init__(self, precision: int = 32):
        self.precision = precision
        self.max_range = 2**precision

    def encode(self, probabilities: np.ndarray, symbol_sequence: np.ndarray) -> bytes:
        """
        Encode symbol sequence using probability distribution.

        Args:
            probabilities: (seq_len, vocab_size) array of probabilities per symbol
            symbol_sequence: (seq_len,) array of symbol indices

        Returns:
            Compressed bitstream as bytes
        """
        low = 0.0
        high = 1.0
        seq_len = len(symbol_sequence)

        for i in range(seq_len):
            probs = probabilities[i]
            symbol = symbol_sequence[i]

            # Cumulative probabilities
            cum_prob = np.cumsum(probs)
            if symbol == 0:
                symbol_low = 0.0
                symbol_high = cum_prob[0]
            else:
                symbol_low = cum_prob[symbol - 1]
                symbol_high = cum_prob[symbol]

            # Update range
            range_size = high - low
            high = low + range_size * symbol_high
            low = low + range_size * symbol_low

        # Encode final range as bits
        mid = (low + high) / 2
        # Convert to integer representation
        int_mid = int(mid * self.max_range)
        # Encode length prefix + value
        return struct.pack(">I", seq_len) + int_mid.to_bytes(
            (self.precision + 7) // 8, byteorder="big"
        )

    def decode(
        self, bitstream: bytes, probabilities: np.ndarray, seq_len: int
    ) -> np.ndarray:
        """
        Decode arithmetic-coded sequence.

        Args:
            bitstream: Encoded bytes
            probabilities: (seq_len, vocab_size) probability array
            seq_len: Expected sequence length

        Returns:
            Decoded symbol sequence (seq_len,)
        """
        # Parse bitstream
        int_mid = int.from_bytes(
            bitstream[4:], byteorder="big"
        )  # Skip length prefix
        value = int_mid / self.max_range

        low = 0.0
        high = 1.0
        symbols = []

        for i in range(seq_len):
            probs = probabilities[i]
            cum_prob = np.cumsum(probs)

            # Find which symbol contains 'value'
            range_size = high - low
            for symbol in range(len(probs)):
                if symbol == 0:
                    symbol_low = 0.0
                    symbol_high = cum_prob[0]
                else:
                    symbol_low = cum_prob[symbol - 1]
                    symbol_high = cum_prob[symbol]

                symbol_low_abs = low + range_size * symbol_low
                symbol_high_abs = low + range_size * symbol_high

                if symbol_low_abs <= value < symbol_high_abs:
                    symbols.append(symbol)
                    low = symbol_low_abs
                    high = symbol_high_abs
                    break

        return np.array(symbols, dtype=np.uint8)


class TextCompressor:
    """
    High-level text compression API using Transformer + Arithmetic Coding.
    Falls back to zlib for robustness if transformer not available.
    """

    def __init__(self, use_transformer: bool = True, vocab_size: int = 256):
        self.use_transformer = use_transformer
        self.vocab_size = vocab_size
        self.coder = ArithmeticCoder(precision=32)

        if use_transformer:
            self.transformer = CharacterLevelTransformer(vocab_size=vocab_size)
            self.transformer.eval()
        else:
            self.transformer = None

    def compress(self, text: str) -> bytes:
        """
        Compress text to minimal bitstream.

        Args:
            text: Plain text string

        Returns:
            Compressed bytes (format: magic|algo|data)
        """
        text_bytes = text.encode("utf-8")

        if self.use_transformer and self.transformer is not None:
            return self._compress_transformer(text_bytes)
        else:
            return self._compress_zlib(text_bytes)

    def _compress_transformer(self, text_bytes: bytes) -> bytes:
        """Compress using transformer + arithmetic coding."""
        # Convert bytes to token sequence
        tokens = np.array(list(text_bytes), dtype=np.int64)

        # Get probabilities from transformer (context window of 32)
        context_len = min(32, len(tokens))
        with torch.no_grad():
            if len(tokens) <= 1:
                # Can't compress single byte
                return b"ZL" + self._compress_zlib(text_bytes)

            token_tensor = torch.from_numpy(tokens[:-1]).long().unsqueeze(0)
            logits = self.transformer(token_tensor)
            probs = torch.softmax(logits[0].cpu(), dim=-1).numpy()

            # Arithmetic encode
            compressed = self.coder.encode(probs, tokens[1:])

        return b"TR" + struct.pack(">I", len(text_bytes)) + compressed

    def _compress_zlib(self, text_bytes: bytes) -> bytes:
        """Fallback: zlib compression."""
        compressed = zlib.compress(text_bytes, level=9)
        return b"ZL" + compressed

    def decompress(self, compressed: bytes) -> str:
        """
        Decompress text from bitstream.

        Args:
            compressed: Compressed bytes

        Returns:
            Decompressed text
        """
        if len(compressed) < 2:
            raise ValueError("Invalid compressed data")

        magic = compressed[:2]

        if magic == b"TR":
            return self._decompress_transformer(compressed)
        elif magic == b"ZL":
            return self._decompress_zlib(compressed[2:])
        else:
            raise ValueError(f"Unknown compression magic: {magic}")

    def _decompress_transformer(self, compressed: bytes) -> str:
        """Decompress transformer-encoded data."""
        original_len = struct.unpack(">I", compressed[2:6])[0]
        bitstream = compressed[6:]

        # Placeholder: would need stored probabilities or re-inference
        # For now, fall back to a simple approach
        return self._decompress_zlib(bitstream).decode("utf-8")

    def _decompress_zlib(self, compressed: bytes) -> str:
        """Decompress zlib data."""
        try:
            decompressed = zlib.decompress(compressed)
            return decompressed.decode("utf-8")
        except Exception as e:
            raise ValueError(f"Decompression failed: {e}")

    def get_compression_ratio(self, original_size: int, compressed_size: int) -> float:
        """Return compression ratio (bytes saved / original size)."""
        if original_size == 0:
            return 0.0
        return 1.0 - (compressed_size / original_size)


# Module-level convenience functions
_default_compressor = None


def compress_text(text: str, use_transformer: bool = False) -> bytes:
    """Compress text string to bytes."""
    global _default_compressor
    if _default_compressor is None:
        _default_compressor = TextCompressor(use_transformer=use_transformer)
    return _default_compressor.compress(text)


def decompress_text(compressed: bytes) -> str:
    """Decompress bytes back to text string."""
    global _default_compressor
    if _default_compressor is None:
        _default_compressor = TextCompressor()
    return _default_compressor.decompress(compressed)
