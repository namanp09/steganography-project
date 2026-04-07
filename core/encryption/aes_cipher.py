"""
AES-256-GCM Authenticated Encryption Module.

Uses modern authenticated encryption (GCM mode) instead of legacy CBC.
GCM provides both confidentiality AND integrity in a single pass.
Key derivation via PBKDF2-HMAC-SHA256 with OWASP-recommended iterations.
"""

import os
import struct
import hashlib
from Crypto.Cipher import AES

from config.settings import ENCRYPTION


class AESCipher:
    """AES-256-GCM authenticated encryption with PBKDF2 key derivation."""

    def __init__(self, password: str):
        self.password = password.encode("utf-8")

    def _derive_key(self, salt: bytes) -> bytes:
        """Derive a 256-bit key from password using PBKDF2-HMAC-SHA256."""
        return hashlib.pbkdf2_hmac(
            "sha256",
            self.password,
            salt,
            ENCRYPTION.pbkdf2_iterations,
            dklen=ENCRYPTION.key_size,
        )

    def encrypt(self, plaintext: bytes) -> bytes:
        """
        Encrypt data with AES-256-GCM.

        Output format: [salt (16B)] [nonce (12B)] [tag (16B)] [ciphertext]
        Total overhead: 44 bytes.
        """
        salt = os.urandom(ENCRYPTION.salt_size)
        key = self._derive_key(salt)

        cipher = AES.new(key, AES.MODE_GCM, nonce=os.urandom(ENCRYPTION.nonce_size))
        ciphertext, tag = cipher.encrypt_and_digest(plaintext)

        return salt + cipher.nonce + tag + ciphertext

    def decrypt(self, data: bytes) -> bytes:
        """
        Decrypt AES-256-GCM encrypted data.
        Raises ValueError if authentication fails (tampered data).
        """
        s = ENCRYPTION.salt_size
        n = ENCRYPTION.nonce_size
        t = ENCRYPTION.tag_size

        salt = data[:s]
        nonce = data[s : s + n]
        tag = data[s + n : s + n + t]
        ciphertext = data[s + n + t :]

        key = self._derive_key(salt)
        cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)

        return cipher.decrypt_and_verify(ciphertext, tag)

    def encrypt_message(self, message: str) -> bytes:
        """Encrypt a text message, prepending its length for extraction."""
        msg_bytes = message.encode("utf-8")
        length_prefix = struct.pack(">I", len(msg_bytes))
        return self.encrypt(length_prefix + msg_bytes)

    def decrypt_message(self, data: bytes) -> str:
        """Decrypt and return a text message."""
        decrypted = self.decrypt(data)
        length = struct.unpack(">I", decrypted[:4])[0]
        return decrypted[4 : 4 + length].decode("utf-8")
