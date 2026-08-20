"""Encrypt/decrypt secrets at rest with Windows DPAPI.

DPAPI (CryptProtectData) binds the ciphertext to the current Windows user
account, so a stolen `.env` file is unreadable on any other machine or user.
Keys are stored as `enc1:<base64>` values; `decrypt_env_value()` is called
from core.config right after load_dotenv so `os.getenv` sees the real key.
"""

import base64
import ctypes
import os
import sys

PREFIX = "enc1:"

if sys.platform == "win32":
    from ctypes import wintypes

    class DATA_BLOB(ctypes.Structure):
        _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]

    def _blob(data: bytes) -> DATA_BLOB:
        buf = ctypes.create_string_buffer(data, len(data))
        return DATA_BLOB(len(data), ctypes.cast(buf, ctypes.POINTER(ctypes.c_char)))

    def _protect(data: bytes) -> bytes:
        blob_in = _blob(data)
        blob_out = DATA_BLOB()
        if not ctypes.windll.crypt32.CryptProtectData(
            ctypes.byref(blob_in), "ARIA secret", None, None, None, 0, ctypes.byref(blob_out)
        ):
            raise OSError("CryptProtectData failed")
        try:
            return ctypes.string_at(blob_out.pbData, blob_out.cbData)
        finally:
            ctypes.windll.kernel32.LocalFree(blob_out.pbData)

    def _unprotect(data: bytes) -> bytes:
        blob_in = _blob(data)
        blob_out = DATA_BLOB()
        if not ctypes.windll.crypt32.CryptUnprotectData(
            ctypes.byref(blob_in), None, None, None, None, 0, ctypes.byref(blob_out)
        ):
            raise OSError("CryptUnprotectData failed")
        try:
            return ctypes.string_at(blob_out.pbData, blob_out.cbData)
        finally:
            ctypes.windll.kernel32.LocalFree(blob_out.pbData)

    def encrypt(plaintext: str) -> str:
        return PREFIX + base64.b64encode(_protect(plaintext.encode("utf-8"))).decode("ascii")

    def decrypt(ciphertext: str) -> str:
        if not ciphertext.startswith(PREFIX):
            return ciphertext
        raw = base64.b64decode(ciphertext[len(PREFIX):])
        return _unprotect(raw).decode("utf-8")

else:
    # Non-Windows fallback: leave plaintext visible (no local trust anchor).
    def encrypt(plaintext: str) -> str:
        return plaintext

    def decrypt(ciphertext: str) -> str:
        return ciphertext