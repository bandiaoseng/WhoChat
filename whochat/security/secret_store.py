"""Secure, non-plaintext storage for small secrets (currently the AI API key).

Why this module exists
----------------------
``config.json`` used to hold the AI API key in clear text.  A plaintext key in
a configuration file leaks through backups, sync folders, screen shares and
bug reports, so the key is now kept in a separate store that is encrypted with
the operating system's own user-bound facility:

* Windows: **DPAPI** (``CryptProtectData`` / ``CryptUnprotectData``, user
  scope).  The ciphertext can only be decrypted by the same Windows account on
  the same machine, and it is stored in ``<config_dir>/secrets.json``.

There is deliberately **no plaintext fallback**.  When the OS facility is
unavailable the store reports ``unavailable`` and refuses to persist anything;
callers are expected to surface that instead of writing the secret to disk.

Boundaries: this module only touches files under the WhoChat config directory.
It never reads the WeChat process, its memory, or its database.
"""

from __future__ import annotations

import base64
import ctypes
import json
import os
from pathlib import Path

BACKEND_DPAPI = "windows_dpapi"
BACKEND_UNAVAILABLE = "unavailable"

SECRETS_FILE_NAME = "secrets.json"

# Secret names.  Kept stable because they are the keys inside secrets.json.
AI_API_KEY = "ai.api_key"

_CRYPTPROTECT_UI_FORBIDDEN = 0x01


class _DataBlob(ctypes.Structure):
    _fields_ = [
        ("cbData", ctypes.c_uint32),
        ("pbData", ctypes.POINTER(ctypes.c_char)),
    ]


class _DpapiProtector:
    """Thin ctypes wrapper around the Windows DPAPI pair."""

    def __init__(self) -> None:
        self.crypt32 = ctypes.windll.crypt32  # type: ignore[attr-defined]
        self.kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        self.crypt32.CryptProtectData.argtypes = [
            ctypes.POINTER(_DataBlob),
            ctypes.c_wchar_p,
            ctypes.POINTER(_DataBlob),
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.POINTER(_DataBlob),
        ]
        self.crypt32.CryptProtectData.restype = ctypes.c_int
        self.crypt32.CryptUnprotectData.argtypes = [
            ctypes.POINTER(_DataBlob),
            ctypes.c_void_p,
            ctypes.POINTER(_DataBlob),
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.POINTER(_DataBlob),
        ]
        self.crypt32.CryptUnprotectData.restype = ctypes.c_int
        self.kernel32.LocalFree.argtypes = [ctypes.c_void_p]
        self.kernel32.LocalFree.restype = ctypes.c_void_p

    def protect(self, plaintext: bytes) -> bytes:
        buffer = ctypes.create_string_buffer(plaintext, len(plaintext))
        blob_in = _DataBlob(len(plaintext), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_char)))
        blob_out = _DataBlob()
        if not self.crypt32.CryptProtectData(
            ctypes.byref(blob_in),
            None,
            None,
            None,
            None,
            _CRYPTPROTECT_UI_FORBIDDEN,
            ctypes.byref(blob_out),
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            return ctypes.string_at(blob_out.pbData, blob_out.cbData)
        finally:
            self.kernel32.LocalFree(blob_out.pbData)

    def unprotect(self, ciphertext: bytes) -> bytes:
        buffer = ctypes.create_string_buffer(ciphertext, len(ciphertext))
        blob_in = _DataBlob(len(ciphertext), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_char)))
        blob_out = _DataBlob()
        if not self.crypt32.CryptUnprotectData(
            ctypes.byref(blob_in),
            None,
            None,
            None,
            None,
            _CRYPTPROTECT_UI_FORBIDDEN,
            ctypes.byref(blob_out),
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            return ctypes.string_at(blob_out.pbData, blob_out.cbData)
        finally:
            self.kernel32.LocalFree(blob_out.pbData)


_PROTECTOR: _DpapiProtector | None = None
_PROTECTOR_PROBED = False


def _protector() -> _DpapiProtector | None:
    """Return the DPAPI protector, or ``None`` when the OS facility is absent."""
    global _PROTECTOR, _PROTECTOR_PROBED
    if _PROTECTOR_PROBED:
        return _PROTECTOR
    _PROTECTOR_PROBED = True
    if os.name != "nt" or not hasattr(ctypes, "windll"):
        return None
    try:
        candidate = _DpapiProtector()
        # Round-trip once so a broken DPAPI shows up as "unavailable" instead of
        # silently eating the user's API key at save time.
        if candidate.unprotect(candidate.protect(b"whochat-probe")) != b"whochat-probe":
            return None
        _PROTECTOR = candidate
    except Exception:
        _PROTECTOR = None
    return _PROTECTOR


class SecretStore:
    """Encrypted name/value store living beside ``config.json``.

    ``backend`` is either :data:`BACKEND_DPAPI` or :data:`BACKEND_UNAVAILABLE`.
    Writes are ignored when the backend is unavailable — never downgraded to a
    plaintext file.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self.path = self.directory / SECRETS_FILE_NAME

    @property
    def backend(self) -> str:
        return BACKEND_DPAPI if _protector() is not None else BACKEND_UNAVAILABLE

    @property
    def available(self) -> bool:
        return self.backend != BACKEND_UNAVAILABLE

    def get(self, name: str) -> str:
        entry = self._entries().get(name)
        protector = _protector()
        if not entry or protector is None:
            return ""
        try:
            return protector.unprotect(base64.b64decode(entry)).decode("utf-8")
        except Exception:
            return ""

    def set(self, name: str, value: str) -> bool:
        protector = _protector()
        if protector is None or not value:
            return False
        entries = self._entries()
        try:
            ciphertext = protector.protect(value.encode("utf-8"))
        except Exception:
            return False
        entries[name] = base64.b64encode(ciphertext).decode("ascii")
        return self._write(entries)

    def delete(self, name: str) -> bool:
        entries = self._entries()
        if name not in entries:
            return True
        entries.pop(name, None)
        return self._write(entries)

    def clear(self) -> bool:
        return self._write({})

    def names(self) -> list[str]:
        return sorted(self._entries())

    # ------------------------------------------------------------------ files

    def _entries(self) -> dict[str, str]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        entries = data.get("entries") if isinstance(data, dict) else None
        if not isinstance(entries, dict):
            return {}
        return {str(key): str(value) for key, value in entries.items() if isinstance(value, str)}

    def _write(self, entries: dict[str, str]) -> bool:
        payload = json.dumps(
            {"version": 1, "backend": BACKEND_DPAPI, "entries": entries},
            ensure_ascii=False,
            indent=2,
        )
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_name(f"{SECRETS_FILE_NAME}.tmp")
            temporary.write_text(payload, encoding="utf-8")
            os.replace(temporary, self.path)
        except OSError:
            return False
        return True
