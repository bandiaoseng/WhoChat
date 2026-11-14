"""Local account ownership from open file names, never database contents or avatars.

Only previously resolved, manually corroborated source mappings are eligible.
Each read checks the HWND owner and process incarnation before and after a
fresh file-handle enumeration. No positive cache or avatar fallback is used.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import PureWindowsPath


@dataclass(frozen=True)
class NativeAccountProof:
    verified: bool = False
    reason: str = "native_account_evidence_unavailable"
    account_scope_id: str = ""
    process_created: float = 0.0
    stable_identifier: str = ""

    @property
    def token(self) -> tuple:
        return (self.account_scope_id, self.process_created, self.stable_identifier)


def _parts(path: str) -> tuple[str, ...]:
    item = PureWindowsPath(path)
    if not item.is_absolute() or ".." in item.parts:
        return ()
    return tuple(p.casefold() for p in item.parts)


def resolve_open_account_files(mappings, files, created: float) -> NativeAccountProof:
    """Pure matching: a unique login identifier AND its exact data directory."""
    if not math.isfinite(created) or created <= 0:
        return NativeAccountProof()
    paths = {_parts(str(p)) for p in files}
    paths.discard(())
    logins = {p[-2] for p in paths if len(p) >= 3 and p[-3] == "login"
              and p[-1] in {"key_info.db", "key_info.db-wal", "key_info.db-shm"}}
    # Detect unknown/second account roots too: matching only known roots could
    # incorrectly retain the old account during a login switch.
    roots = {p[:p.index("db_storage")+1] for p in paths if "db_storage" in p}
    if len(logins) != 1 or len(roots) != 1:
        return NativeAccountProof(reason="native_account_evidence_ambiguous" if len(logins)>1 or len(roots)>1
                                  else "native_account_evidence_unavailable")
    root = next(iter(roots))
    families = {p[len(root)] for p in paths if p[:len(root)] == root and len(p)>len(root)+1
                and p[-1].endswith((".db", ".db-wal", ".db-shm", ".kvdb", ".kvdb-wal", ".kvdb-shm"))}
    if not {"contact", "session", "message"}.issubset(families):
        return NativeAccountProof()
    matches = [m for m in mappings if str(m.get("stable_identifier", "")).casefold() in logins
               and _parts(str(m.get("source_data_directory", ""))) == root
               and m.get("mapping_status") == "resolved" and m.get("confidence") == 1.0
               and m.get("match_basis") == "decrypted_self_contact_alias_plus_existing_manual_account_scope_binding"
               and m.get("account_scope_id")]
    if len(matches) != 1:
        return NativeAccountProof(reason="native_account_mapping_unverified")
    match = matches[0]
    return NativeAccountProof(True, "native_account_verified", str(match["account_scope_id"]),
                              created, str(match["stable_identifier"]).casefold())


class NativeAccountIdentity:
    def __init__(self, mapping_reader, *, process_factory=None, owner_reader=None, file_reader=None):
        self.mapping_reader = mapping_reader
        self.process_factory = process_factory
        self.owner_reader = owner_reader
        self.file_reader = file_reader
        self._closed = False

    def close(self) -> None:
        self._closed = True
        if self.file_reader is not None:
            self.file_reader.close()

    def read(self, window) -> NativeAccountProof:
        try:
            if self._closed or not window.hwnd or not window.process_id or window.target.value != "wechat":
                return NativeAccountProof()
            if self.process_factory is None:
                import psutil
                factory = psutil.Process
            else:
                factory = self.process_factory
            if self.owner_reader is None:
                import win32gui, win32process
                def owner(hwnd):
                    if not win32gui.IsWindow(hwnd):
                        raise ValueError("native window missing")
                    return int(win32process.GetWindowThreadProcessId(hwnd)[1])
            else:
                owner = self.owner_reader
            if owner(window.hwnd) != window.process_id:
                return NativeAccountProof(reason="native_account_owner_changed")
            process = factory(window.process_id)
            created = float(process.create_time())
            executable = str(process.exe())
            expected = str(window.executable_path or "")
            if not expected or _parts(expected) != _parts(executable):
                return NativeAccountProof(reason="native_account_owner_changed")
            mappings = self.mapping_reader()
            files = (self.file_reader.read(window.process_id, created) if self.file_reader is not None
                     else [f.path for f in process.open_files()])
            # A fresh Process object avoids psutil's cached create_time value.
            after = factory(window.process_id)
            if (owner(window.hwnd) != window.process_id or float(after.create_time()) != created
                    or _parts(str(after.exe())) != _parts(executable)):
                return NativeAccountProof(reason="native_account_owner_changed")
            if self._closed:
                return NativeAccountProof()
            return resolve_open_account_files(mappings, files, created)
        except Exception:
            # Access denied, disappeared handles, and incomplete evidence never
            # fall back to avatar-based identity or an earlier positive sample.
            return NativeAccountProof()
