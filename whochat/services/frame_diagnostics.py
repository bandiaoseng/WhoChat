"""Memory-only pixel difference location; never a source of copy permission.

These summaries contain SHA-256 hashes of whole rows/columns, not pixels,
OCR text, thumbnails or recoverable image encodings.  Only numeric locations
and fixed diagnostic reasons leave ``compare_signature``.  No hashes should
be logged or persisted, including via a source proof/performance trace.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib

from PIL import Image


_MAX_AXIS = 8192
_MAX_PIXELS = 16_777_216


@dataclass(frozen=True, repr=False)
class _RegionSignature:
    bounds: tuple[int, int, int, int]
    rows: tuple[bytes, ...] = field(repr=False)
    columns: tuple[bytes, ...] = field(repr=False)


@dataclass(frozen=True, repr=False)
class FrameDiagnosticSignature:
    """Opaque transient diagnostic state; never put in persisted proof data."""

    binding: tuple = field(repr=False)
    title: _RegionSignature = field(repr=False)
    messages: _RegionSignature = field(repr=False)


def _rect(value) -> tuple[int, int, int, int]:
    rect = value.as_tuple()
    if (type(rect) is not tuple or len(rect) != 4
            or any(type(number) is not int for number in rect)
            or rect[0] >= rect[2] or rect[1] >= rect[3]):
        raise ValueError("diagnostic_geometry_invalid")
    return rect


def _binding(image, layout, snapshot_hash, job_id, title_rect):
    if (not isinstance(image, Image.Image) or type(job_id) is not int or job_id < 1
            or not isinstance(snapshot_hash, str) or len(snapshot_hash) != 32
            or any(char not in "0123456789abcdef" for char in snapshot_hash)):
        raise ValueError("diagnostic_source_invalid")
    outer = _rect(layout.window_rect)
    if (image.size != (outer[2] - outer[0], outer[3] - outer[1])
            or image.width > _MAX_AXIS or image.height > _MAX_AXIS
            or image.width * image.height > _MAX_PIXELS):
        raise ValueError("diagnostic_frame_invalid")
    title = _rect(title_rect if title_rect is not None else layout.title_rect)
    messages = _rect(layout.message_rect)
    inputs = _rect(layout.input_rect)
    client_value = getattr(layout, "client_rect", None)
    client = _rect(client_value) if client_value is not None else None
    for area in (title, messages, inputs, client):
        if area is not None and not (
            outer[0] <= area[0] < area[2] <= outer[2]
            and outer[1] <= area[1] < area[3] <= outer[3]
        ):
            raise ValueError("diagnostic_geometry_outside_frame")
    local = lambda area: (area[0] - outer[0], area[1] - outer[1],
                          area[2] - outer[0], area[3] - outer[1])
    return ((snapshot_hash, job_id, outer, client, title, messages, inputs,
             str(layout.target_app.value), str(layout.bubble_profile),
             bool(layout.input_boundary_verified)), local(title), local(messages))


def _region_signature(rgb: Image.Image, bounds) -> _RegionSignature:
    crop = rgb.crop(bounds)
    transposed = None
    try:
        # A transpose turns columns into contiguous rows.  This avoids a Python
        # pixel loop and thousands of tiny image crops; borrowed RGB is intact.
        raw = memoryview(crop.tobytes())
        stride = crop.width * 3
        rows = tuple(hashlib.sha256(raw[start:start + stride]).digest()
                     for start in range(0, len(raw), stride))
        transposed = crop.transpose(Image.Transpose.TRANSPOSE)
        raw_columns = memoryview(transposed.tobytes())
        stride = crop.height * 3
        columns = tuple(hashlib.sha256(raw_columns[start:start + stride]).digest()
                        for start in range(0, len(raw_columns), stride))
        return _RegionSignature(bounds, rows, columns)
    finally:
        if transposed is not None:
            transposed.close()
        crop.close()


def _sign_regions(image, title, messages):
    rgb = image if image.mode == "RGB" else image.convert("RGB")
    try:
        return _region_signature(rgb, title), _region_signature(rgb, messages)
    finally:
        if rgb is not image:
            rgb.close()


def build_signature(image, layout, snapshot_hash, job_id, *, title_rect=None):
    """Return an opaque in-memory anchor or None, without changing any guard.

    Production callers pass their existing identity-title rectangle explicitly;
    the optional default is useful for callers that diagnose a whole title band.
    A diagnostic failure never changes acceptance of the actual source proof.
    """
    try:
        binding, title, messages = _binding(image, layout, snapshot_hash, job_id, title_rect)
        title_signature, message_signature = _sign_regions(image, title, messages)
        return FrameDiagnosticSignature(binding, title_signature, message_signature)
    except (AttributeError, TypeError, ValueError, OSError, MemoryError):
        return None


def _difference(original: _RegionSignature, current: _RegionSignature) -> dict:
    if (original.bounds != current.bounds or len(original.rows) != len(current.rows)
            or len(original.columns) != len(current.columns)):
        raise ValueError("diagnostic_signature_shape_invalid")
    rows = [index for index, pair in enumerate(zip(original.rows, current.rows))
            if pair[0] != pair[1]]
    columns = [index for index, pair in enumerate(zip(original.columns, current.columns))
               if pair[0] != pair[1]]
    if bool(rows) != bool(columns):
        raise ValueError("diagnostic_signature_inconsistent")
    left, top, _, _ = original.bounds
    return {
        "bounds": list(original.bounds),
        "raw_equal": not rows,
        "changed_rows": len(rows),
        "changed_columns": len(columns),
        "changed_bbox": ([left + columns[0], top + rows[0],
                          left + columns[-1] + 1, top + rows[-1] + 1]
                         if rows else None),
    }


def compare_signature(signature, image, layout, snapshot_hash, job_id, *, title_rect=None):
    """Locate raw differences only; no tolerance, masking or authorization.

    The bounding box is in full-frame local coordinates. Row/column counts
    are not a changed-pixel count: that count cannot be recovered from hashes.
    No source/customer identifiers, hashes, text or image data are returned.
    """
    if not isinstance(signature, FrameDiagnosticSignature):
        return {"schema": 1, "available": False, "reason": "source_signature_unavailable"}
    try:
        binding, title, messages = _binding(image, layout, snapshot_hash, job_id, title_rect)
        if signature.binding != binding:
            return {"schema": 1, "available": False, "reason": "source_binding_mismatch"}
        title_signature, message_signature = _sign_regions(image, title, messages)
        return {
            "schema": 1,
            "available": True,
            "frame_size": [image.width, image.height],
            "title": _difference(signature.title, title_signature),
            "messages": _difference(signature.messages, message_signature),
        }
    except (AttributeError, TypeError, ValueError, OSError, MemoryError):
        return {"schema": 1, "available": False, "reason": "source_signature_invalid"}
