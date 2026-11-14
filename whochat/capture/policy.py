from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock

from PIL import Image

from whochat.core.runtime import CaptureDecision, CapturePolicy, RuntimeState, WindowState


@dataclass
class CaptureGate:
    policy: CapturePolicy
    last_capture_ms: int = 0
    last_hash: str | None = None
    last_rect: tuple[int, int, int, int] | None = None
    last_rect_size: tuple[int, int] | None = None
    _lock: object = field(default_factory=RLock, repr=False, compare=False)

    def evaluate(self, state: RuntimeState, image_path: Path | None = None) -> CaptureDecision:
        # Polls and UI refreshes are observations, not captures. In particular,
        # checking twice on a manual click must not consume its own interval.
        with self._lock:
            return self._evaluate(state, image_path)

    def _evaluate(self, state: RuntimeState, image_path: Path | None = None) -> CaptureDecision:
        if state.paused:
            return CaptureDecision(False, "用户已暂停采集")
        if state.window.state != WindowState.VISIBLE:
            return CaptureDecision(False, f"窗口状态不可采集：{state.window.state.value}")
        if state.window.covered:
            return CaptureDecision(False, state.window.diagnostic or "目标窗口被遮挡，暂停采集")
        if state.layout is None:
            return CaptureDecision(False, "区域未识别，跳过截图")
        if state.layout.confidence < 0.5:
            return CaptureDecision(False, "区域置信度过低，等待校准")

        now_ms = int(time.monotonic() * 1000)
        elapsed = now_ms - self.last_capture_ms if self.last_capture_ms else None
        if elapsed is not None and elapsed < self.policy.screenshot_min_interval_ms:
            return CaptureDecision(False, "截图节流中", elapsed_ms=elapsed)

        rect = state.layout.message_rect.as_tuple()
        rect_size = _rect_size(rect)
        if self.last_rect_size and self.last_rect_size != rect_size and elapsed is not None and elapsed < self.policy.window_stable_delay_ms:
            return CaptureDecision(False, "窗口区域刚变化，等待稳定", elapsed_ms=elapsed)

        snapshot_hash = image_hash(image_path) if image_path else None
        if snapshot_hash and self.last_hash:
            distance = hash_distance(snapshot_hash, self.last_hash)
            if distance < self.policy.min_hash_distance:
                return CaptureDecision(False, "截图内容变化很小，跳过 OCR", snapshot_hash=snapshot_hash, elapsed_ms=elapsed)

        return CaptureDecision(True, "允许采集", snapshot_hash=snapshot_hash, elapsed_ms=elapsed)

    def record_capture(self, state: RuntimeState, *, captured_at_ms: int,
                       snapshot_hash: str | None = None) -> None:
        """Account for one successful, still-current primary screenshot only.

        The pipeline owns job/request/lifecycle validation. This method only
        updates the scheduling ledger; it never publishes identity evidence.
        """
        if state.layout is None:
            return
        with self._lock:
            if captured_at_ms <= self.last_capture_ms:
                return
            self.last_capture_ms = captured_at_ms
            self.last_rect = state.layout.message_rect.as_tuple()
            self.last_rect_size = _rect_size(self.last_rect)
            if snapshot_hash:
                self.last_hash = snapshot_hash


def _rect_size(rect: tuple[int, int, int, int]) -> tuple[int, int]:
    left, top, right, bottom = rect
    return (max(0, right - left), max(0, bottom - top))


def image_hash(path: Path, size: int = 8) -> str:
    with Image.open(path) as image:
        grayscale = image.convert("L").resize((size, size))
        pixels = list(grayscale.getdata())
    return _average_hash(pixels, size)


def image_hash_region(path: Path, box: tuple[int, int, int, int], size: int = 8) -> str:
    """Average-hash of a single region of an image.

    Used to fingerprint the conversation column: the title band that feeds OCR is
    deliberately padded leftwards as a safety net against a clipped contact name,
    and that padding overlaps the chat list.  Hashing the padded crop would make
    an unrelated chat-list repaint (a new badge, a re-ordered row) look like a new
    conversation and force a needless re-OCR.
    """
    with Image.open(path) as image:
        box = _clamp_box(box, image.size)
        if box[2] - box[0] < 1 or box[3] - box[1] < 1:
            # Degenerate region — the caller passed a band that lies outside the
            # capture, or in a different coordinate space.  Fall back to the
            # whole-image hash instead of letting PIL raise
            # "height and width must be > 0" and abort the capture job.
            grayscale = image.convert("L").resize((size, size))
            return _average_hash(list(grayscale.getdata()), size)
        grayscale = image.convert("L").crop(box).resize((size, size))
        pixels = list(grayscale.getdata())
    return _average_hash(pixels, size)


def _average_hash(pixels: list[int], size: int) -> str:
    average = sum(pixels) / max(1, len(pixels))
    bits = ["1" if pixel >= average else "0" for pixel in pixels]
    return f"{int(''.join(bits), 2):0{size * size // 4}x}"


def _clamp_box(box: tuple[int, int, int, int], size: tuple[int, int]) -> tuple[int, int, int, int]:
    width, height = size
    left = max(0, min(int(box[0]), width))
    top = max(0, min(int(box[1]), height))
    right = max(left, min(int(box[2]), width))
    bottom = max(top, min(int(box[3]), height))
    return left, top, right, bottom


def hash_distance(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()

