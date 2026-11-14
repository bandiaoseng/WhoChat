"""Disposable local OCR precomputation, never customer or AI authority.

Only immutable raw OCR is retained.  Consumers must capture fresh pixels and
run the ordinary identity/source pipeline.  This module does not submit a
pipeline job, publish a signal, observe an account, persist a customer or call
AI.  The caller schedules ticks only while manual generation is idle.
"""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from dataclasses import asdict, is_dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import tempfile
from threading import RLock
import time

from PIL import Image

from whochat.core.runtime import Rect
from whochat.ocr.models import OcrResult
from whochat.services.account_scope import RawOcrAccountPreparation


_POLICY = "local_raw_ocr_exact_v1"
_KINDS = ("title", "content")


def _value(value):
    return getattr(value, "value", value)


def _state_key(state):
    window = state.window
    return (window.hwnd, window.process_id, state.window_generation,
            state.capture_generation, state.account_identity_epoch,
            state.account_identity_runtime_key, window.rect, window.client_rect)


def _computation_owner_key(state):
    # A completed pure OCR computation is not an in-flight capture lease.
    # Focus handoff increments capture_generation, even when all OCR pixels
    # are identical.  Reusing only that computation is safe after a fresh
    # exact-pixel key comparison; the caller still creates new job-bound
    # source proofs and performs both independent on-site title observations.
    # _state_key/_alive retain capture_generation to reject late pre-read work.
    window = state.window
    return (window.hwnd, window.process_id, state.window_generation,
            state.account_identity_epoch, state.account_identity_runtime_key,
            window.rect, window.client_rect)


def _eligible(state):
    try:
        window = state.window
        return bool(not state.paused and _value(window.target) == "wechat"
                    and _value(window.state) == "visible" and window.foreground
                    and not window.covered and type(window.hwnd) is int
                    and type(window.process_id) is int and window.hwnd > 0
                    and window.process_id > 0 and isinstance(window.rect, Rect)
                    and isinstance(window.client_rect, Rect) and state.layout is not None
                    and state.layout.window_rect == window.rect
                    and state.layout.client_rect == window.client_rect)
    except (AttributeError, TypeError):
        return False


def _account_key(state, account):
    # A preparation is permitted only as a raw computation partition. It has
    # no trusted identity and cannot authorize any pipeline/action consumer.
    prepared = type(account) is RawOcrAccountPreparation
    trusted = (getattr(account, "actions_allowed", None) is True
               or _value(getattr(account, "identity_state", None)) == "trusted")
    values = (getattr(account, "account_scope_id", ""),
              getattr(account, "target_instance_id", ""),
              getattr(account, "identity_epoch", None),
              getattr(account, "runtime_key", ""))
    if ((not trusted and not prepared) or not values[0] or not values[1]
            or values[2] != state.account_identity_epoch
            or values[3] != state.account_identity_runtime_key or not values[3]):
        raise ValueError("preread_account_not_trusted")
    return values


def _serialized(value):
    payload = asdict(value) if is_dataclass(value) else value
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, default=str,
                      separators=(",", ":"), allow_nan=False)


def _region_hash(frame, layout, rect):
    origin = layout.window_rect
    bounds = (rect.left - origin.left, rect.top - origin.top,
              rect.right - origin.left, rect.bottom - origin.top)
    if not (0 <= bounds[0] < bounds[2] <= frame.width
            and 0 <= bounds[1] < bounds[3] <= frame.height):
        raise ValueError("preread_crop_outside_frame")
    with frame.crop(bounds) as cropped:
        digest = hashlib.sha256()
        digest.update(f"{cropped.mode}:{cropped.width}:{cropped.height}:".encode("ascii"))
        digest.update(cropped.tobytes())
        return digest.hexdigest()


def _cache_key(state, account, layout, frame, engine):
    # Delayed imports avoid pipeline -> cache -> pipeline import recursion.
    from whochat.services.account_scope import account_avatar_screen_rect
    from whochat.services.pipeline import _prepare_title_ocr_input, _prepare_ocr_input

    if (not _eligible(state) or layout.window_rect != state.window.rect
            or layout.client_rect != state.window.client_rect
            or layout.input_boundary_verified is not True
            or frame.size != (layout.window_rect.width, layout.window_rect.height)):
        raise ValueError("preread_layout_unavailable")
    account_key = _account_key(state, account)
    dummy = Path("preread-key.png")
    title = _prepare_title_ocr_input(dummy, layout, materialize=False)
    content = _prepare_ocr_input(dummy, layout, materialize=False)
    # Include the actual OCR input pixels, not a masked/perceptual subset.  A
    # caret/draft repaint may cause a conservative miss, never stale text reuse.
    regions = (account_avatar_screen_rect(layout), layout.title_rect,
               layout.message_rect, title[3] or layout.window_rect,
               content[3] or layout.window_rect)
    return (_POLICY, _computation_owner_key(state), account_key, _serialized(layout),
            id(engine), type(engine).__module__, type(engine).__qualname__,
            str(getattr(engine, "name", "")), _serialized(getattr(engine, "config", None)),
            os.environ.get("WHOCHAT_OCR_CROP_CONTENT", "1"),
            _serialized(title[1]), title[2], _serialized(content[1]), content[2],
            tuple(_region_hash(frame, layout, rect) for rect in regions))


class RawOcrCache:
    """Two frame entries, hard TTL, immutable-on-read raw OCR only."""

    def __init__(self, *, clock=time.monotonic, ttl_seconds=15.0,
                 capacity=2, max_bytes=512 * 1024):
        self.clock = clock
        self.ttl = min(15.0, max(0.001, float(ttl_seconds)))
        self.capacity = min(2, max(1, int(capacity)))
        self.max_bytes = min(512 * 1024, max(1, int(max_bytes)))
        self._entries = OrderedDict()
        self._lock = RLock()
        self._hits = {kind: 0 for kind in _KINDS}
        self._misses = {kind: 0 for kind in _KINDS}

    def _prune(self):
        now = self.clock()
        for key, entry in tuple(self._entries.items()):
            if entry[0] <= now:
                self._entries.pop(key, None)

    def _contains(self, key):
        with self._lock:
            self._prune()
            return key in self._entries

    def _put(self, key, results):
        if set(results) != {"content"}:
            return False
        stored = {}
        for kind, result in results.items():
            if (type(result) is not OcrResult or result.warning or not result.boxes
                    or len(result.boxes) > 512):
                return False
            stored[kind] = replace(deepcopy(result), source_image="")
        size = len(_serialized(stored).encode("utf-8")) + len(repr(key).encode("utf-8"))
        if size > self.max_bytes:
            return False
        with self._lock:
            self._prune()
            self._entries[key] = (self.clock() + self.ttl, stored, size)
            self._entries.move_to_end(key)
            while (len(self._entries) > self.capacity
                   or sum(item[2] for item in self._entries.values()) > self.max_bytes):
                self._entries.popitem(last=False)
        return True

    def lookup(self, kind, state, account, layout, frame, engine):
        # StableIdentityLock requires independent on-site title OCR for both
        # button-owned frames.  Pre-read never supplies title inference.
        if kind != "content":
            return None
        try:
            key = _cache_key(state, account, layout, frame, engine)
        except Exception:
            with self._lock:
                self._misses[kind] += 1
            return None
        with self._lock:
            self._prune()
            entry = self._entries.get(key)
            if entry is None:
                self._misses[kind] += 1
                return None
            self._hits[kind] += 1
            # Neither a lookup nor LRU movement renews the original deadline.
            self._entries.move_to_end(key)
            return deepcopy(entry[1][kind])

    def remember_content(self, state, account, layout, frame, engine, result):
        """Keep a fresh worker's raw computation, never its parser/proof result.

        The same account, engine, layout and exact input-pixel key used for
        pre-reading is required. An incomplete parse may still have reusable
        OCR pixels; the next capture independently reruns its parser and all
        identity/source checks. Callers must not pass locally recovered text.
        """
        try:
            key = _cache_key(state, account, layout, frame, engine)
            return self._put(key, {"content": result})
        except Exception:
            return False

    def clear(self):
        with self._lock:
            self._entries.clear()

    def status_snapshot(self):
        with self._lock:
            self._prune()
            return {"entries": len(self._entries), "bytes": sum(e[2] for e in self._entries.values()),
                    "title_hits": self._hits["title"], "content_hits": self._hits["content"],
                    "title_misses": self._misses["title"], "content_misses": self._misses["content"],
                    "ttl_seconds": self.ttl, "capacity": self.capacity}


class _Cancelled(Exception):
    pass


class LocalPrereader:
    """Caller-owned idle scheduler sharing the pipeline's single executor.

    tick resumes after pause; the caller MUST NOT tick during manual work.
    pause cancels queued work and invalidates in-flight results without waiting
    for or killing native OCR.  Completed cache entries remain reusable.
    """

    def __init__(self, pipeline, *, native_reader=None, clock=time.monotonic,
                 temp_root=None, interval_seconds=0.75):
        self.pipeline = pipeline
        self.clock = clock
        self.native_reader = native_reader
        self.temp_root = temp_root
        self.interval = max(0.75, float(interval_seconds))
        self.cache = RawOcrCache(clock=clock)
        self._lock = RLock()
        self._generation = 0
        self._future = None
        self._closed = False
        self._paused = False
        self._next_tick = 0.0
        self._status = "idle"
        self._counts = {"submitted": 0, "completed": 0, "unchanged": 0,
                        "discarded": 0, "failed": 0, "ocr_calls": 0}
        self._preload_engine = None
        self._preload_status = "not_started"
        self.start_model_preload()

    def start_model_preload(self):
        """Optional model load shares the OCR executor and shutdown future."""
        with self._lock:
            engine = self.pipeline.ocr_engine
            preload = getattr(engine, "preload_local_models", None)
            if (self._closed or engine is self._preload_engine or not callable(preload)
                    or self.pipeline.is_running
                    or (self._future is not None and not self._future.done())):
                return False
            executor = self.pipeline.executor
            self._generation += 1
            self._paused = False
            generation = self._generation
            self._preload_engine = engine
            self._preload_status = "queued"
            try:
                self._future = executor.submit(self._run_model_preload,
                    generation, executor, engine, preload)
                return True
            except Exception:
                self._preload_status = "submit_failed"
                return False

    def _run_model_preload(self, generation, executor, engine, preload):
        def current():
            with self._lock:
                return bool(not self._closed and generation == self._generation
                    and executor is self.pipeline.executor and engine is self.pipeline.ocr_engine)
        if not current():
            return
        with self._lock:
            self._preload_status = "loading"
        try:
            result = preload(is_current=current)
            status = result.get("status", "failed")
        except Exception:
            status = "failed"
        with self._lock:
            if current():
                self._preload_status = status if status in {"ready", "failed", "cancelled", "unsupported"} else "failed"
            elif engine is self._preload_engine:
                self._preload_status = "cancelled"

    def _checkpoint(self, state):
        reader = getattr(getattr(self.pipeline, "account_scope_resolver", None),
                         "account_identity_checkpoint", None)
        checkpoint = reader(state.window) if callable(reader) else None
        _account_key(state, checkpoint)
        if (getattr(checkpoint, "hwnd", None) != state.window.hwnd
                or getattr(checkpoint, "process_id", None) != state.window.process_id):
            raise _Cancelled()
        return checkpoint

    def _can_prepare(self, state):
        resolver = getattr(self.pipeline, "account_scope_resolver", None)
        try:
            return bool(getattr(resolver, "native_identity", None) is not None
                and callable(getattr(resolver, "read_raw_ocr_preparation", None))
                and resolver.account_identity_checkpoint(state.window) is None)
        except Exception:
            return False

    def _preparation(self, state):
        resolver = self.pipeline.account_scope_resolver
        prepared = resolver.read_raw_ocr_preparation(state.window,
            expected_identity_epoch=state.account_identity_epoch,
            expected_runtime_key=state.account_identity_runtime_key)
        if (type(prepared) is not RawOcrAccountPreparation
                or prepared.hwnd != state.window.hwnd
                or prepared.process_id != state.window.process_id):
            raise _Cancelled()
        _account_key(state, prepared)
        return prepared

    def tick(self, state):
        # Also covers an OCR engine replaced by settings or watchdog recovery.
        # No retry loop for an engine whose optional preload already failed.
        if self.start_model_preload():
            return False
        with self._lock:
            if (self._closed or self.clock() < self._next_tick
                    or (self._future is not None and not self._future.done())
                    or self.pipeline.is_running):
                return False
            self._next_tick = self.clock() + self.interval
            if not _eligible(state):
                return False
            try:
                checkpoint = self._checkpoint(state)
            except Exception:
                if not self._can_prepare(state):
                    self._status = "account_unavailable"
                    return False
                checkpoint = None
            self._generation += 1
            self._paused = False
            generation = self._generation
            executor = self.pipeline.executor
            engine = self.pipeline.ocr_engine
            self._status = "queued"
            self._counts["submitted"] += 1
            try:
                self._future = executor.submit(self._run, deepcopy(state), checkpoint,
                                               generation, executor, engine)
            except Exception:
                self._status = "submit_failed"
                self._counts["failed"] += 1
                return False
            return True

    def pause(self):
        with self._lock:
            if self._paused:
                return
            self._paused = True
            self._generation += 1
            self._status = "paused"
            if self._preload_status in {"queued", "loading"}:
                self._preload_status = "cancelled"
            if self._future is not None:
                self._future.cancel()

    def shutdown(self):
        with self._lock:
            # Return this worker's future so its owner can defer destruction of
            # the shared OCR engine until a running native call has unwound.
            # Do not wait on the UI thread or invalidate a replacement engine.
            future = self._future
            self._closed = True
            self.pause()
            self.cache.clear()
            return future

    @property
    def is_running(self):
        with self._lock:
            return bool(self._future is not None and not self._future.done())

    def status_snapshot(self):
        with self._lock:
            cached = self.cache.status_snapshot()
            return {"local_only": True, "enabled": not self._closed, "paused": self._paused,
                    "model_preload": self._preload_status,
                    "status": self._status, "last_reason": self._status, "closed": self._closed,
                    "running": bool(self._future is not None and not self._future.done()),
                    **self._counts, "cache": cached, "cached_frames": cached["entries"],
                    "cache_hits": cached["title_hits"] + cached["content_hits"]}

    def _alive(self, state, checkpoint, generation, executor, engine):
        with self._lock:
            if (self._closed or generation != self._generation
                    or executor is not self.pipeline.executor or engine is not self.pipeline.ocr_engine):
                raise _Cancelled()
        current_reader = getattr(self.pipeline, "current_state_provider", None)
        current = current_reader() if callable(current_reader) else None
        if current is None or not _eligible(current) or _state_key(current) != _state_key(state):
            raise _Cancelled()
        if checkpoint is None or type(checkpoint) is RawOcrAccountPreparation:
            if not self._can_prepare(current):
                raise _Cancelled()
        elif self._checkpoint(current) != checkpoint:
            raise _Cancelled()

    def _native(self, state):
        reader = self.native_reader
        if reader is None:
            from whochat.services.manual_suggestions import read_native_window
            reader = read_native_window
        native = reader(state.window.hwnd)
        if (native.hwnd != state.window.hwnd or native.process_id != state.window.process_id
                or native.rect != state.window.rect.as_tuple()
                or native.client_rect != state.window.client_rect.as_tuple()
                or native.foreground is not True or native.visible is not True
                or native.minimized is not False
                or native.foreground_hwnd != state.window.hwnd
                or native.foreground_process_id != state.window.process_id):
            raise _Cancelled()
        return native

    def _root(self):
        if self.temp_root is None:
            from whochat.core.paths import app_data_dir
            root = app_data_dir() / "capture" / "local_preread"
        else:
            root = Path(self.temp_root)
        root = root.resolve()
        if root.drive.casefold() != "f:":
            raise ValueError("preread_requires_f_storage")
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _run(self, state, checkpoint, generation, executor, engine):
        from whochat.services.pipeline import (
            _prepare_title_ocr_input, _prepare_ocr_input, _offset_ocr_result,
            refine_layout_from_image, constrain_layout_to_client,
        )
        args = (state, checkpoint, generation, executor, engine)
        try:
            self._alive(*args)
            if checkpoint is None:
                checkpoint = self._preparation(state)
                args = (state, checkpoint, generation, executor, engine)
                self._alive(*args)
            native = self._native(state)
            with tempfile.TemporaryDirectory(prefix="frame_", dir=self._root()) as folder:
                path = Path(folder) / "frame.png"
                self._alive(*args)
                actual = Path(self.pipeline.capture_func(state.window.rect.as_tuple(), path)).resolve()
                if actual != path.resolve():
                    raise _Cancelled()
                self._alive(*args)
                if self._native(state) != native:
                    raise _Cancelled()
                layout = constrain_layout_to_client(refine_layout_from_image(path, state.layout),
                                                    state.window.client_rect)
                with Image.open(path) as decoded:
                    frame = decoded.copy()
                try:
                    key = _cache_key(state, checkpoint, layout, frame, engine)
                    if self.cache._contains(key):
                        with self._lock:
                            self._alive(*args)
                            self._counts["unchanged"] += 1
                            self._status = "unchanged"
                        self._prepare_title_geometry(path, layout, frame, args)
                        return
                    results = {}
                    for kind, prepare in (("content", _prepare_ocr_input),):
                        self._alive(*args)
                        if self._native(state) != native:
                            raise _Cancelled()
                        crop_path, crop_layout, offset, _ = prepare(path, layout, frame_image=frame)
                        with self._lock:
                            self._alive(*args)
                            self._status = "reading_" + kind
                            self._counts["ocr_calls"] += 1
                        raw = engine.recognize(crop_path, crop_layout)
                        self._alive(*args)
                        results[kind] = _offset_ocr_result(raw, offset, path)
                    verify_path = Path(folder) / "verify.png"
                    self._alive(*args)
                    if self._native(state) != native:
                        raise _Cancelled()
                    actual = Path(self.pipeline.capture_func(state.window.rect.as_tuple(), verify_path)).resolve()
                    self._alive(*args)
                    if actual != verify_path.resolve() or self._native(state) != native:
                        raise _Cancelled()
                    with Image.open(verify_path) as verify:
                        if _cache_key(state, checkpoint, layout, verify, engine) != key:
                            raise _Cancelled()
                    if type(checkpoint) is RawOcrAccountPreparation:
                        if self._preparation(state) != checkpoint or self._native(state) != native:
                            raise _Cancelled()
                    with self._lock:
                        self._alive(*args)
                        if self.cache._put(key, results):
                            self._counts["completed"] += 1
                            self._status = "ready"
                        else:
                            self._counts["discarded"] += 1
                            self._status = "ocr_not_cacheable"
                    self._prepare_title_geometry(path, layout, frame, args)
                finally:
                    frame.close()
        except _Cancelled:
            with self._lock:
                self._counts["discarded"] += 1
                if generation == self._generation:
                    self._status = "discarded"
        except Exception:
            with self._lock:
                self._counts["failed"] += 1
                if generation == self._generation:
                    self._status = "failed"

    def _prepare_title_geometry(self, path, layout, frame, args):
        """Only disposable detector boxes; account/source state stays untouched."""
        engine = args[-1]
        prepare = getattr(engine, "prepare_title_detection", None)
        if not callable(prepare):
            return
        self._alive(*args)
        from whochat.services.pipeline import _prepare_title_ocr_input
        crop, crop_layout, _offset, _rect = _prepare_title_ocr_input(path, layout, frame_image=frame)
        def current():
            try:
                self._alive(*args)
                return True
            except _Cancelled:
                return False
        # The same executor owns body OCR and this optional geometry work.
        # Do not hold the UI-facing lock during inference or retain any title.
        try:
            prepare(crop, crop_layout, is_current=current)
        except Exception:
            pass
        self._alive(*args)
