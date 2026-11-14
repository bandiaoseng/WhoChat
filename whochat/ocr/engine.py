from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import replace
import json
import math
from numbers import Real
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
from typing import Any

from PIL import Image

from whochat.config import OcrConfig
from whochat.core.paths import app_data_dir
from whochat.core.runtime import LayoutRegions, Rect
from whochat.diagnostics import append_diagnostics_log, configure_native_runtime_limits
from whochat.ocr.models import OcrRegion, OcrResult, OcrTextBox


# Bound ONNX's per-model CPU pools; detector/classifier/recognizer otherwise
# each select their own automatic pool. This affects scheduling only: the
# models, source pixels, preprocessing and recognition thresholds stay intact.
RAPID_OCR_CPU_THREADS = 4


def _rapid_ocr_thread_params() -> dict[str, int]:
    cpu_count = os.cpu_count()
    # RapidOCR validates explicit thread counts against os.cpu_count() itself.
    # Unknown/invalid counts must keep its old defaults, not invent capacity.
    if type(cpu_count) is not int or cpu_count <= 0:
        return {}
    return {
        "EngineConfig.onnxruntime.intra_op_num_threads": min(RAPID_OCR_CPU_THREADS, cpu_count),
        "EngineConfig.onnxruntime.inter_op_num_threads": 1,
    }


class OcrEngine(ABC):
    name: str

    @abstractmethod
    def recognize(self, image_path: Path, layout: LayoutRegions) -> OcrResult:
        raise NotImplementedError


class PreviewOcrEngine(OcrEngine):
    name = "preview-fixture"

    def recognize(self, image_path: Path, layout: LayoutRegions) -> OcrResult:
        if not image_path.exists():
            return OcrResult([], str(image_path), self.name, "截图不存在，无法生成 OCR 预览")
        with Image.open(image_path) as image:
            width, height = image.size
        if width < 40 or height < 40:
            return OcrResult([], str(image_path), self.name, "截图尺寸过小")

        boxes = [
            OcrTextBox("标题候选", _box_inside(layout.title_rect, 0.04, 0.25, 0.24, 0.68), 0.72, OcrRegion.TITLE, self.name),
            OcrTextBox("聊天列表候选", _box_inside(layout.chat_list_rect, 0.08, 0.06, 0.68, 0.13), 0.68, OcrRegion.CHAT_LIST, self.name),
            OcrTextBox("对方消息候选", _box_inside(layout.message_rect, 0.08, 0.12, 0.42, 0.22), 0.66, OcrRegion.MESSAGE, self.name),
            OcrTextBox("我的消息候选", _box_inside(layout.message_rect, 0.58, 0.28, 0.94, 0.38), 0.66, OcrRegion.MESSAGE, self.name),
            OcrTextBox("输入区候选", _box_inside(layout.input_rect, 0.06, 0.22, 0.38, 0.46), 0.61, OcrRegion.INPUT, self.name),
        ]
        return OcrResult(
            boxes=boxes,
            source_image=str(image_path),
            engine=self.name,
            warning="当前为校准预览引擎，不代表真实 OCR 结果",
        )


class RapidOcrEngine(OcrEngine):
    name = "rapidocr"

    def __init__(self, config: OcrConfig) -> None:
        self.config = config
        self._engine = None
        self._title_engine: RapidOcrEngine | None = None
        self.last_timings: dict[str, int] = {}

    def recognize_title(self, image_path: Path, layout: LayoutRegions) -> OcrResult:
        """Title-only detection sizing; body/line settings remain untouched.

        The returned boxes are fresh OCR evidence, never an identity or a cached
        customer name. Ordinary pipeline ownership and two-frame checks still
        decide whether those boxes may be used. Unsupported/custom small limits
        retain the ordinary engine. Uncertain fast results use the original
        configuration without lowering any recognition confidence threshold.
        """
        limit = self.config.detector_min_side
        if type(limit) is not int or limit <= 256:
            return self.recognize(image_path, layout)
        # The pipeline can fall back to a full-window image when cropping is
        # unavailable or disabled. Never apply title sizing to that fallback.
        title = getattr(layout, "title_rect", None)
        try:
            with Image.open(image_path) as source:
                width, height = source.size
            title_crop = (isinstance(title, Rect) and title.top == 0
                          and title.height == height and width >= 120 and height >= 28
                          and 0 <= title.left < width and title.right >= width)
        except Exception:
            title_crop = False
        if not title_crop:
            return self.recognize(image_path, layout)
        started = time.monotonic()
        title_config = replace(self.config, detector_min_side=256)
        if self._title_engine is None or self._title_engine.config != title_config:
            self._title_engine = RapidOcrEngine(title_config)
        result = self._title_engine.recognize(image_path, layout)
        fast_ms = _elapsed_ms(started)
        fast_timings = dict(self._title_engine.last_timings)
        # This is only a computation-routing rule, not an acceptance threshold.
        # All actual scores are retained and the existing title gate still runs.
        fallback = bool(result.warning or not result.boxes or
                        any(box.confidence < 0.95 for box in result.boxes))
        if fallback:
            result = self.recognize(image_path, layout)
            timings = dict(self.last_timings)
        else:
            timings = fast_timings
        self.last_timings = {
            **timings, "title_detector_min_side": 256,
            "title_fast_attempt_ms": fast_ms, "title_sizing_fallback": int(fallback),
            "total_ms": _elapsed_ms(started),
        }
        return result

    def preload_local_models(self, *, is_current=lambda: True):
        """Load the unchanged local models on their owning worker, without OCR.

        The owner checks cancellation between model loads. An early click can
        cancel the optional remaining work; ordinary lazy loading still works.
        No image, inference result or account authority is created here.
        """
        if type(self) is not RapidOcrEngine:
            return {"status": "unsupported", "models_ready": 0}
        count = 0
        try:
            if not is_current():
                return {"status": "cancelled", "models_ready": count}
            self._get_engine()
            count += 1
            if not is_current():
                return {"status": "cancelled", "models_ready": count}
            limit = self.config.detector_min_side
            if type(limit) is int and limit > 256:
                config = replace(self.config, detector_min_side=256)
                if self._title_engine is None or self._title_engine.config != config:
                    self._title_engine = RapidOcrEngine(config)
                self._title_engine._get_engine()
                count += 1
            return {"status": "ready" if is_current() else "cancelled", "models_ready": count}
        except Exception:
            return {"status": "failed", "models_ready": count}

    def recognize(self, image_path: Path, layout: LayoutRegions) -> OcrResult:
        if not image_path.exists():
            return OcrResult([], str(image_path), self.name, "截图不存在，无法执行 RapidOCR")
        call_started = time.monotonic()
        try:
            init_started = time.monotonic()
            engine = self._get_engine()
            engine_init_ms = _elapsed_ms(init_started)
            preprocess_started = time.monotonic()
            image, scale_x, scale_y = _prepare_ocr_image(image_path, self.config)
            preprocess_ms = _elapsed_ms(preprocess_started)
            inference_started = time.monotonic()
            raw = engine(image)
            inference_ms = _elapsed_ms(inference_started)
        except Exception as exc:
            self.last_timings = {"total_ms": _elapsed_ms(call_started)}
            return OcrResult([], str(image_path), self.name, f"RapidOCR 不可用：{exc}")
        postprocess_started = time.monotonic()
        boxes = _boxes_from_rapid_result(raw, self.config.min_confidence, self.name)
        if scale_x != 1.0 or scale_y != 1.0:
            boxes = _scale_ocr_boxes(boxes, scale_x, scale_y)
        self.last_timings = {
            "engine_init_ms": engine_init_ms,
            "preprocess_ms": preprocess_ms,
            "inference_ms": inference_ms,
            "postprocess_ms": _elapsed_ms(postprocess_started),
            "total_ms": _elapsed_ms(call_started),
        }
        return OcrResult(boxes, str(image_path), self.name, None if boxes else "RapidOCR 未返回文本")

    def prepare_title_detection(self, image_path: Path, layout: LayoutRegions, *, is_current=lambda: True):
        """Idle-worker geometry only; no recognized title or identity escapes.

        Prepare both existing detector sizes, with cancellation between them.
        An early click can cancel the remaining optional work. Each later
        title call still classifies and recognizes its own freshly read image.
        """
        if type(self) is not RapidOcrEngine or not is_current():
            return False
        try:
            title = getattr(layout, "title_rect", None)
            with Image.open(image_path) as source:
                width, height = source.size
            if not (isinstance(title, Rect) and title.top == 0 and title.height == height
                    and width >= 120 and height >= 28 and 0 <= title.left < width
                    and title.right >= width):
                return False
            engines = [self]
            limit = self.config.detector_min_side
            if type(limit) is int and limit > 256:
                config = replace(self.config, detector_min_side=256)
                if self._title_engine is None or self._title_engine.config != config:
                    self._title_engine = RapidOcrEngine(config)
                engines.insert(0, self._title_engine)
            from whochat.ocr.detection_cache import prepare_detection
            prepared = False
            for engine in engines:
                if not is_current():
                    return False
                core = engine._get_engine()
                image, _scale_x, _scale_y = _prepare_ocr_image(image_path, engine.config)
                try:
                    prepared = prepare_detection(core, image) or prepared
                finally:
                    image.close()
            return prepared and is_current()
        except Exception:
            # Optional precomputation never blocks normal lazy recognition.
            return False

    def recognize_line(self, image: Image.Image) -> tuple[str, float] | None:
        """Read text in the supplied full crop without detecting new boxes.

        This is recognition evidence only, not proof of crop coverage or emoji
        meaning. The caller owns those checks and the stricter consensus gate.
        Never call RapidOCR's parameter-mutating public entry point here:
        detection/classification settings and ordinary-call timings are shared
        with subsequent title, content and local-preread work.
        """
        if not isinstance(image, Image.Image) or image.width < 1 or image.height < 1:
            return None
        try:
            from rapidocr.ch_ppocr_rec.typings import TextRecInput

            engine = self._get_engine()
            # Preserve every source pixel and protect the caller's live frame.
            # load_img converts this copy to the recognizer's own BGR array.
            with image.copy() as source:
                raw = engine.text_rec(TextRecInput(
                    img=[engine.load_img(source)], return_word_box=False))
            if raw is None or getattr(raw, "warning", None):
                return None
            texts, scores = getattr(raw, "txts", None), getattr(raw, "scores", None)
            if (not isinstance(texts, (tuple, list)) or len(texts) != 1
                    or not isinstance(scores, (tuple, list)) or len(scores) != 1
                    or not isinstance(texts[0], str) or not texts[0].strip()):
                return None
            score = scores[0]
            if (isinstance(score, bool) or not isinstance(score, Real)
                    or not math.isfinite(score) or not 0 <= score <= 1):
                return None
            # Return the actual score, including low confidence. The adapter
            # must neither promote it nor substitute a configurable threshold.
            return texts[0].strip(), float(score)
        except Exception as exc:
            append_diagnostics_log(
                "ocr_worker", f"rapid_line_recognition_failed error_type={type(exc).__name__}")
            return None

    def _get_engine(self):
        if self._engine is None:
            from rapidocr import RapidOCR

            self._engine = RapidOCR(
                params={
                    "Det.limit_side_len": int(self.config.detector_min_side),
                    "Det.limit_type": "min",
                    **_rapid_ocr_thread_params(),
                }
            )
            from whochat.ocr.detection_cache import attach_detection_cache
            attach_detection_cache(self._engine)
        return self._engine


class PaddleOcrEngine(OcrEngine):
    name = "paddleocr"

    def __init__(self, config: OcrConfig) -> None:
        self.config = config
        self._engine = None
        self._failure_count = 0
        self._unavailable_until = 0.0
        self._daemon: subprocess.Popen | None = None
        self._daemon_stdout: queue.Queue[str] | None = None
        self._daemon_lock = threading.Lock()

    def recognize(self, image_path: Path, layout: LayoutRegions) -> OcrResult:
        if not image_path.exists():
            return OcrResult([], str(image_path), self.name, "截图不存在，无法执行 PaddleOCR")
        now = time.monotonic()
        if self._unavailable_until > now:
            remaining = int(self._unavailable_until - now)
            return OcrResult([], str(image_path), self.name, f"PaddleOCR 暂停中：连续失败后熔断，约 {remaining}s 后重试")
        worker_mode = os.environ.get("WHOCHAT_PADDLEOCR_WORKER_MODE", "daemon").strip().lower()
        if worker_mode == "daemon":
            return self._record_health(self._recognize_in_daemon(image_path))
        if os.environ.get("WHOCHAT_PADDLEOCR_SUBPROCESS", "1") != "0":
            return self._record_health(self._recognize_in_subprocess(image_path))
        try:
            engine = self._get_engine()
            raw = _run_paddle_ocr(engine, image_path)
        except Exception as exc:
            return self._record_health(OcrResult([], str(image_path), self.name, f"PaddleOCR 不可用：{exc}"))
        boxes = _boxes_from_paddle_result(raw, self.config.min_confidence, self.name)
        return self._record_health(OcrResult(boxes, str(image_path), self.name, None if boxes else "PaddleOCR 未返回文本"))

    def shutdown(self) -> None:
        self._stop_daemon()

    def _record_health(self, result: OcrResult) -> OcrResult:
        if result.warning and ("失败" in result.warning or "不可用" in result.warning or "超时" in result.warning or "不可解析" in result.warning):
            self._failure_count += 1
            append_diagnostics_log("ocr_worker", f"paddle_failure_count={self._failure_count} warning={result.warning}")
            if self._failure_count >= 2:
                cooldown = int(os.environ.get("WHOCHAT_PADDLEOCR_FAILURE_COOLDOWN_SECONDS", "180"))
                self._unavailable_until = time.monotonic() + cooldown
                append_diagnostics_log("ocr_worker", f"paddle_circuit_open cooldown={cooldown}s")
        else:
            if self._failure_count:
                append_diagnostics_log("ocr_worker", "paddle_recovered")
            self._failure_count = 0
            self._unavailable_until = 0.0
        return result

    def _recognize_in_subprocess(self, image_path: Path) -> OcrResult:
        timeout = int(os.environ.get("WHOCHAT_PADDLEOCR_TIMEOUT_SECONDS", "90"))
        command = [
            sys.executable,
            "-m",
            "whochat.ocr.paddle_worker",
            str(image_path),
            "--language",
            self.config.language,
            "--min-confidence",
            str(self.config.min_confidence),
            "--use-gpu",
            "1" if self.config.use_gpu else "0",
        ]
        env = {**os.environ, "PYTHONUTF8": "1"}
        started = time.monotonic()
        try:
            completed = subprocess.run(
                command,
                cwd=str(Path(__file__).resolve().parents[2]),
                env=env,
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                timeout=timeout,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except subprocess.TimeoutExpired:
            _log_paddle_worker(image_path, timeout * 1000, "timeout", f"timeout={timeout}s")
            return OcrResult([], str(image_path), self.name, f"PaddleOCR 超时：{timeout}s")
        except Exception as exc:
            _log_paddle_worker(image_path, _elapsed_ms(started), "spawn_failed", str(exc))
            return OcrResult([], str(image_path), self.name, f"PaddleOCR 子进程不可用：{exc}")
        elapsed_ms = _elapsed_ms(started)
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip().splitlines()[-4:]
            _log_paddle_worker(image_path, elapsed_ms, f"exit={completed.returncode}", " | ".join(detail))
            return OcrResult([], str(image_path), self.name, "PaddleOCR 子进程失败：" + " | ".join(detail))
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            detail = (completed.stdout or completed.stderr or "").strip().splitlines()[-4:]
            _log_paddle_worker(image_path, elapsed_ms, "bad_json", f"{exc}; {' | '.join(detail)}")
            return OcrResult([], str(image_path), self.name, f"PaddleOCR 输出不可解析：{exc}; {' | '.join(detail)}")
        boxes = [
            OcrTextBox(
                text=item["text"],
                rect=Rect(*item["rect"]),
                confidence=float(item["confidence"]),
                region=OcrRegion.UNKNOWN,
                source=self.name,
            )
            for item in payload.get("boxes", [])
        ]
        warning = payload.get("warning")
        stderr_tail = " | ".join((completed.stderr or "").strip().splitlines()[-3:])
        _log_paddle_worker(image_path, elapsed_ms, "ok", f"boxes={len(boxes)} warning={warning or '-'} stderr={stderr_tail or '-'}")
        return OcrResult(boxes, str(image_path), self.name, warning if warning else None)

    def _recognize_in_daemon(self, image_path: Path) -> OcrResult:
        timeout = int(os.environ.get("WHOCHAT_PADDLEOCR_TIMEOUT_SECONDS", "90"))
        started = time.monotonic()
        with self._daemon_lock:
            try:
                daemon = self._ensure_daemon(timeout)
                assert self._daemon_stdout is not None
                daemon.stdin.write(json.dumps({"image_path": str(image_path)}, ensure_ascii=False) + "\n")
                daemon.stdin.flush()
                line = self._daemon_stdout.get(timeout=timeout)
            except queue.Empty:
                self._stop_daemon()
                _log_paddle_worker(image_path, timeout * 1000, "daemon_timeout", f"timeout={timeout}s")
                return OcrResult([], str(image_path), self.name, f"PaddleOCR daemon 超时：{timeout}s")
            except Exception as exc:
                self._stop_daemon()
                _log_paddle_worker(image_path, _elapsed_ms(started), "daemon_failed", str(exc))
                return OcrResult([], str(image_path), self.name, f"PaddleOCR daemon 不可用：{exc}")
        elapsed_ms = _elapsed_ms(started)
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            self._stop_daemon()
            _log_paddle_worker(image_path, elapsed_ms, "daemon_bad_json", f"{exc}; {line[:500]}")
            return OcrResult([], str(image_path), self.name, f"PaddleOCR daemon 输出不可解析：{exc}")
        if payload.get("status") != "ok":
            error = str(payload.get("error") or payload.get("warning") or payload.get("status") or "unknown")
            _log_paddle_worker(image_path, elapsed_ms, "daemon_error", error)
            return OcrResult([], str(image_path), self.name, f"PaddleOCR daemon 失败：{error}")
        boxes = [
            OcrTextBox(
                text=item["text"],
                rect=Rect(*item["rect"]),
                confidence=float(item["confidence"]),
                region=OcrRegion.UNKNOWN,
                source=self.name,
            )
            for item in payload.get("boxes", [])
        ]
        warning = payload.get("warning")
        _log_paddle_worker(image_path, elapsed_ms, "daemon_ok", f"boxes={len(boxes)} warning={warning or '-'}")
        return OcrResult(boxes, str(image_path), self.name, warning if warning else None)

    def _ensure_daemon(self, timeout: int) -> subprocess.Popen:
        if self._daemon is not None and self._daemon.poll() is None and self._daemon_stdout is not None:
            return self._daemon
        self._stop_daemon()
        command = [
            sys.executable,
            "-m",
            "whochat.ocr.paddle_daemon",
            "--language",
            self.config.language,
            "--min-confidence",
            str(self.config.min_confidence),
            "--use-gpu",
            "1" if self.config.use_gpu else "0",
        ]
        env = {**os.environ, "PYTHONUTF8": "1"}
        self._daemon = subprocess.Popen(
            command,
            cwd=str(Path(__file__).resolve().parents[2]),
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        self._daemon_stdout = queue.Queue()
        assert self._daemon.stdout is not None
        threading.Thread(target=_read_daemon_stdout, args=(self._daemon.stdout, self._daemon_stdout), daemon=True).start()
        if self._daemon.stderr is not None:
            threading.Thread(target=_drain_daemon_stderr, args=(self._daemon.stderr,), daemon=True).start()
        try:
            line = self._daemon_stdout.get(timeout=timeout)
        except queue.Empty as exc:
            self._stop_daemon()
            raise TimeoutError(f"daemon startup timed out after {timeout}s") from exc
        payload = json.loads(line)
        if payload.get("status") != "ready":
            self._stop_daemon()
            raise RuntimeError(f"daemon startup failed: {payload}")
        return self._daemon

    def _stop_daemon(self) -> None:
        daemon = self._daemon
        self._daemon = None
        self._daemon_stdout = None
        if daemon is None:
            return
        try:
            if daemon.poll() is None:
                daemon.kill()
        except Exception:
            return

    def _get_engine(self):
        if self._engine is None:
            _configure_paddle_cache()
            from paddleocr import PaddleOCR

            attempts = [
                {
                    "device": "gpu" if self.config.use_gpu else "cpu",
                    "enable_mkldnn": False,
                    "text_detection_model_name": "PP-OCRv5_mobile_det",
                    "text_recognition_model_name": "PP-OCRv5_mobile_rec",
                    "use_doc_orientation_classify": False,
                    "use_doc_unwarping": False,
                    "use_textline_orientation": False,
                    "cpu_threads": 1,
                },
                {"lang": self.config.language, "use_angle_cls": False},
                {"lang": self.config.language},
            ]
            last_error: Exception | None = None
            for kwargs in attempts:
                try:
                    self._engine = PaddleOCR(**kwargs)
                    break
                except Exception as exc:
                    last_error = exc
            if self._engine is None and last_error is not None:
                raise last_error
        return self._engine


def create_ocr_engine(config: OcrConfig | None = None) -> OcrEngine:
    config = config or OcrConfig()
    if config.provider == "RapidOCR":
        return RapidOcrEngine(config)
    if config.provider == "PaddleOCR":
        return PaddleOcrEngine(config)
    return PreviewOcrEngine()


# RapidOCR's detector normalises the *shorter* side of its input to
# ``Det.limit_side_len``.  When the crop is already narrower than that value the
# detector has to upscale, and the resampling smears WeChat's small grey glyphs
# (nicknames, timestamps, media durations) until they disappear.  Feeding a
# sharp, larger source instead keeps them detectable.
MIN_MULTILINE_HEIGHT = 240
# A WeChat title strip is a single thin line.  RapidOCR recognised the real
# glyphs natively, but also returned window-control fragments; a restrained 2x
# Lanczos source made the complete nickname stable and removed those fragments
# without the false edges introduced by sharpening / thresholding.
TITLE_BAND_MAX_HEIGHT = 128
TITLE_BAND_MIN_WIDTH = 120
TITLE_BAND_UPSCALE = 2.0
# Only skip the pre-scale when the shortest side already meets the detector
# limit, i.e. when the detector will leave the pixels untouched.
DETECTOR_SAFE_RATIO = 1.0
MIN_UPSCALE = 1.5
MAX_UPSCALE = 2.5


def _prepare_ocr_image(image_path: Path, config: OcrConfig) -> tuple[Any, float, float]:
    """Load ``image_path``, upscaling it when RapidOCR would otherwise blur it.

    Returns the image plus the horizontal / vertical factors that must be
    divided back out of the recognised boxes.
    """
    with Image.open(image_path) as handle:
        image = handle.convert("RGB")
    factor = _upscale_factor(image.size, config)
    if factor <= 1.0:
        return image, 1.0, 1.0
    size = (max(1, round(image.width * factor)), max(1, round(image.height * factor)))
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    scaled = image.resize(size, resampling)
    return scaled, image.width / size[0], image.height / size[1]


def _upscale_factor(size: tuple[int, int], config: OcrConfig) -> float:
    width, height = size
    shortest = min(width, height)
    if shortest <= 0:
        return 1.0
    if height <= TITLE_BAND_MAX_HEIGHT and width >= TITLE_BAND_MIN_WIDTH:
        return TITLE_BAND_UPSCALE
    if height < MIN_MULTILINE_HEIGHT:
        return 1.0
    limit = max(1, int(config.detector_min_side))
    if shortest >= limit * DETECTOR_SAFE_RATIO:
        return 1.0
    target = max(limit, int(config.upscale_target_side))
    return min(MAX_UPSCALE, max(MIN_UPSCALE, target / shortest))


def _scale_ocr_boxes(boxes: list[OcrTextBox], scale_x: float, scale_y: float) -> list[OcrTextBox]:
    scaled: list[OcrTextBox] = []
    for box in boxes:
        rect = box.rect
        scaled.append(
            OcrTextBox(
                text=box.text,
                rect=Rect(
                    round(rect.left * scale_x),
                    round(rect.top * scale_y),
                    round(rect.right * scale_x),
                    round(rect.bottom * scale_y),
                ),
                confidence=box.confidence,
                region=box.region,
                source=box.source,
            )
        )
    return scaled


def _configure_paddle_cache() -> None:
    configure_native_runtime_limits()
    cache_home = app_data_dir() / "ocr_cache" / "home"
    paddle_home = cache_home / ".cache" / "paddle"
    paddleocr_home = cache_home / ".cache" / "paddleocr"
    paddle_home.mkdir(parents=True, exist_ok=True)
    paddleocr_home.mkdir(parents=True, exist_ok=True)
    # PaddlePaddle still uses expanduser("~/.cache/paddle") in some import paths.
    os.environ["USERPROFILE"] = str(cache_home)
    os.environ["PADDLE_HOME"] = str(paddle_home)
    os.environ["PADDLEOCR_HOME"] = str(paddleocr_home)
    os.environ["XDG_CACHE_HOME"] = str(cache_home / ".cache")
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")


def _elapsed_ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)


def _log_paddle_worker(image_path: Path, elapsed_ms: int, status: str, detail: str) -> None:
    clipped = " ".join(detail.split())[:1000]
    append_diagnostics_log("ocr_worker", f"image={image_path.name} elapsed_ms={elapsed_ms} status={status} {clipped}")


def _read_daemon_stdout(stream, output: queue.Queue[str]) -> None:
    for line in stream:
        cleaned = line.strip()
        if cleaned:
            output.put(cleaned)


def _drain_daemon_stderr(stream) -> None:
    for line in stream:
        cleaned = line.strip()
        if cleaned:
            append_diagnostics_log("ocr_worker", f"paddle_daemon_stderr {cleaned[:1000]}")


def _box_inside(rect: Rect, left: float, top: float, right: float, bottom: float) -> Rect:
    return Rect(
        rect.left + round(rect.width * left),
        rect.top + round(rect.height * top),
        rect.left + round(rect.width * right),
        rect.top + round(rect.height * bottom),
    )


def _boxes_from_rapid_result(raw: Any, min_confidence: float, source: str) -> list[OcrTextBox]:
    items = getattr(raw, "boxes", None)
    texts = getattr(raw, "txts", None) or getattr(raw, "texts", None)
    scores = getattr(raw, "scores", None)
    if items is not None and texts is not None and scores is not None:
        return _boxes_from_parts(items, texts, scores, min_confidence, source)
    if isinstance(raw, tuple) and len(raw) >= 3:
        return _boxes_from_parts(raw[0], raw[1], raw[2], min_confidence, source)
    if isinstance(raw, list):
        return _boxes_from_records(raw, min_confidence, source)
    return []


def _boxes_from_paddle_result(raw: Any, min_confidence: float, source: str) -> list[OcrTextBox]:
    if isinstance(raw, list) and len(raw) == 1 and isinstance(raw[0], list):
        raw = raw[0]
    if isinstance(raw, list) and raw and all(isinstance(item, dict) for item in raw):
        result: list[OcrTextBox] = []
        for page in raw:
            texts = page.get("rec_texts")
            scores = page.get("rec_scores")
            boxes = page.get("rec_polys") or page.get("dt_polys") or page.get("rec_boxes")
            if texts is not None and scores is not None and boxes is not None:
                result.extend(_boxes_from_parts(boxes, texts, scores, min_confidence, source))
            else:
                parsed = _parse_ocr_record(page)
                if parsed is not None:
                    points, text, confidence = parsed
                    if confidence >= min_confidence and text.strip():
                        result.append(OcrTextBox(text.strip(), _rect_from_points(points), confidence, OcrRegion.UNKNOWN, source))
        return result
    if isinstance(raw, list):
        return _boxes_from_records(raw, min_confidence, source)
    return []


def _boxes_from_parts(boxes: Any, texts: Any, scores: Any, min_confidence: float, source: str) -> list[OcrTextBox]:
    result: list[OcrTextBox] = []
    for box, text, score in zip(_items(boxes), _items(texts), _items(scores)):
        confidence = _float(score)
        if confidence < min_confidence:
            continue
        value = str(text).strip()
        if not value:
            continue
        result.append(OcrTextBox(value, _rect_from_points(box), confidence, OcrRegion.UNKNOWN, source))
    return result


def _boxes_from_records(records: Any, min_confidence: float, source: str) -> list[OcrTextBox]:
    result: list[OcrTextBox] = []
    for record in _items(records):
        parsed = _parse_ocr_record(record)
        if parsed is None:
            continue
        points, text, confidence = parsed
        if confidence < min_confidence or not text.strip():
            continue
        result.append(OcrTextBox(text.strip(), _rect_from_points(points), confidence, OcrRegion.UNKNOWN, source))
    return result


def _parse_ocr_record(record: Any) -> tuple[Any, str, float] | None:
    if isinstance(record, dict):
        points = record.get("box") or record.get("points") or record.get("dt_polys")
        text = record.get("text") or record.get("rec_text") or record.get("label")
        score = record.get("score") or record.get("confidence") or record.get("rec_score")
        if points is not None and text is not None and score is not None:
            return points, str(text), _float(score)
    if isinstance(record, (list, tuple)) and len(record) >= 2:
        points = record[0]
        payload = record[1]
        if isinstance(payload, (list, tuple)) and len(payload) >= 2:
            return points, str(payload[0]), _float(payload[1])
        if len(record) >= 3:
            return points, str(record[1]), _float(record[2])
    return None


def _run_paddle_ocr(engine: Any, image_path: Path) -> Any:
    if hasattr(engine, "ocr"):
        return engine.ocr(str(image_path))
    if hasattr(engine, "predict"):
        return engine.predict(str(image_path))
    raise RuntimeError("PaddleOCR engine has neither ocr() nor predict()")


def _rect_from_points(points: Any) -> Rect:
    normalized = []
    for point in [] if points is None else points:
        if hasattr(point, "__len__") and len(point) >= 2:
            normalized.append((int(round(float(point[0]))), int(round(float(point[1])))))
    if not normalized:
        return Rect(0, 0, 0, 0)
    xs = [point[0] for point in normalized]
    ys = [point[1] for point in normalized]
    return Rect(min(xs), min(ys), max(xs), max(ys))


def _items(value: Any) -> Any:
    if value is None:
        return []
    return value


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
