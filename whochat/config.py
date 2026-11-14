from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from whochat.core.paths import config_dir
from whochat.security.secret_store import AI_API_KEY, SecretStore


@dataclass
class AIProviderConfig:
    provider: str = "OpenAI Compatible"
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4.1-mini"
    api_key: str = ""
    temperature: float = 0.7
    context_tokens: int = 16000
    timeout_seconds: int = 60
    request_cooldown_seconds: int = 8
    dedupe_context_minutes: int = 30
    max_daily_cloud_requests: int = 100
    failure_backoff_threshold: int = 3
    failure_backoff_minutes: int = 5


@dataclass
class PrivacyConfig:
    enable_long_term_memory: bool = True
    save_debug_screenshots: bool = False
    trim_context_for_cloud: bool = True
    require_cloud_prompt_review: bool = False
    manual_protection_blocks_replies: bool = True
    diagnostic_log_retention_days: int = 14
    debug_sample_retention_days: int = 14
    capture_retention_days: int = 7
    calibration_retention_days: int = 30
    reply_feedback_retention_days: int = 180


@dataclass
class OcrConfig:
    # RapidOCR is the default on purpose: PaddleOCR's default configuration
    # downloads / initialises heavy models on first call and can stall for tens
    # of seconds, which read as a freeze in the UI.  RapidOCR ships its models
    # in-package and returns in well under a second for a chat crop.
    provider: str = "RapidOCR"
    language: str = "ch"
    min_confidence: float = 0.5
    use_gpu: bool = False
    # RapidOCR's detector normalises the *shorter* side of its input to
    # ``detector_min_side``.  A crop narrower than this is upscaled by the
    # detector itself, and that resampling smears WeChat's small grey glyphs
    # (nicknames, timestamps, media durations) until they vanish.  The OCR
    # engine therefore pre-scales such crops with a sharp filter up to
    # ``upscale_target_side`` so the detector never has to resample them.
    detector_min_side: int = 736
    upscale_target_side: int = 1200


@dataclass
class CaptureConfig:
    auto_capture_enabled: bool = True
    foreground_only: bool = True
    scroll_debounce_ms: int = 900
    ocr_min_interval_ms: int = 5000
    pause_ai_on_unknown_page: bool = True
    block_memory_for_unconfirmed_contact: bool = True


@dataclass
class ProactiveConfig:
    """Local-only controls for one-contact activation assessment."""

    silence_days: int = 7
    activation_prewarm_enabled: bool = True
    activation_prewarm_ttl_days: int = 7
    activation_prewarm_daily_provider_cap: int = 100
    activation_prewarm_idle_seconds: int = 120
    activation_prewarm_interval_seconds: int = 300
    activation_prewarm_initial_delay_seconds: int = 600
    activation_l1_capacity: int = 512


@dataclass
class HistoricalSchedulerConfig:
    """Conservative defaults for low-priority Historical Backfill.

    The scheduler is deliberately disabled and manually paused after upgrade.
    Enabling it is a separate, explicit production action.  The time window is
    local-time configuration rather than a hard-coded night-time assumption.
    """

    enabled: bool = False
    start_time: str = "00:00"
    end_time: str = "06:00"
    daily_historical_call_cap: int = 100
    max_customers_per_batch: int = 20
    target_chunks_min: int = 70
    target_chunks_max: int = 80
    provider_call_safety_margin: int = 3
    max_runtime_seconds: int = 7200
    max_batches_per_day: int = 1
    observation_batch_count: int = 7


@dataclass
class FloatingConfig:
    placement_preference: str = "auto"
    opacity_percent: int = 96
    suggestion_count: int = 2


@dataclass
class TargetWindowConfig:
    app_id: str
    label: str
    enabled: bool
    bubble_profile: str = "auto"
    process_names: list[str] = field(default_factory=list)
    title_keywords: list[str] = field(default_factory=list)
    exclude_title_keywords: list[str] = field(default_factory=list)


@dataclass
class AppConfig:
    ai: AIProviderConfig = field(default_factory=AIProviderConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    ocr: OcrConfig = field(default_factory=OcrConfig)
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    proactive: ProactiveConfig = field(default_factory=ProactiveConfig)
    historical_scheduler: HistoricalSchedulerConfig = field(default_factory=HistoricalSchedulerConfig)
    floating: FloatingConfig = field(default_factory=FloatingConfig)
    targets: list[TargetWindowConfig] = field(default_factory=list)


class ConfigStore:
    """Reads/writes ``config.json`` plus the encrypted secret store beside it.

    The AI API key is **never** written to the JSON file.  It lives in
    :class:`~whochat.security.secret_store.SecretStore` (Windows DPAPI, user
    scope).  ``save`` returns the backend name so the UI can tell the user where
    the key went, or warn when no secure backend is available.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or default_config_path()
        self.secrets = SecretStore(self.path.parent)

    @property
    def secret_backend(self) -> str:
        return self.secrets.backend

    def load(self) -> AppConfig:
        if not self.path.exists():
            config = AppConfig()
            config.targets = default_target_windows()
            config.ai.api_key = self.secrets.get(AI_API_KEY)
            return config
        data = json.loads(self.path.read_text(encoding="utf-8"))
        capture_data = _migrate_capture_config(data.get("capture", {}))
        ai_data = dict(data.get("ai", {}))
        # A key written directly into the file (hand edit, or a config produced
        # by an older build) is an explicit user choice and wins over the store.
        file_key = str(ai_data.get("api_key") or "").strip()
        ai_data["api_key"] = file_key or self.secrets.get(AI_API_KEY)
        return AppConfig(
            ai=AIProviderConfig(**ai_data),
            privacy=PrivacyConfig(**data.get("privacy", {})),
            ocr=OcrConfig(**data.get("ocr", {})),
            capture=CaptureConfig(**capture_data),
            proactive=ProactiveConfig(**_migrate_proactive_config(data.get("proactive", {}))),
            historical_scheduler=HistoricalSchedulerConfig(**data.get("historical_scheduler", {})),
            floating=FloatingConfig(**{**data.get("floating", {}), "suggestion_count": 2}),
            targets=_load_targets(data.get("targets")),
        )

    def save(self, config: AppConfig) -> str:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        config.floating.suggestion_count = 2
        data: dict[str, Any] = asdict(config)
        api_key = str(data.get("ai", {}).get("api_key") or "").strip()
        # Keep the key out of the plaintext file no matter what: the encrypted
        # store (or nothing at all) holds it.
        data["ai"]["api_key"] = ""
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        if api_key:
            self.secrets.set(AI_API_KEY, api_key)
        else:
            self.secrets.delete(AI_API_KEY)
        return self.secrets.backend


def default_config_path() -> Path:
    return config_dir() / "config.json"


def _migrate_capture_config(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    migrated = dict(value)
    if "foreground_only" not in migrated:
        # Older builds stored auto_capture_enabled=False as their default.
        # Absence of foreground_only identifies that legacy shape; migrate it
        # to the current hands-off, foreground-safe default.
        migrated["auto_capture_enabled"] = True
        migrated["foreground_only"] = True
    if "ocr_min_interval_ms" not in migrated:
        migrated["ocr_min_interval_ms"] = CaptureConfig.ocr_min_interval_ms
    elif int(migrated.get("ocr_min_interval_ms") or 0) == 30000 and "foreground_only" not in value:
        migrated["ocr_min_interval_ms"] = CaptureConfig.ocr_min_interval_ms
    return migrated


def _migrate_proactive_config(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    migrated = dict(value)
    try:
        migrated["silence_days"] = min(365, max(1, int(migrated.get("silence_days", 7))))
    except (TypeError, ValueError):
        migrated["silence_days"] = 7
    migrated["activation_prewarm_enabled"] = bool(
        migrated.get("activation_prewarm_enabled", True)
    )
    integer_defaults = {
        "activation_prewarm_ttl_days": (7, 1, 90),
        "activation_prewarm_daily_provider_cap": (100, 0, 1000),
        "activation_prewarm_idle_seconds": (120, 30, 3600),
        "activation_prewarm_interval_seconds": (300, 30, 86400),
        "activation_prewarm_initial_delay_seconds": (600, 30, 86400),
        "activation_l1_capacity": (512, 8, 5000),
    }
    for key, (default, lower, upper) in integer_defaults.items():
        try:
            migrated[key] = min(upper, max(lower, int(migrated.get(key, default))))
        except (TypeError, ValueError):
            migrated[key] = default
    return migrated


def default_target_windows() -> list[TargetWindowConfig]:
    return [
        TargetWindowConfig(
            app_id="wechat",
            label="微信",
            enabled=True,
            bubble_profile="wechat_green",
            # Only the WeChat *main* window hosts the chat list, and it is owned
            # by Weixin.exe (4.x) or WeChat.exe (3.x).  WeChatAppEx.exe is the
            # mini-program / embedded-browser runtime; capturing it OCR'd
            # whatever mini-program happened to be open (e.g. 腾讯WorkBuddy)
            # instead of a chat, so it is deliberately NOT a wechat process.
            process_names=["Weixin.exe", "WeChat.exe"],
            title_keywords=["微信", "WeChat"],
            exclude_title_keywords=[
                "图片和视频",
                "设置",
                "聊天记录",
                "转发",
                "发送给",
                "选择联系人",
                "WorkBuddy",
            ],
        ),
        TargetWindowConfig(
            app_id="telegram",
            label="Telegram",
            enabled=False,
            bubble_profile="geometry",
            process_names=["Telegram.exe"],
            title_keywords=["Telegram"],
            exclude_title_keywords=[],
        ),
        TargetWindowConfig(
            app_id="generic_chat",
            label="通用聊天",
            enabled=False,
            bubble_profile="geometry",
            process_names=[],
            title_keywords=[],
            exclude_title_keywords=[],
        ),
    ]


def _load_targets(value: Any) -> list[TargetWindowConfig]:
    if not isinstance(value, list):
        return default_target_windows()
    result: list[TargetWindowConfig] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        try:
            app_id = str(item.get("app_id", "")).strip()
            label = str(item.get("label", "")).strip() or app_id
            if not app_id:
                continue
            result.append(
                TargetWindowConfig(
                    app_id=app_id,
                    label=label,
                    enabled=bool(item.get("enabled", False)),
                    bubble_profile=str(item.get("bubble_profile", "auto")).strip() or "auto",
                    process_names=_string_list(item.get("process_names")),
                    title_keywords=_string_list(item.get("title_keywords")),
                    exclude_title_keywords=_string_list(item.get("exclude_title_keywords")),
                )
            )
        except TypeError:
            continue
    defaults = {target.app_id: target for target in default_target_windows()}
    merged = {target.app_id: target for target in result}
    for app_id, target in defaults.items():
        if app_id not in merged:
            merged[app_id] = target
        else:
            _upgrade_target_rules(app_id, merged[app_id], target)
    return list(merged.values())


# Process names that an older build treated as the WeChat main window but that
# never host the chat list.  Dropping them keeps existing installs from OCR'ing
# a mini-program window after the upgrade.
LEGACY_PROCESS_NAMES: dict[str, set[str]] = {
    "wechat": {"WeChatAppEx.exe"},
}


def _upgrade_target_rules(app_id: str, target: TargetWindowConfig, default: TargetWindowConfig) -> None:
    """Fold newer default rules into a target persisted by an older build.

    Runs on load so an existing ``config.json`` also stops matching the
    WeChatAppEx.exe mini-program runtime and picks up newly added title
    exclusions, without discarding the user's own edits.
    """
    legacy = LEGACY_PROCESS_NAMES.get(app_id, set())
    if legacy:
        normalized_legacy = {name.lower() for name in legacy}
        target.process_names = [name for name in target.process_names if name.strip().lower() not in normalized_legacy]
    existing = {keyword.strip().lower() for keyword in target.exclude_title_keywords}
    for keyword in default.exclude_title_keywords:
        key = keyword.strip().lower()
        if key and key not in existing:
            target.exclude_title_keywords.append(keyword)
            existing.add(key)


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = str(item).strip()
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result
