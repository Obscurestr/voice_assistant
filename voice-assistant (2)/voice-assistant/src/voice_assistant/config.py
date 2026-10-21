"""Runtime settings. Every value can be overridden with an environment variable."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_path(name: str, default: str) -> Path:
    return Path(os.getenv(name, default)).expanduser()


def _env_optional_int(name: str) -> int | None:
    value = os.getenv(name)
    return int(value) if value not in (None, "") else None


@dataclass
class Settings:
    data_path: Path = field(default_factory=lambda: _env_path("VA_DATA", "data/alexa_data.csv"))
    heldout_path: Path = field(
        default_factory=lambda: _env_path("VA_HELDOUT", "data/heldout_utterances.csv")
    )
    model_path: Path = field(
        default_factory=lambda: _env_path("VA_MODEL", "models/intent_model.joblib")
    )
    memory_path: Path = field(
        default_factory=lambda: _env_path("VA_MEMORY", "~/.voice_assistant/memory.json")
    )
    documents_dir: Path = field(
        default_factory=lambda: _env_path("VA_DOCUMENTS", "~/Documents/Voice Assistant")
    )
    reports_dir: Path = field(default_factory=lambda: _env_path("VA_REPORTS", "reports"))

    # Predictions below this probability are answered with a clarification request.
    confidence_threshold: float = float(os.getenv("VA_CONFIDENCE", "0.35"))

    # Voice I/O
    whisper_model: str = os.getenv("VA_WHISPER_MODEL", "base")
    sample_rate: int = 16_000
    record_seconds: float = float(os.getenv("VA_RECORD_SECONDS", "5"))
    input_device: int | None = field(default_factory=lambda: _env_optional_int("VA_INPUT_DEVICE"))
    speech_rate: int = 165
