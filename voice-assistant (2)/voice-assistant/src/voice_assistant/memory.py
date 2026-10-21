"""Small persistent key-value store for user facts, lists and device state."""

from __future__ import annotations

import copy
import json
import os
import tempfile
from pathlib import Path
from typing import Any

DEFAULT_MEMORY: dict[str, Any] = {
    "name": None,
    "preferences": {},
    "shopping_list": [],
    "todos": [],
    "reminders": [],
    "alarms": [],
    "devices": {},
}


class MemoryStore:
    """JSON-backed store. Writes are atomic so a crash never leaves a half-written file."""

    def __init__(self, path: str | Path | None):
        self.path = Path(path) if path else None
        self.data = copy.deepcopy(DEFAULT_MEMORY)
        if self.path and self.path.exists():
            try:
                stored = json.loads(self.path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Memory file {self.path} is not valid JSON") from exc
            self.data.update(stored)

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.data[key] = value
        self.save()

    def append(self, key: str, value: Any) -> None:
        self.data.setdefault(key, []).append(value)
        self.save()

    def save(self) -> None:
        if self.path is None:  # in-memory mode, used by tests and the demo
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(self.data, fh, indent=2, ensure_ascii=False)
        os.replace(tmp, self.path)
