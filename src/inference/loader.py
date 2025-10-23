"""Model loading utilities built on top of the memory manager."""

from __future__ import annotations

import json
from dataclasses import dataclass
from threading import Lock
from typing import Any, Dict, cast

from .config import ModelConfig
from .memory import MemoryManager, MemoryReservation


@dataclass(slots=True)
class LoadedModel:
    """Representation of a loaded model artifact."""

    config: ModelConfig
    artifact: Dict[str, Any]
    reservation: MemoryReservation

    def predict(self, text: str) -> str:
        behavior = self.artifact.get("behavior", "echo")
        if behavior == "reverse":
            return text[::-1]
        if behavior == "uppercase":
            return text.upper()
        if behavior == "lowercase":
            return text.lower()
        prefix = self.artifact.get("prefix")
        suffix = self.artifact.get("suffix")
        if prefix or suffix:
            return f"{prefix or ''}{text}{suffix or ''}"
        return text


class ModelLoader:
    """Load and cache models while obeying the configured memory constraints."""

    def __init__(self, memory_manager: MemoryManager) -> None:
        self._memory_manager = memory_manager
        self._loaded_models: dict[str, LoadedModel] = {}
        self._lock = Lock()

    def load(self, config: ModelConfig) -> LoadedModel:
        with self._lock:
            cached = self._loaded_models.get(config.name)
            if cached is not None:
                return cached
            artifact = self._load_artifact(config)
            reservation = self._memory_manager.reserve(
                config.memory_bytes, owner=config.name
            )
            loaded_model = LoadedModel(
                config=config,
                artifact=artifact,
                reservation=reservation,
            )
            self._loaded_models[config.name] = loaded_model
            return loaded_model

    def unload(self, name: str) -> None:
        with self._lock:
            loaded_model = self._loaded_models.pop(name, None)
        if loaded_model is not None:
            self._memory_manager.release(loaded_model.reservation)

    def clear(self) -> None:
        with self._lock:
            names = list(self._loaded_models.keys())
        for name in names:
            self.unload(name)

    def _load_artifact(self, config: ModelConfig) -> Dict[str, Any]:
        path = config.resolve_path()
        if path is None or not path.exists():
            return dict(config.metadata)
        if path.suffix.lower() in {".json", ""}:
            try:
                return cast(Dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
            except json.JSONDecodeError as exc:  # pragma: no cover - defensive
                raise ValueError(f"Invalid JSON model artifact at {path}") from exc
        if path.suffix.lower() in {".txt", ".md"}:
            return {
                "behavior": "echo",
                "prefix": path.read_text(encoding="utf-8").strip(),
            }
        return {"behavior": "echo"}
