"""Dependency factories used by the FastAPI app and Celery worker."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Dict

from inference import InferenceEngine, MemoryManager, ModelConfig, ModelLoader

from .database import Database


def _default_metadata() -> Dict[str, str]:
    metadata: Dict[str, str] = {
        "behavior": os.environ.get("DEFAULT_MODEL_BEHAVIOR", "echo"),
    }
    prefix = os.environ.get("DEFAULT_MODEL_PREFIX")
    suffix = os.environ.get("DEFAULT_MODEL_SUFFIX")
    if prefix:
        metadata["prefix"] = prefix
    if suffix:
        metadata["suffix"] = suffix
    return metadata


@lru_cache
def get_memory_manager() -> MemoryManager:
    limit = int(os.environ.get("INFERENCE_MEMORY_LIMIT", str(512 * 1024 * 1024)))
    return MemoryManager(limit)


@lru_cache
def get_model_loader() -> ModelLoader:
    return ModelLoader(get_memory_manager())


@lru_cache
def get_engine() -> InferenceEngine:
    default_model_path = os.environ.get("DEFAULT_MODEL_PATH")
    resolved_path: Path | None = None
    if default_model_path:
        resolved_path = Path(default_model_path).expanduser().resolve()
    default_model = ModelConfig(
        name=os.environ.get("DEFAULT_MODEL_NAME", "default"),
        path=resolved_path,
        metadata=_default_metadata(),
    )
    return InferenceEngine(get_model_loader(), default_model)


@lru_cache
def get_database() -> Database:
    db_path = os.environ.get("INFERENCE_DB_PATH")
    if db_path:
        path = Path(db_path).expanduser()
    else:
        data_dir = Path(os.environ.get("APP_DATA_DIR", ".")).expanduser()
        path = data_dir / "inference.db"
    return Database(path)
