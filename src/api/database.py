"""Lightweight SQLite wrapper used by the API and Celery worker."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


@dataclass(slots=True)
class InferenceRecord:
    task_id: str
    model_name: str
    input_text: str
    output_text: Optional[str]
    status: str
    error: Optional[str]
    created_at: str


class Database:
    """Simple SQLite-based persistence layer for inference results."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @property
    def path(self) -> Path:
        return self._path

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._path)

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS inference_results (
                    task_id TEXT PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    input_text TEXT NOT NULL,
                    output_text TEXT,
                    status TEXT NOT NULL,
                    error TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()

    def upsert_result(
        self,
        task_id: str,
        *,
        model_name: str,
        input_text: str,
        status: str,
        output_text: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO inference_results (task_id, model_name, input_text, output_text, status, error)
                VALUES (:task_id, :model_name, :input_text, :output_text, :status, :error)
                ON CONFLICT(task_id) DO UPDATE SET
                    output_text = excluded.output_text,
                    status = excluded.status,
                    error = excluded.error
                """,
                {
                    "task_id": task_id,
                    "model_name": model_name,
                    "input_text": input_text,
                    "output_text": output_text,
                    "status": status,
                    "error": error,
                },
            )
            conn.commit()

    def get_result(self, task_id: str) -> Optional[InferenceRecord]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT task_id, model_name, input_text, output_text, status, error, created_at
                FROM inference_results
                WHERE task_id = :task_id
                """,
                {"task_id": task_id},
            ).fetchone()
        if row is None:
            return None
        return InferenceRecord(*row)

    def list_results(self, *, limit: int = 100) -> Iterable[InferenceRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT task_id, model_name, input_text, output_text, status, error, created_at
                FROM inference_results
                ORDER BY created_at DESC
                LIMIT :limit
                """,
                {"limit": limit},
            ).fetchall()
        for row in rows:
            yield InferenceRecord(*row)
