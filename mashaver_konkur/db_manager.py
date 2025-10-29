"""Database management utilities for Mashaver Konkur application."""
from __future__ import annotations

import os
import sqlite3
from contextlib import closing
from typing import Any, Dict, Iterable, List, Optional


class DatabaseManager:
    """Simple SQLite based storage for tasks and chat history."""

    def __init__(self, db_path: str = "data/mashaver.db") -> None:
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._connection = sqlite3.connect(self.db_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        with closing(self._connection.cursor()) as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'شروع نشده',
                    due_date TEXT,
                    notes TEXT
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            self._connection.commit()

    # ------------------------------------------------------------------
    # Task CRUD operations
    # ------------------------------------------------------------------
    def add_task(
        self,
        title: str,
        status: str = "شروع نشده",
        due_date: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> int:
        with closing(self._connection.cursor()) as cursor:
            cursor.execute(
                "INSERT INTO tasks (title, status, due_date, notes) VALUES (?, ?, ?, ?)",
                (title.strip(), status, due_date, notes),
            )
            self._connection.commit()
            return cursor.lastrowid

    def update_task_status(self, task_id: int, status: str) -> None:
        with closing(self._connection.cursor()) as cursor:
            cursor.execute(
                "UPDATE tasks SET status = ? WHERE id = ?",
                (status, task_id),
            )
            self._connection.commit()

    def update_task(self, task_id: int, **fields: Any) -> None:
        if not fields:
            return
        columns = ", ".join(f"{key} = ?" for key in fields)
        values: List[Any] = list(fields.values())
        values.append(task_id)
        with closing(self._connection.cursor()) as cursor:
            cursor.execute(f"UPDATE tasks SET {columns} WHERE id = ?", values)
            self._connection.commit()

    def delete_task(self, task_id: int) -> None:
        with closing(self._connection.cursor()) as cursor:
            cursor.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
            self._connection.commit()

    def get_tasks(self) -> List[Dict[str, Any]]:
        with closing(self._connection.cursor()) as cursor:
            cursor.execute("SELECT id, title, status, due_date, notes FROM tasks ORDER BY id DESC")
            rows = cursor.fetchall()
        return [dict(row) for row in rows]

    # ------------------------------------------------------------------
    # Conversation storage
    # ------------------------------------------------------------------
    def add_conversation(self, question: str, answer: str) -> int:
        with closing(self._connection.cursor()) as cursor:
            cursor.execute(
                "INSERT INTO conversations (question, answer) VALUES (?, ?)",
                (question.strip(), answer.strip()),
            )
            self._connection.commit()
            return cursor.lastrowid

    def get_conversations(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        query = "SELECT id, question, answer, created_at FROM conversations ORDER BY created_at DESC"
        if limit is not None:
            query += " LIMIT ?"
            params: Iterable[Any] = (limit,)
        else:
            params = ()
        with closing(self._connection.cursor()) as cursor:
            cursor.execute(query, params)
            rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def close(self) -> None:
        self._connection.close()


__all__ = ["DatabaseManager"]
