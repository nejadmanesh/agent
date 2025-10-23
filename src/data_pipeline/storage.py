"""Persistence helpers for the Persian data pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from sqlalchemy import Engine, create_engine


@dataclass
class DataStorage:
    """Store pipeline artefacts in PostgreSQL and parquet files."""

    engine: Engine
    parquet_path: Path

    @classmethod
    def from_connection_string(cls, dsn: str, parquet_path: str | Path) -> "DataStorage":
        return cls(create_engine(dsn, future=True), Path(parquet_path))

    def store(
        self,
        dataframe: pd.DataFrame,
        metadata_table: str = "labeled_sample_metadata",
    ) -> None:
        if dataframe.empty:
            return
        parquet_ready = dataframe.copy()
        parquet_ready["labels"] = parquet_ready["labels"].apply(_ensure_string_list)
        self._write_parquet(parquet_ready)
        metadata_frame = pd.DataFrame(
            {
                "record_id": parquet_ready["record_id"],
                "source": parquet_ready["source"],
                "label_count": parquet_ready["labels"].apply(len),
                "labels": parquet_ready["labels"].apply(json.dumps),
            }
        )
        self._write_metadata(metadata_frame, metadata_table)

    def _write_metadata(self, dataframe: pd.DataFrame, table_name: str) -> None:
        with self.engine.begin() as connection:
            dataframe.to_sql(table_name, connection, if_exists="append", index=False)

    def _write_parquet(self, dataframe: pd.DataFrame) -> None:
        self.parquet_path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_parquet(self.parquet_path, index=False)


def _ensure_string_list(raw: Iterable[object]) -> list[str]:
    if isinstance(raw, str):
        return [raw]
    if not isinstance(raw, Iterable):
        return []
    normalised: list[str] = []
    for item in raw:
        normalised.append(str(item))
    return normalised
