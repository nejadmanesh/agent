"""High level orchestration for the Persian data pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Sequence

import pandas as pd

from .cleaning import PersianTextCleaner
from .labelers import LabelledRecord
from .storage import DataStorage


@dataclass
class DataPipeline:
    """Glue together the cleaner, label sources, and persistence layer."""

    cleaner: PersianTextCleaner
    storage: DataStorage
    label_sources: Sequence[object] = field(default_factory=list)

    def run(self) -> pd.DataFrame:
        """Fetch, clean, and persist labelled text samples."""

        records = self._collect_records()
        dataframe = self._build_dataframe(records)
        if not dataframe.empty:
            self.storage.store(dataframe)
        return dataframe

    def _collect_records(self) -> List[LabelledRecord]:
        records: List[LabelledRecord] = []
        for source in self.label_sources:
            fetched = getattr(source, "fetch_records", None)
            if fetched is None:
                raise AttributeError(f"Label source {source!r} does not provide fetch_records()")
            for record in fetched():
                cleaned_text = self.cleaner.clean(record.text)
                records.append(
                    LabelledRecord(
                        record_id=record.record_id,
                        text=record.text,
                        labels=list(record.labels),
                        source=record.source,
                        raw_payload=dict(record.raw_payload),
                        clean_text=cleaned_text,
                    )
                )
        return records

    def _build_dataframe(self, records: Iterable[LabelledRecord]) -> pd.DataFrame:
        rows = []
        for record in records:
            clean_text = record.clean_text or self.cleaner.clean(record.text)
            rows.append(
                {
                    "record_id": record.record_id,
                    "source": record.source,
                    "text": record.text,
                    "clean_text": clean_text,
                    "labels": list(record.labels),
                }
            )
        return pd.DataFrame(rows, columns=["record_id", "source", "text", "clean_text", "labels"])
