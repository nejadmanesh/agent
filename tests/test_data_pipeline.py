from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine

from data_pipeline import (
    CleaningConfig,
    DataPipeline,
    DataStorage,
    LabelledRecord,
    PersianTextCleaner,
)


class _FakeLabelSource:
    def __init__(self, records: list[LabelledRecord]):
        self._records = records

    def fetch_records(self) -> list[LabelledRecord]:
        return self._records


def test_cleaner_normalises_persian_text() -> None:
    cleaner = PersianTextCleaner()
    dirty = "كِتاب\u0640هاى ١٢۳!"
    assert cleaner.clean(dirty) == "کتابهای 123!"


def test_pipeline_persists_to_parquet_and_sql(tmp_path: Path) -> None:
    cleaner = PersianTextCleaner(CleaningConfig())
    engine = create_engine(f"sqlite:///{tmp_path / 'meta.db'}", future=True)
    storage = DataStorage(engine=engine, parquet_path=tmp_path / "dataset.parquet")
    record = LabelledRecord(
        record_id="42",
        text="كِتاب خوبی بود.",
        labels=["مثبت"],
        source="label_studio",
        raw_payload={"data": {"text": "كِتاب خوبی بود."}},
    )
    pipeline = DataPipeline(cleaner=cleaner, storage=storage, label_sources=[_FakeLabelSource([record])])

    dataframe = pipeline.run()

    assert dataframe["clean_text"].tolist() == ["کتاب خوبی بود."]
    assert storage.parquet_path.exists()

    metadata = pd.read_sql_table("labeled_sample_metadata", engine)
    assert metadata.shape[0] == 1
    assert metadata.iloc[0]["record_id"] == "42"
    assert metadata.iloc[0]["label_count"] == 1
    assert json.loads(metadata.iloc[0]["labels"]) == ["مثبت"]
