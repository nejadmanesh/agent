"""Adapters for ingesting human annotations from labelling tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional

import requests


@dataclass
class LabelledRecord:
    """Container for a labelled text sample."""

    record_id: str
    text: str
    labels: List[str]
    source: str
    raw_payload: Dict[str, Any]
    clean_text: Optional[str] = None


@dataclass
class LabelStudioClient:
    """Fetch annotated samples from a Label Studio project."""

    base_url: str
    api_token: str
    project_id: int
    session: requests.Session = field(default_factory=requests.Session)

    def fetch_records(self) -> List[LabelledRecord]:
        url = f"{self.base_url.rstrip('/')}/api/projects/{self.project_id}/export"
        response = self.session.get(
            url,
            headers={"Authorization": f"Token {self.api_token}"},
            params={"exportType": "JSON"},
            timeout=60,
        )
        response.raise_for_status()
        exported: Iterable[Dict[str, Any]] = response.json()
        records: List[LabelledRecord] = []
        for task in exported:
            text = _extract_label_studio_text(task)
            if text is None:
                continue
            labels = _extract_label_studio_labels(task)
            record_id = str(task.get("id", len(records)))
            records.append(
                LabelledRecord(
                    record_id=record_id,
                    text=text,
                    labels=labels,
                    source="label_studio",
                    raw_payload=dict(task),
                )
            )
        return records


def _extract_label_studio_text(task: Mapping[str, Any]) -> Optional[str]:
    data = task.get("data")
    if isinstance(data, Mapping):
        for key in ("text", "Text", "content"):
            value = data.get(key)
            if isinstance(value, str):
                return value
    return None


def _extract_label_studio_labels(task: Mapping[str, Any]) -> List[str]:
    annotations = task.get("annotations", [])
    labels: List[str] = []
    if not isinstance(annotations, Iterable):
        return labels
    for annotation in annotations:
        if not isinstance(annotation, Mapping):
            continue
        for result in annotation.get("result", []):
            if not isinstance(result, Mapping):
                continue
            value = result.get("value")
            if isinstance(value, Mapping):
                label_values = value.get("labels")
                if isinstance(label_values, list):
                    labels.extend(str(label) for label in label_values)
    return labels


@dataclass
class DoccanoClient:
    """Fetch annotated samples from Doccano."""

    base_url: str
    api_token: str
    project_id: int
    label_mapping: Optional[Mapping[int, str]] = None
    session: requests.Session = field(default_factory=requests.Session)

    def fetch_records(self) -> List[LabelledRecord]:
        url = f"{self.base_url.rstrip('/')}/v1/projects/{self.project_id}/docs/"
        headers = {"Authorization": f"Token {self.api_token}"}
        params: Dict[str, Any] = {"page": 1}
        records: List[LabelledRecord] = []
        next_url: Optional[str] = url
        while next_url:
            response = self.session.get(next_url, headers=headers, params=params, timeout=60)
            response.raise_for_status()
            payload = response.json()
            params = {}
            next_url = payload.get("next")
            for item in payload.get("results", []):
                if not isinstance(item, Mapping):
                    continue
                text = item.get("text")
                if not isinstance(text, str):
                    continue
                labels = _extract_doccano_labels(item.get("annotations", []), self.label_mapping)
                record_id = str(item.get("id", len(records)))
                records.append(
                    LabelledRecord(
                        record_id=record_id,
                        text=text,
                        labels=labels,
                        source="doccano",
                        raw_payload=dict(item),
                    )
                )
        return records


def _extract_doccano_labels(
    annotations: Any, label_mapping: Optional[Mapping[int, str]]
) -> List[str]:
    labels: List[str] = []
    if not isinstance(annotations, Iterable):
        return labels
    for annotation in annotations:
        if not isinstance(annotation, Mapping):
            continue
        label_id = annotation.get("label")
        if isinstance(label_id, int):
            if label_mapping and label_id in label_mapping:
                labels.append(str(label_mapping[label_id]))
            else:
                labels.append(str(label_id))
        elif isinstance(annotation.get("labels"), list):
            labels.extend(str(label) for label in annotation["labels"])
    return labels
