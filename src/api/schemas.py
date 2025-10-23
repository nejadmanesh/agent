"""Pydantic models for API request and response bodies."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class InferenceRequest(BaseModel):
    input_text: str = Field(..., description="Raw text that should be processed by the model.")
    model_name: Optional[str] = Field(
        default=None,
        description="Optional identifier of the model to use. Defaults to the configured model.",
    )


class InferenceResponse(BaseModel):
    task_id: str = Field(..., description="Identifier of the Celery task handling the inference request.")
    status: str = Field(..., description="Current status of the request in the processing pipeline.")


class InferenceResultResponse(BaseModel):
    task_id: str
    model_name: str
    input_text: str
    output_text: Optional[str]
    status: str
    error: Optional[str]
    created_at: datetime


class SynchronousInferenceResponse(BaseModel):
    model_name: str
    input_text: str
    output_text: str
    metadata: dict[str, str]
