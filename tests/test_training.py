"""Tests for the lightweight Transformer training workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

from mlflow.tracking import MlflowClient
import pytest

from models import TrainingConfig, train_model, TrainingResult


@pytest.fixture(scope="module")
def training_result(tmp_path_factory: pytest.TempPathFactory) -> Tuple[TrainingResult, Path]:
    root_dir = tmp_path_factory.mktemp("trainer")
    tracking_dir = root_dir / "mlruns"
    output_dir = root_dir / "model"

    config = TrainingConfig(
        output_dir=str(output_dir),
        mlflow_tracking_uri=tracking_dir.as_uri(),
        run_name="unit-test-run",
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=2,
        num_hidden_layers=2,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        learning_rate=5e-4,
    )

    result = train_model(config)
    return result, tracking_dir


def test_train_model_produces_metrics(training_result: Tuple[TrainingResult, Path]) -> None:
    result, _ = training_result
    assert "eval_loss" in result.metrics
    assert "eval_accuracy" in result.metrics
    assert 0.0 <= result.metrics["eval_accuracy"] <= 1.0


def test_mlflow_logs_metrics(training_result: Tuple[TrainingResult, Path]) -> None:
    result, tracking_dir = training_result
    client = MlflowClient(tracking_uri=tracking_dir.as_uri())
    experiment = client.get_experiment_by_name("Default")
    assert experiment is not None

    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    run_ids = {run.info.run_id for run in runs}
    assert result.run_id in run_ids

    matching_run = next(run for run in runs if run.info.run_id == result.run_id)
    assert "eval_accuracy" in matching_run.data.metrics
