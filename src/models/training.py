"""Training utilities built on top of HuggingFace Transformers.

The module exposes a light-weight training workflow that integrates with
MLflow for experiment tracking.  The default configuration intentionally uses
small model dimensions and synthetic data so that unit tests can exercise the
end-to-end pipeline quickly without requiring external datasets or network
access.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import mlflow
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)


class SimpleSequenceDataset(Dataset):
    """Tiny synthetic dataset for binary sequence classification.

    Each example is a short token sequence accompanied by an attention mask and
    binary label.  The dataset is deterministic to provide stable behaviour
    across tests.
    """

    def __init__(self, sequences: Sequence[Sequence[int]], labels: Sequence[int]):
        if len(sequences) != len(labels):
            msg = "Sequences and labels must be the same length."
            raise ValueError(msg)
        self._sequences = [torch.tensor(seq, dtype=torch.long) for seq in sequences]
        self._labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        input_ids = self._sequences[idx]
        attention_mask = torch.ones_like(input_ids)
        label = self._labels[idx]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label,
        }


@dataclass
class TrainingConfig:
    """Container for hyper-parameters and MLflow settings."""

    output_dir: str
    mlflow_tracking_uri: str = "file:./mlruns"
    run_name: str = "transformer-training"
    num_labels: int = 2
    vocab_size: int = 32
    hidden_size: int = 32
    num_hidden_layers: int = 2
    num_attention_heads: int = 2
    intermediate_size: int = 64
    num_train_epochs: float = 5.0
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    learning_rate: float = 5e-4
    weight_decay: float = 0.0
    seed: int = 42

    def to_mlflow_params(self) -> Dict[str, str]:
        params = asdict(self)
        params.pop("mlflow_tracking_uri", None)
        return {key: str(value) for key, value in params.items()}


@dataclass
class TrainingResult:
    """Return object for :func:`train_model`."""

    metrics: Dict[str, float]
    run_id: str

    def __getitem__(self, item: str) -> float:  # pragma: no cover - convenience
        return self.metrics[item]


def _build_default_datasets() -> Tuple[SimpleSequenceDataset, SimpleSequenceDataset]:
    """Construct deterministic training and evaluation datasets."""

    train_sequences = (
        [1, 1, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 2, 1, 0, 0, 0],
        [2, 2, 2, 2, 2, 1, 1, 1],
        [2, 2, 2, 2, 2, 1, 1, 0],
        [1, 2, 1, 2, 1, 0, 0, 0],
        [2, 1, 2, 1, 2, 1, 1, 1],
    )
    train_labels = (0, 0, 1, 1, 0, 1)

    eval_sequences = (
        [1, 1, 1, 1, 0, 0, 0, 0],
        [2, 2, 2, 2, 1, 1, 1, 1],
        [1, 2, 1, 2, 0, 0, 0, 0],
        [2, 1, 2, 1, 1, 1, 1, 1],
    )
    eval_labels = (0, 1, 0, 1)

    return SimpleSequenceDataset(train_sequences, train_labels), SimpleSequenceDataset(
        eval_sequences, eval_labels
    )


def _create_model(config: TrainingConfig) -> BertForSequenceClassification:
    model_config = BertConfig(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        intermediate_size=config.intermediate_size,
        num_labels=config.num_labels,
    )
    return BertForSequenceClassification(model_config)


def _compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = float(np.mean(predictions == labels))
    return {"accuracy": accuracy}


def train_model(
    config: TrainingConfig,
    train_dataset: Optional[Dataset] = None,
    eval_dataset: Optional[Dataset] = None,
) -> TrainingResult:
    """Train a compact Transformer model and log to MLflow.

    Parameters
    ----------
    config:
        Hyper-parameters and MLflow configuration.
    train_dataset / eval_dataset:
        Optional datasets that override the deterministic defaults.  The
        function accepts objects compatible with the 🤗 ``Trainer`` API.
    """

    set_seed(config.seed)
    if train_dataset is None or eval_dataset is None:
        train_dataset, eval_dataset = _build_default_datasets()

    model = _create_model(config)
    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_path),
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="no",
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        report_to=[],  # Disable default integrations for repeatable tests.
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=_compute_metrics,
        data_collator=default_data_collator,
    )

    mlflow.set_tracking_uri(config.mlflow_tracking_uri)

    with mlflow.start_run(run_name=config.run_name) as active_run:
        mlflow.log_params(config.to_mlflow_params())
        trainer.train()
        metrics = trainer.evaluate()
        metrics = {key: float(value) for key, value in metrics.items()}
        mlflow.log_metrics(metrics)
        trainer.save_model(str(output_path))
        mlflow.log_artifacts(str(output_path), artifact_path="model")

    return TrainingResult(metrics=metrics, run_id=active_run.info.run_id)


__all__ = [
    "SimpleSequenceDataset",
    "TrainingConfig",
    "TrainingResult",
    "train_model",
]
