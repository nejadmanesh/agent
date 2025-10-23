"""High level inference engine built on top of the model loader."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping

from .config import ModelConfig
from .loader import ModelLoader


@dataclass(slots=True)
class InferenceResult:
    """Result returned after running a prediction."""

    model_name: str
    input_text: str
    output_text: str
    metadata: Mapping[str, str] = field(default_factory=dict)


class InferenceEngine:
    """Provide inference APIs that coordinate model loading and execution."""

    def __init__(
        self,
        loader: ModelLoader,
        default_model: ModelConfig,
        *,
        additional_models: Iterable[ModelConfig] | None = None,
    ) -> None:
        self._loader = loader
        self._model_configs: Dict[str, ModelConfig] = {default_model.name: default_model}
        if additional_models is not None:
            for config in additional_models:
                self._model_configs[config.name] = config
        self._default_model_name = default_model.name

    @property
    def default_model_name(self) -> str:
        return self._default_model_name

    def register_model(self, config: ModelConfig) -> None:
        self._model_configs[config.name] = config

    def predict(self, input_text: str, *, model_name: str | None = None) -> InferenceResult:
        if not input_text:
            raise ValueError("input_text cannot be empty")
        target_name = model_name or self._default_model_name
        config = self._model_configs.get(target_name)
        if config is None:
            raise KeyError(f"Unknown model '{target_name}'")
        loaded_model = self._loader.load(config)
        output = loaded_model.predict(input_text)
        metadata: Dict[str, str] = {
            "behavior": loaded_model.artifact.get("behavior", "echo"),
        }
        return InferenceResult(
            model_name=config.name,
            input_text=input_text,
            output_text=output,
            metadata=metadata,
        )

    def list_models(self) -> Mapping[str, ModelConfig]:
        return dict(self._model_configs)
