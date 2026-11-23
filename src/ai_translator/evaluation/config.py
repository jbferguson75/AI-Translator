from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .errors import EvaluationConfigurationError


@dataclass(frozen=True, slots=True)
class EvaluatorModel:
    alias: str
    model_name: str
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    base_url: str
    timeout_seconds: float
    evaluator_models: dict[str, EvaluatorModel]
    candidate_models: tuple[str, ...]
    plugin_ids: tuple[str, ...]
    batch_size: int = 10
    temperature: float = 0.0
    num_ctx: int | None = None

    @classmethod
    def from_file(cls, path: str | Path) -> "EvaluationConfig":
        config_path = Path(path).expanduser().resolve()
        try:
            with config_path.open("rb") as stream:
                raw = tomllib.load(stream)
        except OSError as exc:
            raise EvaluationConfigurationError(f"Cannot read evaluator configuration {config_path}: {exc}") from exc
        except tomllib.TOMLDecodeError as exc:
            raise EvaluationConfigurationError(f"Invalid TOML in {config_path}: {exc}") from exc
        ollama = raw.get("ollama", {})
        evaluation = raw.get("evaluation", {})
        models_table = raw.get("evaluator_models", {})
        if not all(isinstance(value, dict) for value in (ollama, evaluation, models_table)):
            raise EvaluationConfigurationError("[ollama], [evaluation], and [evaluator_models] must be tables")
        base_url = str(ollama.get("base_url", "http://localhost:11434")).rstrip("/")
        parsed = urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise EvaluationConfigurationError("ollama.base_url must be an HTTP or HTTPS URL")
        timeout = float(ollama.get("timeout_seconds", 600))
        if timeout <= 0:
            raise EvaluationConfigurationError("ollama.timeout_seconds must be positive")
        model_map: dict[str, EvaluatorModel] = {}
        for alias, record in models_table.items():
            if isinstance(record, str):
                name, options = record, {}
            elif isinstance(record, dict):
                name = record.get("name", record.get("model"))
                options = record.get("options", {})
            else:
                raise EvaluationConfigurationError(f"evaluator_models.{alias} must be a name or table")
            if not isinstance(name, str) or not name.strip() or not isinstance(options, dict):
                raise EvaluationConfigurationError(f"evaluator_models.{alias} requires a model name and options table")
            model_map[alias] = EvaluatorModel(alias, name.strip(), dict(options))
        candidates = evaluation.get("candidate_models", list(model_map))
        plugins = evaluation.get("plugins", ["fidelity", "readability", "consistency"])
        if not isinstance(candidates, list) or not candidates or not all(isinstance(v, str) for v in candidates):
            raise EvaluationConfigurationError("evaluation.candidate_models must be a non-empty string array")
        if not isinstance(plugins, list) or not plugins or not all(isinstance(v, str) for v in plugins):
            raise EvaluationConfigurationError("evaluation.plugins must be a non-empty string array")
        missing = [name for name in candidates if name not in model_map]
        if missing:
            raise EvaluationConfigurationError(f"No [evaluator_models.{missing[0]}] configuration was found")
        batch_size = int(evaluation.get("batch_size", 10))
        if batch_size < 1:
            raise EvaluationConfigurationError("evaluation.batch_size must be positive")
        return cls(
            base_url=base_url,
            timeout_seconds=timeout,
            evaluator_models=model_map,
            candidate_models=tuple(candidates),
            plugin_ids=tuple(plugins),
            batch_size=batch_size,
            temperature=float(evaluation.get("temperature", 0.0)),
            num_ctx=evaluation.get("num_ctx"),
        )

    def resolve_models(self, selection: list[str] | tuple[str, ...] | None = None) -> tuple[EvaluatorModel, ...]:
        aliases = tuple(selection) if selection is not None else self.candidate_models
        if not aliases:
            raise EvaluationConfigurationError("Select at least one evaluator model")
        try:
            models = tuple(self.evaluator_models[alias] for alias in aliases)
        except KeyError as exc:
            raise EvaluationConfigurationError(f"Evaluator model alias {exc.args[0]!r} is not configured") from exc
        if len({model.alias for model in models}) != len(models):
            raise EvaluationConfigurationError("Evaluator model selection contains duplicates")
        return models
