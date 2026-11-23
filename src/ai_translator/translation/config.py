from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .errors import TranslationConfigurationError


SUPPORTED_FAMILIES = frozenset({"qwen", "mistral", "gemma"})


@dataclass(frozen=True, slots=True)
class ModelConfig:
    alias: str
    model_name: str
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TranslationConfig:
    base_url: str
    timeout_seconds: float
    candidate_models: tuple[str, ...]
    models: dict[str, ModelConfig]
    target_locale: str | None = None
    task_kind: str = "translation"
    target_reading_level: str | None = None
    temperature: float = 0.2
    num_ctx: int | None = None
    num_predict: int | None = None
    prompt_file: Path | None = None

    @classmethod
    def from_file(cls, path: str | Path) -> "TranslationConfig":
        config_path = Path(path).expanduser().resolve()
        try:
            with config_path.open("rb") as stream:
                raw = tomllib.load(stream)
        except OSError as exc:
            raise TranslationConfigurationError(f"Cannot read model configuration {config_path}: {exc}") from exc
        except tomllib.TOMLDecodeError as exc:
            raise TranslationConfigurationError(f"Invalid TOML in {config_path}: {exc}") from exc
        ollama = raw.get("ollama", {})
        translation = raw.get("translation", {})
        if not isinstance(ollama, dict) or not isinstance(translation, dict):
            raise TranslationConfigurationError("[ollama] and [translation] must be TOML tables")
        base_url = str(ollama.get("base_url", "http://localhost:11434")).rstrip("/")
        parsed_url = urlparse(base_url)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            raise TranslationConfigurationError("ollama.base_url must be an HTTP or HTTPS URL")
        timeout = float(ollama.get("timeout_seconds", 600))
        if timeout <= 0:
            raise TranslationConfigurationError("ollama.timeout_seconds must be positive")
        raw_models = raw.get("models", {})
        if not isinstance(raw_models, dict):
            raise TranslationConfigurationError("[models] must be a TOML table")
        models: dict[str, ModelConfig] = {}
        for alias, item in raw_models.items():
            if isinstance(item, str):
                model_name, options = item, {}
            elif isinstance(item, dict):
                model_name = item.get("name", item.get("model"))
                options = item.get("options", {})
                if not isinstance(options, dict):
                    raise TranslationConfigurationError(f"models.{alias}.options must be a table")
            else:
                raise TranslationConfigurationError(f"models.{alias} must be a model name or table")
            if not isinstance(model_name, str) or not model_name.strip():
                raise TranslationConfigurationError(f"models.{alias}.name is required")
            models[alias] = ModelConfig(alias=alias, model_name=model_name.strip(), options=dict(options))
        candidates = translation.get("candidate_models", list(models))
        if not isinstance(candidates, list) or not candidates or not all(isinstance(v, str) and v for v in candidates):
            raise TranslationConfigurationError("translation.candidate_models must be a non-empty string array")
        missing = [alias for alias in candidates if alias not in models]
        if missing:
            raise TranslationConfigurationError(f"No [models.{missing[0]}] configuration was found")
        prompt_file = translation.get("prompt_file")
        if prompt_file is not None:
            prompt_file = Path(prompt_file).expanduser()
            if not prompt_file.is_absolute():
                prompt_file = config_path.parent / prompt_file
            prompt_file = prompt_file.resolve()
        target_locale = translation.get("target_locale")
        if target_locale is not None and not isinstance(target_locale, str):
            raise TranslationConfigurationError("translation.target_locale must be a string")
        target_reading_level = translation.get("target_reading_level")
        if target_reading_level is not None and not isinstance(target_reading_level, str):
            raise TranslationConfigurationError("translation.target_reading_level must be a string")
        return cls(
            base_url=base_url,
            timeout_seconds=timeout,
            candidate_models=tuple(candidates),
            models=models,
            target_locale=target_locale,
            task_kind=str(translation.get("task_kind", "translation")),
            target_reading_level=target_reading_level,
            temperature=float(translation.get("temperature", 0.2)),
            num_ctx=translation.get("num_ctx"),
            num_predict=translation.get("num_predict"),
            prompt_file=prompt_file,
        )

    def resolve_models(self, selection: list[str] | tuple[str, ...] | None = None) -> tuple[ModelConfig, ...]:
        requested = tuple(selection) if selection is not None else self.candidate_models
        if not requested:
            raise TranslationConfigurationError("At least one candidate model must be selected")
        resolved: list[ModelConfig] = []
        for name in requested:
            if name in self.models:
                model = self.models[name]
            else:
                family = name.split(":", 1)[0].lower()
                is_supported_family = any(family.startswith(item) for item in SUPPORTED_FAMILIES)
                if not is_supported_family and "/" not in name and ":" not in name:
                    raise TranslationConfigurationError(
                        f"Unknown model alias {name!r}; configure it under [models.{name}] or use qwen, mistral, or gemma"
                    )
                model = ModelConfig(alias=name, model_name=name)
            resolved.append(model)
        if len({m.alias for m in resolved}) != len(resolved):
            raise TranslationConfigurationError("Candidate model selection contains duplicates")
        return tuple(resolved)
