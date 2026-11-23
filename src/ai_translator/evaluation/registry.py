from __future__ import annotations

from .errors import EvaluationConfigurationError
from .plugins import (ConsistencyEvaluator, EvaluationPlugin, FidelityEvaluator,
                      ReadabilityEvaluator)


class EvaluatorRegistry:
    """Explicit plugin registry; third-party plugins register through this interface."""

    def __init__(self, plugins: list[EvaluationPlugin] | None = None):
        self._plugins: dict[str, EvaluationPlugin] = {}
        for plugin in plugins or []:
            self.register(plugin)

    def register(self, plugin: EvaluationPlugin) -> None:
        plugin_id = getattr(plugin, "plugin_id", None)
        version = getattr(plugin, "version", None)
        if not isinstance(plugin_id, str) or not plugin_id.strip():
            raise EvaluationConfigurationError("Evaluator plugin must declare a non-empty plugin_id")
        if not isinstance(version, str) or not version.strip():
            raise EvaluationConfigurationError(f"Evaluator {plugin_id} must declare a version")
        if plugin_id in self._plugins:
            raise EvaluationConfigurationError(f"Evaluator plugin {plugin_id!r} is already registered")
        if not callable(getattr(plugin, "build_prompt", None)) or not callable(getattr(plugin, "parse_output", None)):
            raise EvaluationConfigurationError(f"Evaluator {plugin_id!r} does not implement the plugin interface")
        if not isinstance(getattr(plugin, "prompt_template", None), str):
            raise EvaluationConfigurationError(f"Evaluator {plugin_id!r} must provide prompt_template text")
        self._plugins[plugin_id] = plugin

    def get(self, plugin_id: str) -> EvaluationPlugin:
        try:
            return self._plugins[plugin_id]
        except KeyError as exc:
            raise EvaluationConfigurationError(
                f"Unknown evaluator plugin {plugin_id!r}; available: {', '.join(self.ids())}"
            ) from exc

    def ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._plugins))


def default_registry() -> EvaluatorRegistry:
    return EvaluatorRegistry([FidelityEvaluator(), ReadabilityEvaluator(), ConsistencyEvaluator()])

