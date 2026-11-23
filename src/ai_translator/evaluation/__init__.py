"""Plugin-based evaluation of immutable translation versions."""

from .plugins import EvaluationPlugin, FidelityEvaluator, ReadabilityEvaluator, ConsistencyEvaluator
from .registry import EvaluatorRegistry, default_registry
from .service import EvaluationService

__all__ = [
    "EvaluationPlugin", "FidelityEvaluator", "ReadabilityEvaluator",
    "ConsistencyEvaluator", "EvaluatorRegistry", "default_registry",
    "EvaluationService",
]

