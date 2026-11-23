"""Isolated scripture-specific evaluation plugin for AI Translator."""

from .evaluators import (
    AuthorVoiceEvaluator,
    DoctrinalConsistencyEvaluator,
    GlossaryComplianceEvaluator,
    TerminologyDriftEvaluator,
)
from .registry import scripture_registry

__all__ = [
    "AuthorVoiceEvaluator",
    "DoctrinalConsistencyEvaluator",
    "GlossaryComplianceEvaluator",
    "TerminologyDriftEvaluator",
    "scripture_registry",
]
