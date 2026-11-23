from __future__ import annotations

from ai_translator.evaluation.plugins import (
    ConsistencyEvaluator,
    FidelityEvaluator,
    ReadabilityEvaluator,
)
from ai_translator.evaluation.registry import EvaluatorRegistry

from .evaluators import (
    AuthorVoiceEvaluator,
    DoctrinalConsistencyEvaluator,
    GlossaryComplianceEvaluator,
    TerminologyDriftEvaluator,
)
from .glossary import DoctrineGlossary
from .metadata import ScriptureMetadata


def scripture_registry(*, metadata: ScriptureMetadata | None = None,
                       doctrine_glossary: DoctrineGlossary | None = None,
                       glossary_terms: tuple[dict[str, str], ...] = ()) -> EvaluatorRegistry:
    """Build a fresh registry; built-in core plugins are reused without editing core."""
    return EvaluatorRegistry([
        FidelityEvaluator(), ReadabilityEvaluator(), ConsistencyEvaluator(),
        DoctrinalConsistencyEvaluator(doctrine_glossary, metadata),
        TerminologyDriftEvaluator(glossary_terms),
        AuthorVoiceEvaluator(metadata),
        GlossaryComplianceEvaluator(glossary_terms),
    ])
