from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID


@dataclass(frozen=True, slots=True)
class EvaluationSegment:
    chunk_id: UUID
    stable_key: str
    source_text: str
    target_text: str
    locator: dict[str, Any]


@dataclass(frozen=True, slots=True)
class EvaluationInput:
    translation_version_id: UUID
    translation_content_hash: str | None
    source_locale: str
    target_locale: str
    domain: str
    task_kind: str
    target_reading_level: str | None
    segments: tuple[EvaluationSegment, ...]
    total_segment_count: int

    def for_segments(self, segments: tuple[EvaluationSegment, ...]) -> "EvaluationInput":
        return EvaluationInput(
            translation_version_id=self.translation_version_id,
            translation_content_hash=self.translation_content_hash,
            source_locale=self.source_locale,
            target_locale=self.target_locale,
            domain=self.domain,
            task_kind=self.task_kind,
            target_reading_level=self.target_reading_level,
            segments=segments,
            total_segment_count=self.total_segment_count,
        )


@dataclass(frozen=True, slots=True)
class EvaluationOutput:
    score: float
    confidence: float
    reasoning: str
    recommendations: tuple[str, ...]
    detected_issues: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "recommendations": list(self.recommendations),
            "detected_issues": list(self.detected_issues),
        }


@dataclass(frozen=True, slots=True)
class EvaluationCandidateResult:
    plugin_id: str
    model_alias: str
    model_name: str
    evaluation_version_id: UUID
    version_number: int
    status: str
    score: float | None
    confidence: float | None
    issue_count: int
    recommendation_count: int
    error: str | None = None

