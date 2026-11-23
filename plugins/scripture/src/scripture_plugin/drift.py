from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from ai_translator.evaluation.types import EvaluationSegment


@dataclass(frozen=True, slots=True)
class DriftFinding:
    source_term: str
    preferred_translation: str
    references: tuple[str, ...]
    missing_references: tuple[str, ...]
    target_samples: tuple[str, ...]


class TerminologyDriftDetector:
    """Deterministically finds repeated glossary terms not rendered with their approved form."""

    def scan(self, segments: Iterable[EvaluationSegment],
             glossary_terms: tuple[dict[str, str], ...]) -> tuple[DriftFinding, ...]:
        source_segments = tuple(segments)
        findings: list[DriftFinding] = []
        for item in glossary_terms:
            source_term = item.get("source_term", "").strip()
            preferred = item.get("preferred_translation", "").strip()
            if not source_term or not preferred:
                continue
            source_pattern = self._term_pattern(source_term)
            target_pattern = self._term_pattern(preferred)
            matches = [segment for segment in source_segments
                       if source_pattern.search(segment.source_text)]
            if len(matches) < 2:
                continue
            missing = [segment for segment in matches if not target_pattern.search(segment.target_text)]
            if not missing:
                continue
            refs = tuple(segment.stable_key for segment in matches)
            missing_refs = tuple(segment.stable_key for segment in missing)
            findings.append(DriftFinding(
                source_term=source_term,
                preferred_translation=preferred,
                references=refs,
                missing_references=missing_refs,
                target_samples=tuple(self._sample(segment.target_text) for segment in missing[:5]),
            ))
        return tuple(findings)

    @staticmethod
    def _term_pattern(term: str) -> re.Pattern[str]:
        return re.compile(r"(?<!\w)" + re.escape(term) + r"(?!\w)", re.IGNORECASE | re.UNICODE)

    @staticmethod
    def _sample(text: str, limit: int = 220) -> str:
        normalized = " ".join(text.split())
        return normalized if len(normalized) <= limit else normalized[:limit - 1] + "…"
