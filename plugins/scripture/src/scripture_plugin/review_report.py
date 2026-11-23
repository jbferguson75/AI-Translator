from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from ai_translator.db.models import (
    Chunk, Evaluation, EvaluationResult, EvaluationVersion,
    Translation, TranslationSegment, TranslationVersion,
)
from ai_translator.evaluation.types import EvaluationSegment
from ai_translator.glossary.service import GlossaryService

from .drift import TerminologyDriftDetector
from .review_service import LEVEL_NAMES


class HierarchicalConsistencyReport:
    def render(self, session: Session, translation_version_id: UUID,
               review_run_id: UUID | None = None) -> str:
        version = session.get(TranslationVersion, translation_version_id)
        if version is None:
            raise ValueError(f"Translation version {translation_version_id} does not exist")
        translation: Translation = version.translation
        if translation.document.domain.casefold() != "scripture":
            raise ValueError("Consistency reports require a scripture document")
        rows = list(session.execute(
            select(EvaluationVersion, Evaluation, EvaluationResult)
            .join(Evaluation, Evaluation.id == EvaluationVersion.evaluation_id)
            .outerjoin(EvaluationResult, EvaluationResult.evaluation_version_id == EvaluationVersion.id)
            .where(EvaluationVersion.translation_version_id == version.id,
                   Evaluation.evaluator_name.like("hierarchical:%"))
            .order_by(EvaluationVersion.created_at)
        ).all())
        if review_run_id is not None:
            rows = [row for row in rows if row[0].run_manifest.get("review_run_id") == str(review_run_id)]
        elif rows:
            latest_run = rows[-1][0].run_manifest.get("review_run_id")
            rows = [row for row in rows if row[0].run_manifest.get("review_run_id") == latest_run]

        segments = tuple(self._load_segments(session, version.id))
        entries = GlossaryService(session).entries_for_scope(
            translation.document.project_id, translation.document.source_locale, translation.target_locale,
        )
        term_records = tuple({"source_term": entry.source_term,
                              "preferred_translation": entry.target_term}
                             for entry in entries)
        drift = TerminologyDriftDetector().scan(segments, term_records)

        scores: dict[int, tuple[float, int]] = {}
        for evaluation_version, _evaluation, result in rows:
            level = evaluation_version.run_manifest.get("level")
            count = int(evaluation_version.run_manifest.get("input_segment_count", 1))
            if result is None or result.score is None or evaluation_version.status != "completed":
                continue
            current_total, current_count = scores.get(level, (0.0, 0))
            scores[level] = (current_total + result.score * count, current_count + count)

        report_run = (str(review_run_id) if review_run_id else
                      (str(rows[0][0].run_manifest.get("review_run_id")) if rows else "none"))
        lines = [
            "# Hierarchical Scripture Consistency Report", "",
            f"- Document: {translation.document.title} (`{translation.document_id}`)",
            f"- Translation version: `{version.id}` (v{version.version_number})",
            f"- Source hash: `{version.content_hash or 'unavailable'}`",
            f"- Review run: `{report_run}`",
            f"- Evaluator records: {len(rows)}",
            "", "## Aggregated level scores", "",
            "Completed evaluator/model results are averaged with segment-count weighting. Failed and partial "
            "results remain visible below but are excluded from aggregate scores.", "",
        ]
        for level in range(1, 5):
            if level in scores:
                total, weight = scores[level]
                lines.append(f"- Level {level} ({LEVEL_NAMES[level]}): **{total / weight:.2f}** ({weight} segment-weight units)")
            else:
                lines.append(f"- Level {level} ({LEVEL_NAMES[level]}): no completed scores")
        lines.extend(["", "## Scope results", ""])
        if not rows:
            lines.append("No hierarchical reviews are stored for this version and run.")
        for evaluation_version, evaluation, result in rows:
            manifest = evaluation_version.run_manifest
            level = int(manifest.get("level", 0))
            label = manifest.get("scope_label", manifest.get("scope_key", "unknown scope"))
            lines.extend([
                f"### Level {level} ({LEVEL_NAMES.get(level, 'unknown')}) — {label}",
                f"- Evaluator: `{evaluation_version.plugin_manifest.get('plugin_id', evaluation.evaluator_name)}`",
                f"- Version: `{evaluation_version.id}`; status: {evaluation_version.status}",
            ])
            if evaluation_version.model_run:
                lines.append(f"- Model: {evaluation_version.model_run.model_name}")
            if result is not None:
                lines.append(f"- Score: {result.score}; confidence: {result.confidence}")
                if result.reasoning:
                    lines.append(f"- Reasoning: {self._md(result.reasoning)}")
                if result.detected_issues:
                    lines.append("- Issues: " + "; ".join(self._md(item) for item in result.detected_issues))
                if result.recommendations:
                    lines.append("- Recommendations: " + "; ".join(self._md(item) for item in result.recommendations))
            elif manifest.get("error"):
                lines.append(f"- Error: {self._md(str(manifest['error']))}")
            lines.append("")
        lines.extend(["## Project glossary drift scan", "",
                      "Repeated source terms whose approved target form is absent in any occurrence:", ""])
        if not drift:
            lines.append("No findings detected.")
        for finding in drift:
            lines.append(
                f"- **{self._md(finding.source_term)} → {self._md(finding.preferred_translation)}**: "
                f"preferred form missing at {', '.join(f'`{self._md(ref)}`' for ref in finding.missing_references)}"
            )
        lines.extend(["", "Automated findings support human review; they do not automatically change translation text.", ""])
        return "\n".join(lines)

    @staticmethod
    def _load_segments(session: Session, translation_version_id: UUID):
        rows = session.execute(
            select(TranslationSegment, Chunk)
            .join(Chunk, Chunk.id == TranslationSegment.chunk_id)
            .where(TranslationSegment.translation_version_id == translation_version_id)
            .order_by(Chunk.ordinal)
        ).all()
        for segment, chunk in rows:
            yield EvaluationSegment(chunk.id, chunk.stable_key, chunk.source_text,
                                    segment.target_text, dict(chunk.locator or {}))

    @staticmethod
    def _md(value: str) -> str:
        return value.replace("|", "\\|").replace("\n", " ")
