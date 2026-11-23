from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from ai_translator.db.models import (
    Chunk,
    Evaluation,
    EvaluationResult,
    EvaluationVersion,
    TranslationSegment,
    TranslationVersion,
)

from .drift import TerminologyDriftDetector
from .glossary import DoctrineGlossary
from .metadata import ScriptureMetadata


SCRIPTURE_PLUGIN_IDS = (
    "scripture.doctrinal_consistency",
    "scripture.terminology_drift",
    "scripture.author_voice",
    "scripture.glossary_compliance",
)


class ScriptureReviewReport:
    def __init__(self, *, metadata: ScriptureMetadata | None = None,
                 doctrine_glossary: DoctrineGlossary | None = None):
        self.metadata = metadata or ScriptureMetadata.load()
        self.doctrine_glossary = doctrine_glossary or DoctrineGlossary.load()

    def render(self, session: Session, translation_version_id: UUID,
               glossary_terms: tuple[dict[str, str], ...] = ()) -> str:
        version = session.get(TranslationVersion, translation_version_id)
        if version is None:
            raise ValueError(f"Translation version {translation_version_id} does not exist")
        translation = version.translation
        document = translation.document
        rows = session.execute(
            select(TranslationSegment, Chunk)
            .join(Chunk, Chunk.id == TranslationSegment.chunk_id)
            .where(TranslationSegment.translation_version_id == version.id)
            .order_by(Chunk.ordinal)
        ).all()
        if not rows:
            raise ValueError("Translation version has no translated segments")

        authors: dict[str, set[str]] = {}
        segments = []
        for segment, chunk in rows:
            locator = dict(chunk.locator or {})
            book = str(locator.get("book", ""))
            profile = self.metadata.author_for_book(book)
            author = str(locator.get("author") or (profile.name if profile else "Unknown"))
            authors.setdefault(author, set()).add(book or chunk.stable_key)
            from ai_translator.evaluation.types import EvaluationSegment
            segments.append(EvaluationSegment(
                chunk_id=chunk.id, stable_key=chunk.stable_key, source_text=chunk.source_text,
                target_text=segment.target_text, locator=locator,
            ))
        findings = TerminologyDriftDetector().scan(segments, glossary_terms)

        evaluation_rows = list(session.execute(
            select(EvaluationVersion, Evaluation, EvaluationResult)
            .join(Evaluation, Evaluation.id == EvaluationVersion.evaluation_id)
            .outerjoin(EvaluationResult, EvaluationResult.evaluation_version_id == EvaluationVersion.id)
            .where(EvaluationVersion.translation_version_id == version.id,
                   Evaluation.evaluator_name.in_(SCRIPTURE_PLUGIN_IDS))
            .order_by(Evaluation.evaluator_name, EvaluationVersion.version_number)
        ).all())

        lines = [
            "# Scripture Review Report",
            "",
            f"- Generated: {datetime.now(timezone.utc).isoformat()}",
            f"- Document: {self._md(document.title)} (`{document.id}`)",
            f"- Translation version: `{version.id}` (version {version.version_number})",
            f"- Translation content hash: `{version.content_hash or 'unavailable'}`",
            f"- Language pair: {document.source_locale} → {translation.target_locale}",
            f"- Translation task: {translation.task_kind}; reading level: {translation.target_reading_level or 'unspecified'}",
            f"- Segments reviewed: {len(rows)}",
            "",
            "## Author metadata",
            "",
        ]
        for author, books in sorted(authors.items()):
            lines.append(f"- **{self._md(author)}** — {', '.join(f'`{self._md(book)}`' for book in sorted(books))}")
        lines.extend(["", "## Doctrine glossary reference", ""])
        if not self.doctrine_glossary.entries:
            lines.append("No plugin-owned doctrine glossary terms are configured.")
        for term in self.doctrine_glossary.entries:
            preferred = f" → {self._md(term.preferred_translation)}" if term.preferred_translation else ""
            scope = []
            if term.target_locale:
                scope.append(f"locale {term.target_locale}")
            if term.author:
                scope.append(f"author {term.author}")
            suffix = f" ({'; '.join(scope)})" if scope else ""
            lines.append(f"- **{self._md(term.source_term)}**{preferred}{suffix}: {self._md(term.notes)}")
        lines.extend(["", "## Automated evaluator results", ""])
        if not evaluation_rows:
            lines.append("No Scripture plugin evaluation results are stored for this translation version.")
        for evaluation_version, evaluation, result in evaluation_rows:
            lines.append(
                f"### {self._md(evaluation.evaluator_name or 'Scripture evaluator')} — "
                f"version {evaluation_version.version_number} ({evaluation_version.status})"
            )
            lines.append("")
            if result is None:
                lines.append("No aggregate result was stored for this evaluation version.")
                lines.append("")
                continue
            model = (evaluation_version.model_run.model_name
                     if evaluation_version.model_run else "unknown model")
            lines.extend([
                f"- Model: {self._md(model)}",
                f"- Evaluator configuration version: `{evaluation_version.rubric_version or 'unspecified'}`",
                f"- Score: {result.score if result.score is not None else 'n/a'}; confidence: "
                f"{result.confidence if result.confidence is not None else 'n/a'}",
                f"- Evaluation version ID: `{evaluation_version.id}`",
                "",
                result.reasoning or "No reasoning stored.",
                "",
            ])
            if result.detected_issues:
                lines.append("**Issues**")
                lines.extend(f"- {self._md(item)}" for item in result.detected_issues)
                lines.append("")
            if result.recommendations:
                lines.append("**Recommendations**")
                lines.extend(f"- {self._md(item)}" for item in result.recommendations)
                lines.append("")

        lines.extend(["## Deterministic glossary drift scan", ""])
        lines.append(
            "This scan flags repeated source terms where the configured preferred target form is absent. "
            "Treat findings as review leads; inflection, paraphrase, and context can produce false positives."
        )
        lines.append("")
        if not findings:
            lines.append("No repeated glossary terms with missing preferred forms were detected.")
        for finding in findings:
            lines.extend([
                f"### {self._md(finding.source_term)} → {self._md(finding.preferred_translation)}",
                f"- Appears in: {', '.join(f'`{self._md(ref)}`' for ref in finding.references)}",
                f"- Preferred form not found in: {', '.join(f'`{self._md(ref)}`' for ref in finding.missing_references)}",
                "- Target samples:",
                *(f"  - {self._md(sample)}" for sample in finding.target_samples),
                "",
            ])
        lines.extend([
            "## Review note", "",
            "Automated scores and findings assist human review and do not establish doctrinal correctness. "
            "Review source context, author attribution, and target-language usage before changing a translation "
            "or glossary.", "",
        ])
        return "\n".join(lines)

    @staticmethod
    def _md(value: str) -> str:
        return value.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")
