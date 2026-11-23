from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Callable
from uuid import UUID, uuid4

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ai_translator.db.models import (
    Chunk, Evaluation, EvaluationResult, EvaluationVersion, ModelRun,
    PromptVersion, Translation, TranslationSegment, TranslationVersion,
)
from ai_translator.evaluation.config import EvaluationConfig, EvaluatorModel
from ai_translator.evaluation.plugins import EvaluationPlugin
from ai_translator.evaluation.types import EvaluationInput, EvaluationOutput, EvaluationSegment
from ai_translator.glossary.service import GlossaryService
from ai_translator.translation.errors import OllamaError
from ai_translator.translation.ollama import Generation, OllamaClient

from .evaluators import (
    AuthorVoiceEvaluator, DoctrinalConsistencyEvaluator, GlossaryComplianceEvaluator,
    TerminologyDriftEvaluator,
)
from .glossary import DoctrineGlossary
from .metadata import ScriptureMetadata
from .review_plugins import (
    ChapterFlowEvaluator, ContextPreservationEvaluator, ProjectTerminologyUsageEvaluator,
    PronounResolutionEvaluator, RepeatedPhraseEvaluator,
)


LEVEL_PLUGINS: dict[int, tuple[str, ...]] = {
    1: ("fidelity", "readability", "scripture.glossary_compliance"),
    2: ("consistency", "scripture.chapter_flow", "scripture.context_preservation",
        "scripture.pronoun_resolution"),
    3: ("scripture.author_voice", "scripture.terminology_drift",
        "scripture.repeated_phrase_consistency"),
    4: ("scripture.doctrinal_consistency", "scripture.terminology_drift",
        "scripture.glossary_compliance", "scripture.project_terminology_usage"),
}
LEVEL_NAMES = {1: "verse", 2: "chapter", 3: "book", 4: "global"}


@dataclass(frozen=True, slots=True)
class ScopeResult:
    scope_key: str
    label: str
    segment_count: int
    segments: tuple[EvaluationSegment, ...]


@dataclass(frozen=True, slots=True)
class ReviewCandidate:
    level: int
    level_name: str
    scope_key: str
    scope_label: str
    segment_count: int
    plugin_id: str
    model_alias: str
    model_name: str
    evaluation_version_id: UUID
    version_number: int
    status: str
    score: float | None
    confidence: float | None
    issue_count: int
    error: str | None


@dataclass(frozen=True, slots=True)
class ReviewSummary:
    run_id: UUID
    translation_version_id: UUID
    candidates: tuple[ReviewCandidate, ...]
    level_scores: dict[int, float | None]
    overall_score: float | None


class HierarchicalReviewService:
    """Run versioned evaluators at verse, chapter, book, or document scope."""

    def __init__(self, session: Session, client: OllamaClient, *,
                 metadata: ScriptureMetadata | None = None,
                 doctrine_glossary: DoctrineGlossary | None = None):
        self.session = session
        self.client = client
        self.metadata = metadata or ScriptureMetadata.load()
        self.doctrine_glossary = doctrine_glossary or DoctrineGlossary.load()
        self._model_digests: dict[str, str] | None = None

    def run(self, *, translation_version_id: UUID, level: int,
            config: EvaluationConfig, model_selection: list[str] | None = None,
            book: str | None = None, chapter: int | None = None, verse: int | None = None,
            review_run_id: UUID | None = None,
            progress_callback: Callable[[str], None] | None = None) -> ReviewSummary:
        if level not in LEVEL_PLUGINS:
            raise ValueError("Review level must be 1, 2, 3, or 4")
        version = self.session.get(TranslationVersion, translation_version_id)
        if version is None:
            raise ValueError(f"Translation version {translation_version_id} does not exist")
        translation: Translation = version.translation
        document = translation.document
        if document.domain.casefold() != "scripture":
            raise ValueError("Hierarchical review requires a document with domain 'scripture'")
        segments = self._load_segments(version.id)
        scopes = self._scopes(level, segments, book=book, chapter=chapter, verse=verse)
        if not scopes:
            raise ValueError("No translation segments match the requested review scope")
        models = config.resolve_models(model_selection)
        run_id = review_run_id or uuid4()
        results: list[ReviewCandidate] = []
        total = sum(len(LEVEL_PLUGINS[level]) * len(models) for _scope in scopes)
        done = 0
        glossary_terms = self._glossary_terms(translation)
        project_usage = self._project_usage(translation, glossary_terms)

        for scope in scopes:
            scope_input = EvaluationInput(
                translation_version_id=version.id,
                translation_content_hash=version.content_hash,
                source_locale=document.source_locale,
                target_locale=translation.target_locale,
                domain=document.domain,
                task_kind=translation.task_kind,
                target_reading_level=translation.target_reading_level,
                segments=scope.segments,
                total_segment_count=len(scope.segments),
            )
            for plugin_id in LEVEL_PLUGINS[level]:
                plugin = self._plugin(plugin_id, glossary_terms, project_usage)
                for model in models:
                    done += 1
                    if progress_callback:
                        progress_callback(f"[{done}/{total}] {LEVEL_NAMES[level]} {scope.label}: {plugin_id} / {model.alias}")
                    results.append(self._run_candidate(
                        run_id=run_id, level=level, scope=scope, plugin=plugin,
                        model=model, config=config, evaluation_input=scope_input,
                    ))
        level_scores = {level: self._weighted_score(results)}
        overall = level_scores[level]
        return ReviewSummary(run_id, version.id, tuple(results), level_scores, overall)

    def run_all(self, *, translation_version_id: UUID, levels: tuple[int, ...],
                config: EvaluationConfig, model_selection: list[str] | None = None,
                book: str | None = None, chapter: int | None = None, verse: int | None = None,
                progress_callback: Callable[[str], None] | None = None) -> ReviewSummary:
        candidates: list[ReviewCandidate] = []
        scores: dict[int, float | None] = {}
        run_id = uuid4()
        for level in levels:
            summary = self.run(
                translation_version_id=translation_version_id, level=level,
                config=config, model_selection=model_selection,
                book=book, chapter=chapter, verse=verse, review_run_id=run_id,
                progress_callback=progress_callback,
            )
            candidates.extend(summary.candidates)
            scores[level] = summary.level_scores[level]
        finite_scores = [score for score in scores.values() if score is not None]
        overall = round(sum(finite_scores) / len(finite_scores), 2) if finite_scores else None
        return ReviewSummary(run_id, translation_version_id, tuple(candidates), scores, overall)

    def _load_segments(self, version_id: UUID) -> tuple[EvaluationSegment, ...]:
        rows = self.session.execute(
            select(TranslationSegment, Chunk)
            .join(Chunk, Chunk.id == TranslationSegment.chunk_id)
            .where(TranslationSegment.translation_version_id == version_id)
            .order_by(Chunk.ordinal)
        ).all()
        if not rows:
            raise ValueError("Translation version has no translated segments")
        return tuple(EvaluationSegment(
            chunk_id=chunk.id, stable_key=chunk.stable_key, source_text=chunk.source_text,
            target_text=segment.target_text, locator=dict(chunk.locator or {}),
        ) for segment, chunk in rows)

    def _scopes(self, level: int, segments: tuple[EvaluationSegment, ...], *,
                book: str | None, chapter: int | None, verse: int | None) -> tuple[ScopeResult, ...]:
        selected = [segment for segment in segments
                    if (book is None or str(segment.locator.get("book", "")).casefold() == book.casefold())
                    and (chapter is None or segment.locator.get("chapter") == chapter)
                    and (verse is None or segment.locator.get("verse") == verse)]
        groups: dict[tuple[str, ...], list[EvaluationSegment]] = {}
        for segment in selected:
            locator = segment.locator
            book_name = str(locator.get("book", "Unknown book"))
            if level == 1:
                key = (book_name, str(locator.get("chapter", "?")), str(locator.get("verse", "?")))
            elif level == 2:
                key = (book_name, str(locator.get("chapter", "?")))
            elif level == 3:
                key = (book_name,)
            else:
                key = ("document",)
            groups.setdefault(key, []).append(segment)
        result = []
        for key, members in groups.items():
            if level == 1:
                scope_key = "verse:" + ":".join(key)
                label = f"{key[0]} {key[1]}:{key[2]}"
            elif level == 2:
                scope_key, label = "chapter:" + ":".join(key), f"{key[0]} {key[1]}"
            elif level == 3:
                scope_key, label = "book:" + key[0], key[0]
            else:
                scope_key, label = "global:" + ":".join(key), "whole document"
            result.append(ScopeResult(scope_key, label, len(members), tuple(members)))
        return tuple(result)

    def _plugin(self, plugin_id: str, glossary_terms: tuple[dict[str, str], ...],
                project_usage: tuple[dict[str, str], ...]) -> EvaluationPlugin:
        from ai_translator.evaluation.plugins import ConsistencyEvaluator, FidelityEvaluator, ReadabilityEvaluator
        mapping: dict[str, EvaluationPlugin] = {
            "fidelity": FidelityEvaluator(),
            "readability": ReadabilityEvaluator(),
            "consistency": ConsistencyEvaluator(),
            "scripture.author_voice": AuthorVoiceEvaluator(self.metadata),
            "scripture.doctrinal_consistency": DoctrinalConsistencyEvaluator(self.doctrine_glossary, self.metadata),
            "scripture.glossary_compliance": GlossaryComplianceEvaluator(glossary_terms),
            "scripture.terminology_drift": TerminologyDriftEvaluator(glossary_terms),
            "scripture.chapter_flow": ChapterFlowEvaluator(),
            "scripture.context_preservation": ContextPreservationEvaluator(),
            "scripture.pronoun_resolution": PronounResolutionEvaluator(),
            "scripture.repeated_phrase_consistency": RepeatedPhraseEvaluator(),
            "scripture.project_terminology_usage": ProjectTerminologyUsageEvaluator(project_usage),
        }
        return mapping[plugin_id]

    def _glossary_terms(self, translation: Translation) -> tuple[dict[str, str], ...]:
        entries = GlossaryService(self.session).entries_for_scope(
            translation.document.project_id, translation.document.source_locale, translation.target_locale,
        )
        return tuple({"source_term": entry.source_term, "preferred_translation": entry.target_term,
                      "target_locale": translation.target_locale, "notes": entry.notes or "",
                      "glossary": entry.glossary.name} for entry in entries)

    def _project_usage(self, translation: Translation,
                       terms: tuple[dict[str, str], ...]) -> tuple[dict[str, str], ...]:
        project_id = translation.document.project_id
        latest_completed = (
            select(
                TranslationVersion.translation_id.label("translation_id"),
                func.max(TranslationVersion.version_number).label("version_number"),
            )
            .where(TranslationVersion.status == "completed")
            .group_by(TranslationVersion.translation_id)
            .subquery()
        )
        rows = self.session.execute(
            select(Translation.document_id, Translation.target_locale, Chunk.stable_key,
                   Chunk.source_text, TranslationSegment.target_text)
            .join(TranslationVersion, TranslationVersion.translation_id == Translation.id)
            .join(latest_completed,
                  (latest_completed.c.translation_id == TranslationVersion.translation_id)
                  & (latest_completed.c.version_number == TranslationVersion.version_number))
            .join(TranslationSegment, TranslationSegment.translation_version_id == TranslationVersion.id)
            .join(Chunk, Chunk.id == TranslationSegment.chunk_id)
            .where(Translation.document.has(project_id=project_id),
                   Translation.target_locale == translation.target_locale)
            .order_by(Translation.created_at, TranslationVersion.version_number)
        ).all()
        result = []
        for term in terms:
            source_term = term["source_term"]
            pattern = re.compile(r"(?<!\w)" + re.escape(source_term) + r"(?!\w)", re.IGNORECASE)
            matches = [(row, len(pattern.findall(row.source_text))) for row in rows if pattern.search(row.source_text)]
            forms: dict[str, int] = {}
            for row, _count in matches:
                for form in (term["preferred_translation"],):
                    if form and re.search(r"(?<!\w)" + re.escape(form) + r"(?!\w)", row.target_text, re.IGNORECASE):
                        forms[form] = forms.get(form, 0) + 1
            result.append({
                "source_term": source_term,
                "preferred_translation": term["preferred_translation"],
                "source_occurrences": str(sum(count for _, count in matches)),
                "translated_segments": str(len(matches)),
                "preferred_form_segments": str(next(iter(forms.values()), 0)),
            })
        return tuple(result)

    def _run_candidate(self, *, run_id: UUID, level: int, scope: ScopeResult,
                       plugin: EvaluationPlugin, model: EvaluatorModel,
                       config: EvaluationConfig, evaluation_input: EvaluationInput) -> ReviewCandidate:
        batches = tuple(evaluation_input.segments[i:i + config.batch_size]
                        for i in range(0, len(evaluation_input.segments), config.batch_size))
        if not batches:
            raise ValueError("Review scope contains no translated segments")
        level_name = LEVEL_NAMES[level]
        root_name = f"hierarchical:{level_name}:{plugin.plugin_id}:{self._key_hash(scope.scope_key)}"
        root = self.session.scalar(select(Evaluation).where(
            Evaluation.translation_id == self.session.get(TranslationVersion, evaluation_input.translation_version_id).translation_id,
            Evaluation.evaluator_name == root_name,
        ).limit(1))
        if root is None:
            version_row = self.session.get(TranslationVersion, evaluation_input.translation_version_id)
            root = Evaluation(translation_id=version_row.translation_id, evaluator_name=root_name)
            self.session.add(root)
            self.session.flush()
        prompt_hash = hashlib.sha256(plugin.prompt_template.encode("utf-8")).hexdigest()
        prompt_key = f"hierarchical:{level_name}:{plugin.plugin_id}"
        prompt = self.session.scalar(select(PromptVersion).where(
            PromptVersion.prompt_key == prompt_key, PromptVersion.content_hash == prompt_hash,
        ))
        if prompt is None:
            latest_prompt = self.session.scalar(select(func.max(PromptVersion.version_number)).where(
                PromptVersion.prompt_key == prompt_key,
            )) or 0
            prompt = PromptVersion(prompt_key=prompt_key, version_number=latest_prompt + 1,
                                   template=plugin.prompt_template, content_hash=prompt_hash,
                                   metadata_json={"kind": "hierarchical_review", "level": level_name,
                                                  "plugin_id": plugin.plugin_id})
            self.session.add(prompt)
            self.session.flush()
        self.session.commit()

        options = {"temperature": config.temperature}
        if config.num_ctx is not None:
            options["num_ctx"] = config.num_ctx
        options.update(model.options)
        completed: list[tuple[tuple[EvaluationSegment, ...], EvaluationOutput, str, Generation]] = []
        failures: list[str] = []
        prompts_manifest: list[dict[str, object]] = []
        for batch in batches:
            batch_input = EvaluationInput(
                translation_version_id=evaluation_input.translation_version_id,
                translation_content_hash=evaluation_input.translation_content_hash,
                source_locale=evaluation_input.source_locale, target_locale=evaluation_input.target_locale,
                domain=evaluation_input.domain, task_kind=evaluation_input.task_kind,
                target_reading_level=evaluation_input.target_reading_level,
                segments=batch, total_segment_count=len(batch),
            )
            try:
                rendered = plugin.build_prompt(batch_input)
                rendered_hash = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
                prompts_manifest.append({"segment_keys": [item.stable_key for item in batch],
                                         "prompt_hash": rendered_hash,
                                         "source_hashes": [self._hash(item.source_text) for item in batch],
                                         "translation_hashes": [self._hash(item.target_text) for item in batch]})
                generation = self.client.generate(model=model.model_name, prompt=rendered,
                                                  options=options, format="json")
                output = plugin.parse_output(generation.response)
                completed.append((batch, output, rendered_hash, generation))
            except Exception as exc:
                failures.append(str(exc))
                break
        status = "completed" if len(completed) == len(batches) else "partial" if completed else "failed"
        merged = self._merge(completed) if completed else None
        digest = self._model_digest(model.model_name)
        runtime = {"base_url": self.client.base_url, "endpoint": "/api/generate", "format": "json",
                   "model_alias": model.alias, "review_run_id": str(run_id), "error": failures[0] if failures else None,
                   "model_digest_available": digest is not None,
                   "responses": [{"model": response.model, "created_at": response.created_at,
                                  "total_duration": response.total_duration}
                                 for _, _, _, response in completed]}
        model_run = ModelRun(provider="ollama", model_name=model.model_name, model_digest=digest, role="evaluator",
                             prompt_hash=prompt_hash, options=options, runtime_manifest=runtime)
        self.session.add(model_run)
        self.session.flush()
        latest = self.session.scalar(select(func.max(EvaluationVersion.version_number)).where(
            EvaluationVersion.evaluation_id == root.id,
        )) or 0
        evaluation_version = EvaluationVersion(
            evaluation_id=root.id, translation_version_id=evaluation_input.translation_version_id,
            model_run_id=model_run.id, version_number=latest + 1, status=status,
            rubric_version=plugin.version,
            plugin_manifest={"plugin_id": plugin.plugin_id, "plugin_version": plugin.version,
                            "display_name": plugin.display_name, "prompt_version_id": str(prompt.id),
                            "prompt_hash": prompt_hash},
            run_manifest={"hierarchical_review": True, "level": level, "level_name": level_name,
                          "scope_key": scope.scope_key, "scope_label": scope.label,
                          "input_segment_count": len(evaluation_input.segments), "batch_size": config.batch_size,
                          "review_run_id": str(run_id), "batches": prompts_manifest,
                          "error": failures[0] if failures else None},
        )
        self.session.add(evaluation_version)
        self.session.flush()
        if merged:
            evidence = {"batch_results": [{"segment_keys": [item.stable_key for item in batch],
                                            "result": output.to_dict(), "raw_response": generation.response,
                                            "rendered_prompt_hash": digest}
                                           for batch, output, digest, generation in completed],
                        "scope_key": scope.scope_key, "review_run_id": str(run_id)}
            searchable = "\n".join([merged.reasoning, *merged.detected_issues, *merged.recommendations])
            self.session.add(EvaluationResult(
                evaluation_version_id=evaluation_version.id, plugin_id=plugin.plugin_id,
                metric_key="overall", scope_key=scope.scope_key, score=merged.score,
                confidence=merged.confidence, reasoning=merged.reasoning,
                recommendations=list(merged.recommendations), detected_issues=list(merged.detected_issues),
                searchable_text=searchable, value=merged.score, categorical_value=level_name,
                finding=merged.reasoning, evidence=evidence,
            ))
        self.session.commit()
        return ReviewCandidate(level, level_name, scope.scope_key, scope.label, scope.segment_count, plugin.plugin_id,
                               model.alias, model.model_name, evaluation_version.id,
                               evaluation_version.version_number, status,
                               merged.score if merged else None, merged.confidence if merged else None,
                               len(merged.detected_issues) if merged else 0,
                               failures[0] if failures else None)

    @staticmethod
    def _merge(outputs):
        total = sum(len(batch) for batch, _, _, _ in outputs)
        score = round(sum(result.score * len(batch) for batch, result, _, _ in outputs) / total, 2)
        confidence = round(sum(result.confidence * len(batch) for batch, result, _, _ in outputs) / total, 4)
        reason = []
        issues: list[str] = []
        recommendations: list[str] = []
        for batch, result, _, _ in outputs:
            reason.append(f"[{batch[0].stable_key}–{batch[-1].stable_key}] {result.reasoning}")
            for values, target in ((result.detected_issues, issues), (result.recommendations, recommendations)):
                known = {value.casefold() for value in target}
                for value in values:
                    if value.casefold() not in known:
                        target.append(value)
                        known.add(value.casefold())
        return EvaluationOutput(score, confidence, "\n".join(reason), tuple(recommendations), tuple(issues))

    @staticmethod
    def _weighted_score(results: list[ReviewCandidate]) -> float | None:
        valid = [item for item in results if item.score is not None and item.status == "completed"]
        denominator = sum(item.segment_count for item in valid)
        return (round(sum(item.score * item.segment_count for item in valid) / denominator, 2)
                if denominator else None)

    @staticmethod
    def _key_hash(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]

    @staticmethod
    def _hash(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def _model_digest(self, model_name: str) -> str | None:
        if self._model_digests is None:
            self._model_digests = {}
            try:
                for item in self.client.list_models():
                    name, digest = item.get("name"), item.get("digest")
                    if isinstance(name, str) and isinstance(digest, str):
                        self._model_digests[name] = digest
            except OllamaError:
                pass
        return self._model_digests.get(model_name)
