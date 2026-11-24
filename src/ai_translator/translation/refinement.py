from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..db.models import (
    Chunk,
    Evaluation,
    EvaluationResult,
    EvaluationVersion,
    ModelRun,
    PromptVersion,
    Translation,
    TranslationSegment,
    TranslationVersion,
)
from ..glossary.errors import GlossaryConflictError
from ..glossary.service import GlossaryService
from ..translation_memory.service import TranslationMemoryConflictError, TranslationMemoryService
from .config import ModelConfig, TranslationConfig
from .errors import OllamaError, TranslationError
from .ollama import Generation, OllamaClient
from .prompts import PromptTemplateManager, sha256_text


@dataclass(frozen=True, slots=True)
class RefinementCandidateResult:
    model_alias: str
    model_name: str
    version_id: UUID
    version_number: int
    status: str
    translated_chunks: int
    total_chunks: int
    operation: str
    error: str | None = None


@dataclass(frozen=True, slots=True)
class _InputSegment:
    chunk: Chunk
    source_text: str
    source_hash: str
    targets: dict[UUID, str]


class TranslationRefinementService:
    """Create immutable feedback-guided revisions and multi-candidate syntheses."""

    PROMPTS = {
        "refinement": "refinement.txt",
        "synthesis": "synthesis.txt",
    }

    def __init__(self, session: Session, client: OllamaClient,
                 prompt_manager: PromptTemplateManager | None = None):
        self.session = session
        self.client = client
        self.prompt_manager = prompt_manager or PromptTemplateManager()

    def refine_version(
        self,
        *,
        translation_version_id: UUID,
        config: TranslationConfig,
        model_selection: list[str] | tuple[str, ...] | None = None,
        progress_callback: Callable[[str], None] | None = None,
    ) -> list[RefinementCandidateResult]:
        """Revise one candidate using its own saved evaluator results."""
        version = self._require_version(translation_version_id)
        translation = version.translation
        segments = self._load_segments([version])
        feedback, eval_version_ids = self._load_feedback([version])
        self._require_complete_feedback([version], segments, feedback)
        models = config.resolve_models(model_selection)
        return self._generate(
            operation="refinement",
            versions=[version],
            translation=translation,
            segments=segments,
            feedback=feedback,
            evaluation_version_ids=eval_version_ids,
            models=models,
            config=config,
            progress_callback=progress_callback,
        )

    def synthesize_versions(
        self,
        *,
        translation_version_ids: list[UUID] | tuple[UUID, ...],
        config: TranslationConfig,
        model_selection: list[str] | tuple[str, ...] | None = None,
        progress_callback: Callable[[str], None] | None = None,
    ) -> list[RefinementCandidateResult]:
        """Create source-grounded ensemble candidates from at least two versions."""
        unique_ids = list(dict.fromkeys(translation_version_ids))
        if len(unique_ids) < 2:
            raise TranslationError("Synthesis requires at least two distinct translation version IDs")
        versions = [self._require_version(version_id) for version_id in unique_ids]
        translation_id = versions[0].translation_id
        if any(version.translation_id != translation_id for version in versions):
            raise TranslationError("All synthesis candidates must belong to the same translation task")
        translation = versions[0].translation
        segments = self._load_segments(versions)
        feedback, eval_version_ids = self._load_feedback(versions)
        self._require_complete_feedback(versions, segments, feedback)
        models = config.resolve_models(model_selection)
        return self._generate(
            operation="synthesis",
            versions=versions,
            translation=translation,
            segments=segments,
            feedback=feedback,
            evaluation_version_ids=eval_version_ids,
            models=models,
            config=config,
            progress_callback=progress_callback,
        )

    def _generate(
        self,
        *,
        operation: str,
        versions: list[TranslationVersion],
        translation: Translation,
        segments: list[_InputSegment],
        feedback: dict[tuple[UUID, str], list[dict[str, Any]]],
        evaluation_version_ids: dict[UUID, list[str]],
        models: tuple[ModelConfig, ...],
        config: TranslationConfig,
        progress_callback: Callable[[str], None] | None,
    ) -> list[RefinementCandidateResult]:
        document = translation.document
        prompt_text, prompt_key = self._load_prompt(operation, document.domain)
        prompt_version = self.prompt_manager.get_or_create_version(self.session, prompt_key, prompt_text)
        prompt_version_id = prompt_version.id
        prompt_hash = prompt_version.content_hash
        contexts = self._build_contexts(translation, segments)
        # Save the prompt snapshot and release the database transaction before
        # waiting on Ollama. Every output is then committed independently.
        self.session.commit()

        digests: dict[str, str] = {}
        self._report(progress_callback, "Checking Ollama model metadata...")
        try:
            for item in self.client.list_models():
                name, digest = item.get("name"), item.get("digest")
                if isinstance(name, str) and isinstance(digest, str):
                    digests[name] = digest
        except OllamaError:
            pass

        self._report(progress_callback,
                     f"Starting {operation} for {len(segments)} chunks using "
                     f"{len(versions)} input version(s) and {len(models)} model(s).")
        outputs: list[RefinementCandidateResult] = []
        for model_index, model in enumerate(models, start=1):
            outputs.append(self._generate_model(
                operation=operation,
                model=model,
                model_index=model_index,
                model_count=len(models),
                versions=versions,
                translation=translation,
                segments=segments,
                feedback=feedback,
                evaluation_version_ids=evaluation_version_ids,
                contexts=contexts,
                config=config,
                prompt_text=prompt_text,
                prompt_key=prompt_key,
                prompt_version_id=prompt_version_id,
                prompt_hash=prompt_hash,
                digest=digests.get(model.model_name),
                progress_callback=progress_callback,
            ))
        return outputs

    def _generate_model(
        self,
        *,
        operation: str,
        model: ModelConfig,
        model_index: int,
        model_count: int,
        versions: list[TranslationVersion],
        translation: Translation,
        segments: list[_InputSegment],
        feedback: dict[tuple[UUID, str], list[dict[str, Any]]],
        evaluation_version_ids: dict[UUID, list[str]],
        contexts: dict[UUID, dict[str, Any]],
        config: TranslationConfig,
        prompt_text: str,
        prompt_key: str,
        prompt_version_id: UUID,
        prompt_hash: str,
        digest: str | None,
        progress_callback: Callable[[str], None] | None,
    ) -> RefinementCandidateResult:
        options: dict[str, Any] = {"temperature": config.temperature}
        if config.num_ctx is not None:
            options["num_ctx"] = config.num_ctx
        if config.num_predict is not None:
            options["num_predict"] = config.num_predict
        options.update(model.options)
        generated: list[tuple[_InputSegment, Generation, str, dict[str, Any]]] = []
        attempted_prompts: list[dict[str, Any]] = []
        responses: list[dict[str, Any]] = []
        error: str | None = None
        self._report(progress_callback,
                     f"[{model_index}/{model_count}] {model.alias}: {operation} over "
                     f"{len(segments)} chunks.")

        for index, segment in enumerate(segments, start=1):
            chunk = segment.chunk
            feedback_items = self._feedback_for_segment(segment, versions, feedback)
            candidate_texts = [
                {"translation_version_id": str(version.id),
                 "model": (version.run_manifest or {}).get("model_name", version.author_kind),
                 "text": segment.targets[version.id]}
                for version in versions if version.id in segment.targets
            ]
            prompt_values = {
                "source_locale": translation.document.source_locale,
                "target_locale": translation.target_locale,
                "task_kind": translation.task_kind,
                "reading_level": translation.target_reading_level or "not specified",
                "domain": translation.document.domain,
                "locator": json.dumps(chunk.locator or {}, ensure_ascii=False, sort_keys=True),
                "source_text": segment.source_text,
                "candidate_text": candidate_texts[0]["text"] if candidate_texts else "",
                "candidate_translations": json.dumps(candidate_texts, ensure_ascii=False, sort_keys=True),
                "evaluation_feedback": json.dumps(feedback_items, ensure_ascii=False, sort_keys=True),
                "glossary_terms": json.dumps(contexts[chunk.id]["glossary"], ensure_ascii=False, sort_keys=True),
                "translation_memory": json.dumps(contexts[chunk.id]["memories"], ensure_ascii=False, sort_keys=True),
            }
            try:
                prompt = self.prompt_manager.render(prompt_text, prompt_values)
            except Exception as exc:
                error = str(exc)
                break
            prompt_hash_for_chunk = sha256_text(prompt)
            audit_item = {
                "chunk_id": str(chunk.id),
                "stable_key": chunk.stable_key,
                "source_hash": segment.source_hash,
                "input_target_hashes": {
                    str(version_id): sha256_text(target)
                    for version_id, target in segment.targets.items()
                },
                "evaluation_result_ids": sorted({
                    result_id for item in feedback_items for result_id in item["evaluation_result_ids"]
                }),
                "rendered_prompt_hash": prompt_hash_for_chunk,
                "glossary_snapshot": contexts[chunk.id]["glossary"],
                "translation_memory_snapshot": contexts[chunk.id]["memories"],
            }
            attempted_prompts.append(audit_item)
            self._report(progress_callback,
                         f"[{model_index}/{model_count}] {model.alias} chunk "
                         f"{index}/{len(segments)} ({chunk.stable_key}): waiting for Ollama...")
            started = time.perf_counter()
            try:
                generation = self.client.generate(
                    model=model.model_name,
                    prompt=prompt,
                    options=options,
                )
            except OllamaError as exc:
                error = str(exc)
                self._report(progress_callback,
                             f"[{model_index}/{model_count}] {model.alias} chunk "
                             f"{index}/{len(segments)} failed: {error}")
                break
            generated.append((segment, generation, prompt_hash_for_chunk, audit_item))
            responses.append({
                "chunk_id": str(chunk.id),
                "model_returned": generation.model,
                "created_at": generation.created_at,
                "total_duration": generation.total_duration,
                "prompt_eval_count": generation.prompt_eval_count,
                "eval_count": generation.eval_count,
            })
            self._report(progress_callback,
                         f"[{model_index}/{model_count}] {model.alias} chunk "
                         f"{index}/{len(segments)} completed in {time.perf_counter() - started:.1f}s.")

        status = "completed" if len(generated) == len(segments) else ("partial" if generated else "failed")
        translation_id = translation.id
        current_max = self.session.scalar(select(func.max(TranslationVersion.version_number)).where(
            TranslationVersion.translation_id == translation_id
        )) or 0
        all_input_ids = [str(version.id) for version in versions]
        primary_parent_id = versions[0].id
        model_run = ModelRun(
            provider="ollama",
            model_name=model.model_name,
            model_digest=digest,
            role=operation,
            prompt_hash=prompt_hash,
            options=options,
            runtime_manifest={
                "operation": operation,
                "base_url": self.client.base_url,
                "endpoint": "/api/generate",
                "stream": False,
                "model_alias": model.alias,
                "model_digest_available": digest is not None,
                "responses": responses,
                "error": error,
            },
        )
        self.session.add(model_run)
        self.session.flush()
        content_payload = [{
            "stable_key": segment.chunk.stable_key,
            "target_text": generation.response,
        } for segment, generation, _prompt_hash, _audit in generated]
        content_hash = hashlib.sha256(json.dumps(
            content_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")).hexdigest() if generated else None
        version = TranslationVersion(
            translation_id=translation_id,
            version_number=current_max + 1,
            parent_version_id=primary_parent_id,
            model_run_id=model_run.id,
            author_kind="model",
            status=status,
            content_hash=content_hash,
            run_manifest={
                "operation": operation,
                "operation_version": "1.0",
                "model_alias": model.alias,
                "model_name": model.model_name,
                "prompt_key": prompt_key,
                "prompt_version_id": str(prompt_version_id),
                "prompt_template_hash": prompt_hash,
                "primary_parent_translation_version_id": str(primary_parent_id),
                "input_translation_version_ids": all_input_ids,
                "evaluation_version_ids": sorted({
                    item for values in evaluation_version_ids.values() for item in values
                }),
                "source_locale": translation.document.source_locale,
                "target_locale": translation.target_locale,
                "task_kind": translation.task_kind,
                "target_reading_level": translation.target_reading_level,
                "domain": translation.document.domain,
                "input_chunk_count": len(segments),
                "completed_chunk_count": len(generated),
                "complete": len(generated) == len(segments),
                "chunks": attempted_prompts,
                "error": error,
            },
        )
        self.session.add(version)
        self.session.flush()
        glossary_usage: dict[UUID, set[str]] = defaultdict(set)
        memory_usage: dict[UUID, set[str]] = defaultdict(set)
        for segment, generation, _prompt_hash, _audit in generated:
            self.session.add(TranslationSegment(
                translation_version_id=version.id,
                chunk_id=segment.chunk.id,
                target_text=generation.response,
                target_hash=sha256_text(generation.response),
            ))
            reference = f"{translation.document_id}:{segment.chunk.stable_key}"
            for item in contexts[segment.chunk.id]["glossary"]:
                glossary_usage[UUID(str(item["id"]))].add(reference)
            for item in contexts[segment.chunk.id]["memories"]:
                memory_usage[UUID(str(item["id"]))].add(reference)
        GlossaryService(self.session).record_usage(glossary_usage)
        TranslationMemoryService(self.session).record_usage(memory_usage)
        self.session.commit()
        self._report(progress_callback,
                     f"[{model_index}/{model_count}] {model.alias}: saved {operation} version "
                     f"{version.version_number} ({status}, {len(generated)}/{len(segments)} chunks).")
        return RefinementCandidateResult(
            model_alias=model.alias,
            model_name=model.model_name,
            version_id=version.id,
            version_number=version.version_number,
            status=status,
            translated_chunks=len(generated),
            total_chunks=len(segments),
            operation=operation,
            error=error,
        )

    def _load_segments(self, versions: list[TranslationVersion]) -> list[_InputSegment]:
        version_ids = [version.id for version in versions]
        rows = self.session.execute(
            select(TranslationSegment, Chunk)
            .join(Chunk, Chunk.id == TranslationSegment.chunk_id)
            .where(TranslationSegment.translation_version_id.in_(version_ids))
            .order_by(Chunk.ordinal, TranslationSegment.translation_version_id)
        ).all()
        if not rows:
            raise TranslationError("Input translation version(s) have no translated segments")
        by_chunk: dict[UUID, _InputSegment] = {}
        for segment, chunk in rows:
            current = by_chunk.get(chunk.id)
            if current is None:
                current = _InputSegment(chunk, chunk.source_text, chunk.source_hash, {})
                by_chunk[chunk.id] = current
            elif current.source_hash != chunk.source_hash or current.source_text != chunk.source_text:
                raise TranslationError(f"Source mismatch for chunk {chunk.stable_key}")
            current.targets[segment.translation_version_id] = segment.target_text
        if len(versions) > 1:
            expected = set(by_chunk)
            for version in versions:
                version_chunks = {segment.chunk_id for segment in version.segments}
                if version_chunks != expected:
                    raise TranslationError(
                        "Synthesis inputs must contain exactly the same chunks; use candidates from the same scope"
                    )
        return sorted(by_chunk.values(), key=lambda item: item.chunk.ordinal)

    def _load_feedback(self, versions: list[TranslationVersion]):
        version_ids = [version.id for version in versions]
        rows = self.session.execute(
            select(EvaluationVersion, EvaluationResult, Evaluation, ModelRun)
            .join(EvaluationResult, EvaluationResult.evaluation_version_id == EvaluationVersion.id)
            .join(Evaluation, Evaluation.id == EvaluationVersion.evaluation_id)
            .outerjoin(ModelRun, ModelRun.id == EvaluationVersion.model_run_id)
            .where(
                EvaluationVersion.translation_version_id.in_(version_ids),
                EvaluationVersion.status.in_(("completed", "partial")),
                EvaluationResult.metric_key == "overall",
            )
        ).all()
        # Keep the most recent saved run for each candidate/category/model so
        # rerunning an evaluation doesn't inject stale feedback on later passes.
        latest: dict[tuple[UUID, str, str], tuple] = {}
        for evaluation_version, result, evaluation, model_run in rows:
            run_manifest = evaluation_version.run_manifest or {}
            model_name = (model_run.model_name if model_run else
                          run_manifest.get("evaluator_model", run_manifest.get("evaluator_alias", "unknown")))
            key = (evaluation_version.translation_version_id, evaluation.evaluator_name or result.plugin_id,
                   str(model_name))
            previous = latest.get(key)
            if previous is None or (evaluation_version.created_at, str(evaluation_version.id)) > (
                previous[0].created_at, str(previous[0].id)
            ):
                latest[key] = (evaluation_version, result, evaluation, model_run)
        feedback: dict[tuple[UUID, str], list[dict[str, Any]]] = defaultdict(list)
        evaluation_version_ids: dict[UUID, list[str]] = defaultdict(list)
        for evaluation_version, result, evaluation, model_run in latest.values():
            translation_version_id = evaluation_version.translation_version_id
            model_name = model_run.model_name if model_run else "unknown"
            model_alias = (evaluation_version.run_manifest or {}).get("evaluator_alias", model_name)
            evaluation_version_ids[translation_version_id].append(str(evaluation_version.id))
            evidence = result.evidence or {}
            batch_results = evidence.get("batch_results")
            if isinstance(batch_results, list) and batch_results:
                for batch in batch_results:
                    if not isinstance(batch, dict):
                        continue
                    keys = batch.get("segment_keys", [])
                    output = batch.get("output", {})
                    if not isinstance(output, dict):
                        continue
                    item = self._feedback_record(
                        evaluation_version, result, evaluation, model_alias, model_name, output,
                    )
                    for stable_key in keys if isinstance(keys, list) else []:
                        feedback[(translation_version_id, str(stable_key))].append(item)
            else:
                item = self._feedback_record(
                    evaluation_version, result, evaluation, model_alias, model_name, {
                        "score": result.score,
                        "confidence": result.confidence,
                        "reasoning": result.reasoning,
                        "recommendations": result.recommendations,
                        "detected_issues": result.detected_issues,
                    },
                )
                keys = evidence.get("segment_keys")
                for version in versions:
                    if version.id != translation_version_id:
                        continue
                    known_keys = {chunk.stable_key for _segment, chunk in self._segments_for(version.id)}
                    selected_keys = keys if isinstance(keys, list) else sorted(known_keys)
                    for stable_key in selected_keys:
                        feedback[(translation_version_id, str(stable_key))].append(item)
        if not latest:
            raise TranslationError(
                "No completed evaluation feedback found for the input version(s); run evaluation first"
            )
        for values in feedback.values():
            values.sort(key=lambda item: (item["category"], item["evaluator_model"]))
        return feedback, evaluation_version_ids

    @staticmethod
    def _feedback_record(evaluation_version, result, evaluation, model_alias, model_name, output):
        return {
            "evaluation_version_id": str(evaluation_version.id),
            "translation_version_id": str(evaluation_version.translation_version_id),
            "evaluation_result_ids": [str(result.id)],
            "category": evaluation.evaluator_name or result.plugin_id,
            "evaluator_model": model_name,
            "evaluator_alias": model_alias,
            "score": output.get("score", result.score),
            "confidence": output.get("confidence", result.confidence),
            "reasoning": output.get("reasoning", result.reasoning),
            "recommendations": output.get("recommendations", result.recommendations or []),
            "detected_issues": output.get("detected_issues", result.detected_issues or []),
        }

    @staticmethod
    def _require_complete_feedback(versions, segments, feedback) -> None:
        missing = [
            f"{version.id}:{segment.chunk.stable_key}"
            for version in versions for segment in segments
            if not feedback.get((version.id, segment.chunk.stable_key))
        ]
        if missing:
            preview = ", ".join(missing[:5])
            suffix = "…" if len(missing) > 5 else ""
            raise TranslationError(
                "Saved evaluation feedback is missing for input segment(s): " + preview + suffix
            )

    def _feedback_for_segment(self, segment, versions, feedback):
        result: list[dict[str, Any]] = []
        for version in versions:
            result.extend(feedback.get((version.id, segment.chunk.stable_key), []))
        return result

    def _build_contexts(self, translation: Translation, segments: list[_InputSegment]):
        document = translation.document
        glossary_service = GlossaryService(self.session)
        memory_service = TranslationMemoryService(self.session)
        glossary_entries = glossary_service.entries_for_scope(
            document.project_id, document.source_locale, translation.target_locale,
        )
        memory_entries = memory_service.list_for_project(
            document.project_id, source_locale=document.source_locale,
            target_locale=translation.target_locale,
        )
        contexts = {}
        for segment in segments:
            try:
                glossary = glossary_service.resolve_for_text(
                    project_id=document.project_id,
                    source_locale=document.source_locale,
                    target_locale=translation.target_locale,
                    source_text=segment.source_text,
                    entries=glossary_entries,
                )
                memories = memory_service.lookup(
                    project_id=document.project_id,
                    source_locale=document.source_locale,
                    target_locale=translation.target_locale,
                    source_phrase=segment.source_text,
                    limit=3,
                    entries=memory_entries,
                )
            except (GlossaryConflictError, TranslationMemoryConflictError) as exc:
                raise TranslationError(str(exc)) from exc
            contexts[segment.chunk.id] = {
                "glossary": glossary,
                "memories": [{
                    "id": str(match.entry.id),
                    "source": match.entry.source_text,
                    "target": match.entry.target_text,
                    "similarity": match.similarity,
                } for match in memories],
            }
        return contexts

    def _load_prompt(self, operation: str, domain: str) -> tuple[str, str]:
        filename = self.PROMPTS[operation]
        prompt_path = Path(__file__).resolve().parents[1] / "resources" / "prompts" / filename
        try:
            return prompt_path.read_text(encoding="utf-8"), f"{operation}:{domain}:v1"
        except OSError as exc:
            raise TranslationError(f"Cannot read {operation} prompt template {prompt_path}: {exc}") from exc

    def _require_version(self, version_id: UUID) -> TranslationVersion:
        version = self.session.get(TranslationVersion, version_id)
        if version is None:
            raise TranslationError(f"Translation version {version_id} does not exist")
        return version

    def _segments_for(self, version_id: UUID):
        return self.session.execute(
            select(TranslationSegment, Chunk)
            .join(Chunk, Chunk.id == TranslationSegment.chunk_id)
            .where(TranslationSegment.translation_version_id == version_id)
            .order_by(Chunk.ordinal)
        ).all()

    @staticmethod
    def _report(callback: Callable[[str], None] | None, message: str) -> None:
        if callback is not None:
            callback(message)
