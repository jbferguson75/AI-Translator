from __future__ import annotations

import hashlib
import json
import time
from typing import Callable
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
from ..translation.errors import OllamaError
from ..translation.ollama import Generation, OllamaClient
from .config import EvaluationConfig, EvaluatorModel
from .errors import EvaluationConfigurationError, EvaluationError, EvaluationOutputError
from .plugins import EvaluationPlugin
from .registry import EvaluatorRegistry, default_registry
from .types import (
    EvaluationCandidateResult,
    EvaluationInput,
    EvaluationOutput,
    EvaluationSegment,
)


class EvaluationService:
    """Evaluate one immutable translation version with registered plugins/models."""

    def __init__(self, session: Session, client: OllamaClient,
                 registry: EvaluatorRegistry | None = None):
        self.session = session
        self.client = client
        self.registry = registry or default_registry()

    def evaluate_translation_version(
        self,
        *,
        translation_version_id: UUID,
        config: EvaluationConfig,
        plugin_selection: list[str] | tuple[str, ...] | None = None,
        model_selection: list[str] | tuple[str, ...] | None = None,
        chunk_limit: int | None = None,
        progress_callback: Callable[[str], None] | None = None,
    ) -> list[EvaluationCandidateResult]:
        if chunk_limit is not None and chunk_limit < 1:
            raise EvaluationConfigurationError("Chunk limit must be a positive integer")
        plugins = tuple(self.registry.get(plugin_id) for plugin_id in
                        (plugin_selection if plugin_selection is not None else config.plugin_ids))
        if not plugins:
            raise EvaluationConfigurationError("Select at least one evaluator plugin")
        if len({plugin.plugin_id for plugin in plugins}) != len(plugins):
            raise EvaluationConfigurationError("Evaluator plugin selection contains duplicates")
        models = config.resolve_models(model_selection)
        evaluation_input, translation_id = self._load_input(translation_version_id, chunk_limit)
        self._report(progress_callback,
                     f"Evaluating translation version {evaluation_input.translation_version_id}: "
                     f"{len(evaluation_input.segments)} of {evaluation_input.total_segment_count} chunks.")
        self._report(progress_callback, "Plugins: " + ", ".join(
            f"{plugin.plugin_id} v{plugin.version}" for plugin in plugins
        ))
        self._report(progress_callback, "Models: " + ", ".join(
            f"{model.alias} ({model.model_name})" for model in models
        ))

        roots: dict[str, UUID] = {}
        prompts: dict[str, tuple[UUID, str]] = {}
        for plugin in plugins:
            root = self.session.scalar(select(Evaluation).where(
                Evaluation.translation_id == translation_id,
                Evaluation.evaluator_name == plugin.plugin_id,
            ).order_by(Evaluation.created_at).limit(1))
            if root is None:
                root = Evaluation(translation_id=translation_id, evaluator_name=plugin.plugin_id)
                self.session.add(root)
                self.session.flush()
            roots[plugin.plugin_id] = root.id
            prompt_hash = hashlib.sha256(plugin.prompt_template.encode("utf-8")).hexdigest()
            prompt_key = f"evaluator:{plugin.plugin_id}"
            prompt_version = self._get_or_create_prompt(prompt_key, plugin.prompt_template, prompt_hash)
            prompts[plugin.plugin_id] = (prompt_version.id, prompt_hash)

        # Persist roots and prompt snapshots before inference; no write transaction
        # remains open while waiting for Ollama.
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

        results: list[EvaluationCandidateResult] = []
        candidate_count = len(plugins) * len(models)
        candidate_index = 0
        for plugin in plugins:
            prompt_version_id, prompt_hash = prompts[plugin.plugin_id]
            for model in models:
                candidate_index += 1
                results.append(self._evaluate_candidate(
                    plugin=plugin,
                    model=model,
                    evaluation_input=evaluation_input,
                    evaluation_id=roots[plugin.plugin_id],
                    prompt_version_id=prompt_version_id,
                    prompt_hash=prompt_hash,
                    config=config,
                    digest=digests.get(model.model_name),
                    candidate_index=candidate_index,
                    candidate_count=candidate_count,
                    progress_callback=progress_callback,
                ))
        return results

    def _load_input(self, version_id: UUID, chunk_limit: int | None) -> tuple[EvaluationInput, UUID]:
        version = self.session.get(TranslationVersion, version_id)
        if version is None:
            raise EvaluationConfigurationError(f"Translation version {version_id} does not exist")
        translation: Translation = version.translation
        document = translation.document
        rows = self.session.execute(
            select(TranslationSegment, Chunk)
            .join(Chunk, Chunk.id == TranslationSegment.chunk_id)
            .where(TranslationSegment.translation_version_id == version_id)
            .order_by(Chunk.ordinal)
        ).all()
        if not rows:
            raise EvaluationConfigurationError("Translation version has no translated segments")
        total_segment_count = len(rows)
        if chunk_limit is not None:
            rows = rows[:chunk_limit]
        segments = tuple(EvaluationSegment(
            chunk_id=chunk.id,
            stable_key=chunk.stable_key,
            source_text=chunk.source_text,
            target_text=segment.target_text,
            locator=dict(chunk.locator or {}),
        ) for segment, chunk in rows)
        return EvaluationInput(
            translation_version_id=version.id,
            translation_content_hash=version.content_hash,
            source_locale=document.source_locale,
            target_locale=translation.target_locale,
            domain=document.domain,
            task_kind=translation.task_kind,
            target_reading_level=translation.target_reading_level,
            segments=segments,
            total_segment_count=total_segment_count,
        ), translation.id

    def _get_or_create_prompt(self, prompt_key: str, template: str, digest: str) -> PromptVersion:
        prompt = self.session.scalar(select(PromptVersion).where(
            PromptVersion.prompt_key == prompt_key,
            PromptVersion.content_hash == digest,
        ))
        if prompt is not None:
            return prompt
        latest = self.session.scalar(select(func.max(PromptVersion.version_number)).where(
            PromptVersion.prompt_key == prompt_key
        )) or 0
        prompt = PromptVersion(
            prompt_key=prompt_key,
            version_number=latest + 1,
            template=template,
            content_hash=digest,
            metadata_json={"kind": "evaluation_plugin_prompt", "encoding": "utf-8"},
        )
        self.session.add(prompt)
        self.session.flush()
        return prompt

    def _evaluate_candidate(
        self, *, plugin: EvaluationPlugin, model: EvaluatorModel,
        evaluation_input: EvaluationInput, evaluation_id: UUID,
        prompt_version_id: UUID, prompt_hash: str, config: EvaluationConfig,
        digest: str | None,
        candidate_index: int, candidate_count: int,
        progress_callback: Callable[[str], None] | None,
    ) -> EvaluationCandidateResult:
        batches = tuple(
            evaluation_input.segments[start:start + config.batch_size]
            for start in range(0, len(evaluation_input.segments), config.batch_size)
        )
        options: dict[str, object] = {"temperature": config.temperature}
        if config.num_ctx is not None:
            options["num_ctx"] = config.num_ctx
        options.update(model.options)
        outputs: list[tuple[tuple[EvaluationSegment, ...], EvaluationOutput, str, Generation]] = []
        prompt_manifest: list[dict[str, object]] = []
        error: str | None = None
        self._report(progress_callback,
                     f"[{candidate_index}/{candidate_count}] {plugin.plugin_id} / {model.alias}: "
                     f"evaluating {len(evaluation_input.segments)} chunks in {len(batches)} batch(es).")
        for batch_index, batch in enumerate(batches, start=1):
            batch_input = evaluation_input.for_segments(batch)
            batch_keys = [segment.stable_key for segment in batch]
            batch_label = f"{batch_keys[0]}–{batch_keys[-1]}"
            self._report(progress_callback,
                         f"[{candidate_index}/{candidate_count}] {plugin.plugin_id} / {model.alias} "
                         f"batch {batch_index}/{len(batches)} ({batch_label}): preparing request.")
            started = time.perf_counter()
            try:
                prompt = plugin.build_prompt(batch_input)
                prompt_digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                prompt_manifest.append({
                    "segment_keys": batch_keys,
                    "source_hashes": [self._hash_text(segment.source_text) for segment in batch],
                    "translation_hashes": [self._hash_text(segment.target_text) for segment in batch],
                    "prompt_hash": prompt_digest,
                })
                self._report(progress_callback,
                             f"[{candidate_index}/{candidate_count}] {plugin.plugin_id} / {model.alias} "
                             f"batch {batch_index}/{len(batches)}: waiting for Ollama...")
                generation = self.client.generate(
                    model=model.model_name,
                    prompt=prompt,
                    options=options,
                    format="json",
                )
                output = plugin.parse_output(generation.response)
            except Exception as exc:
                error = str(exc)
                self._report(progress_callback,
                             f"[{candidate_index}/{candidate_count}] {plugin.plugin_id} / {model.alias} "
                             f"batch {batch_index}/{len(batches)} failed: {error}")
                break
            outputs.append((batch, output, prompt_digest, generation))
            elapsed = time.perf_counter() - started
            self._report(progress_callback,
                         f"[{candidate_index}/{candidate_count}] {plugin.plugin_id} / {model.alias} "
                         f"batch {batch_index}/{len(batches)} completed in {elapsed:.1f}s "
                         f"(score {output.score:.2f}, {len(output.detected_issues)} issue(s)).")

        full_evaluation = len(evaluation_input.segments) == evaluation_input.total_segment_count
        status = ("completed" if len(outputs) == len(batches) and full_evaluation
                  else ("partial" if outputs else "failed"))
        merged = self._merge_outputs(outputs) if outputs else None
        batch_evidence = [{
            "segment_keys": [segment.stable_key for segment in batch],
            "output": output.to_dict(),
            "raw_response": generation.response,
            "rendered_prompt_hash": digest_prompt,
        } for batch, output, digest_prompt, generation in outputs]
        response_manifest = [{
            "model_returned": generation.model,
            "created_at": generation.created_at,
            "total_duration": generation.total_duration,
            "prompt_eval_count": generation.prompt_eval_count,
            "eval_count": generation.eval_count,
        } for _batch, _output, _digest_prompt, generation in outputs]
        runtime_manifest = {
            "provider": "ollama",
            "base_url": self.client.base_url,
            "endpoint": "/api/generate",
            "format": "json",
            "stream": False,
            "evaluator_alias": model.alias,
            "model_digest_available": digest is not None,
            "responses": response_manifest,
            "error": error,
        }
        model_run = ModelRun(
            provider="ollama", model_name=model.model_name, model_digest=digest,
            role="evaluator", prompt_hash=prompt_hash,
            options=dict(options), runtime_manifest=runtime_manifest,
        )
        self.session.add(model_run)
        self.session.flush()
        latest = self.session.scalar(select(func.max(EvaluationVersion.version_number)).where(
            EvaluationVersion.evaluation_id == evaluation_id
        )) or 0
        evaluation_version = EvaluationVersion(
            evaluation_id=evaluation_id,
            translation_version_id=evaluation_input.translation_version_id,
            model_run_id=model_run.id,
            version_number=latest + 1,
            status=status,
            rubric_version=plugin.version,
            plugin_manifest={
                "plugin_id": plugin.plugin_id,
                "plugin_version": plugin.version,
                "display_name": plugin.display_name,
                "prompt_version_id": str(prompt_version_id),
                "prompt_hash": prompt_hash,
            },
            run_manifest={
                "translation_version_id": str(evaluation_input.translation_version_id),
                "translation_content_hash": evaluation_input.translation_content_hash,
                "source_locale": evaluation_input.source_locale,
                "target_locale": evaluation_input.target_locale,
                "domain": evaluation_input.domain,
                "task_kind": evaluation_input.task_kind,
                "target_reading_level": evaluation_input.target_reading_level,
                "evaluator_alias": model.alias,
                "evaluator_model": model.model_name,
                "candidate_segment_count": len(evaluation_input.segments),
                "batch_size": config.batch_size,
                "batches": prompt_manifest,
                "error": error,
            },
        )
        self.session.add(evaluation_version)
        self.session.flush()

        if merged is not None:
            segment_keys = [segment.stable_key for segment in evaluation_input.segments]
            scope_key = ("document" if len(evaluation_input.segments) == evaluation_input.total_segment_count
                         else "partial")
            searchable_text = "\n".join((
                merged.reasoning,
                *merged.strengths,
                *merged.detected_issues,
                *(f"{item['type']} {item['severity']} {item['description']}"
                  for item in merged.issue_details),
                *merged.recommendations,
            ))
            result = EvaluationResult(
                evaluation_version_id=evaluation_version.id,
                plugin_id=plugin.plugin_id,
                metric_key="overall",
                scope_key=scope_key,
                score=merged.score,
                confidence=merged.confidence,
                reasoning=merged.reasoning,
                strengths=list(merged.strengths),
                recommendations=list(merged.recommendations),
                detected_issues=list(merged.detected_issues),
                issue_details=[dict(item) for item in merged.issue_details],
                searchable_text=searchable_text,
                value=merged.score,
                categorical_value=plugin.plugin_id,
                finding=merged.reasoning,
                evidence={"batch_results": batch_evidence, "segment_keys": segment_keys,
                          "status": status, "error": error},
            )
            self.session.add(result)
        self.session.commit()
        self._report(progress_callback,
                     f"[{candidate_index}/{candidate_count}] {plugin.plugin_id} / {model.alias}: "
                     f"saved evaluation version {evaluation_version.version_number} ({status}).")
        return EvaluationCandidateResult(
            plugin_id=plugin.plugin_id,
            model_alias=model.alias,
            model_name=model.model_name,
            evaluation_version_id=evaluation_version.id,
            version_number=evaluation_version.version_number,
            status=status,
            score=merged.score if merged else None,
            confidence=merged.confidence if merged else None,
            issue_count=len(merged.detected_issues) if merged else 0,
            recommendation_count=len(merged.recommendations) if merged else 0,
            error=error,
        )

    @staticmethod
    def _merge_outputs(
        outputs: list[tuple[tuple[EvaluationSegment, ...], EvaluationOutput, str, Generation]],
    ) -> EvaluationOutput:
        total_weight = sum(len(batch) for batch, _output, _digest, _generation in outputs)
        score = sum(output.score * len(batch) for batch, output, _digest, _generation in outputs) / total_weight
        confidence = sum(output.confidence * len(batch) for batch, output, _digest, _generation in outputs) / total_weight
        reasonings = []
        recommendations: list[str] = []
        issues: list[str] = []
        strengths: list[str] = []
        issue_details: list[dict[str, str]] = []
        for batch, output, _digest, _generation in outputs:
            span = f"{batch[0].stable_key}–{batch[-1].stable_key}"
            reasonings.append(f"[{span}] {output.reasoning}")
            for value, destination in ((output.recommendations, recommendations),
                                       (output.detected_issues, issues),
                                       (output.strengths, strengths)):
                known = {item.casefold() for item in destination}
                for item in value:
                    if item.casefold() not in known:
                        destination.append(item)
                        known.add(item.casefold())
            known_details = {json.dumps(item, sort_keys=True) for item in issue_details}
            for item in output.issue_details:
                encoded = json.dumps(item, sort_keys=True)
                if encoded not in known_details:
                    issue_details.append(dict(item))
                    known_details.add(encoded)
        return EvaluationOutput(
            score=round(score, 2), confidence=round(confidence, 4),
            reasoning="\n".join(reasonings), recommendations=tuple(recommendations),
            detected_issues=tuple(issues), strengths=tuple(strengths),
            issue_details=tuple(issue_details),
        )

    @staticmethod
    def _hash_text(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    @staticmethod
    def _report(callback: Callable[[str], None] | None, message: str) -> None:
        if callback is not None:
            callback(message)
