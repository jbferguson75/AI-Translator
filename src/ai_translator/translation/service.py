from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from typing import Any, Callable
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..db.models import (
    Chunk,
    Document,
    Glossary,
    GlossaryEntry,
    ModelRun,
    PromptVersion,
    Translation,
    TranslationMemory,
    TranslationSegment,
    TranslationVersion,
)
from .config import ModelConfig, TranslationConfig
from .errors import OllamaError, TranslationError
from .ollama import Generation, OllamaClient
from .prompts import PromptTemplateManager, sha256_text


@dataclass(frozen=True, slots=True)
class CandidateResult:
    model_alias: str
    model_name: str
    version_id: UUID
    version_number: int
    status: str
    translated_chunks: int
    total_chunks: int
    error: str | None = None


@dataclass(frozen=True, slots=True)
class _SourceChunk:
    id: UUID
    stable_key: str
    ordinal: int
    source_text: str
    source_hash: str
    locator: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _MemoryMatch:
    id: UUID
    source_text: str
    target_text: str
    source_hash: str
    score: float


class TranslationService:
    """Generate one immutable TranslationVersion candidate per configured model.

    This service owns short persistence transactions. It commits the task and
    prompt snapshot before model calls, then commits each model candidate
    independently so later model failures cannot erase earlier candidates.
    """

    def __init__(self, session: Session, client: OllamaClient,
                 prompt_manager: PromptTemplateManager | None = None):
        self.session = session
        self.client = client
        self.prompt_manager = prompt_manager or PromptTemplateManager()

    def translate_document(
        self,
        *,
        document_id: UUID,
        target_locale: str,
        config: TranslationConfig,
        model_selection: list[str] | tuple[str, ...] | None = None,
        chunk_limit: int | None = None,
        task_kind: str | None = None,
        target_reading_level: str | None = None,
        progress_callback: Callable[[str], None] | None = None,
    ) -> list[CandidateResult]:
        if not target_locale.strip():
            raise TranslationError("Target locale is required")
        if chunk_limit is not None and chunk_limit < 1:
            raise TranslationError("Chunk limit must be a positive integer")
        models = config.resolve_models(model_selection)
        resolved_task_kind = task_kind or config.task_kind
        resolved_reading_level = (target_reading_level if target_reading_level is not None
                                  else config.target_reading_level)
        if resolved_reading_level is not None and not resolved_reading_level.strip():
            raise TranslationError("Target reading level cannot be blank")
        if resolved_reading_level and resolved_task_kind == "translation":
            resolved_task_kind = "simplification"

        document = self.session.get(Document, document_id)
        if document is None:
            raise TranslationError(f"Document {document_id} does not exist")
        source_chunks = list(self.session.scalars(
            select(Chunk).where(Chunk.document_id == document.id).order_by(Chunk.ordinal)
        ))
        if not source_chunks:
            raise TranslationError("Document has no translatable chunks")
        all_chunk_count = len(source_chunks)
        if chunk_limit is not None:
            source_chunks = source_chunks[:chunk_limit]
        chunks = tuple(_SourceChunk(c.id, c.stable_key, c.ordinal, c.source_text,
                                    c.source_hash, dict(c.locator or {})) for c in source_chunks)
        self._report(progress_callback,
                     f"Prepared {len(chunks)} of {all_chunk_count} source chunks from {document.title!r}.")
        self._report(progress_callback,
                     f"Task: {resolved_task_kind}; {document.source_locale} → {target_locale}; "
                     f"reading level: {resolved_reading_level or 'not specified'}.")
        self._report(progress_callback, "Models: " + ", ".join(
            f"{model.alias} ({model.model_name})" for model in models
        ))

        translation = self.session.scalar(select(Translation).where(
            Translation.document_id == document.id,
            Translation.target_locale == target_locale,
            Translation.task_kind == resolved_task_kind,
            Translation.target_reading_level == resolved_reading_level,
        ).order_by(Translation.created_at).limit(1))
        if translation is None:
            translation = Translation(
                document_id=document.id,
                target_locale=target_locale,
                task_kind=resolved_task_kind,
                target_reading_level=resolved_reading_level,
                domain_profile=document.domain,
            )
            self.session.add(translation)
            self.session.flush()
        translation_id = translation.id
        document_id = document.id
        project_id = document.project_id
        source_locale = document.source_locale
        domain = document.domain
        reading_level = resolved_reading_level or ""

        template_text, prompt_key = self.prompt_manager.load_template(
            domain, task_kind=resolved_task_kind, prompt_file=config.prompt_file
        )
        prompt_version = self.prompt_manager.get_or_create_version(
            self.session, prompt_key, template_text
        )
        prompt_version_id = prompt_version.id
        prompt_template_hash = prompt_version.content_hash

        glossary = self._load_glossary(project_id, source_locale, target_locale)
        memories = self._load_memories(project_id, source_locale, target_locale)
        self._report(progress_callback,
                     f"Loaded {len(glossary)} glossary terms and {len(memories)} approved translation memories.")
        contexts = {chunk.id: self._context_for(chunk, glossary, memories) for chunk in chunks}

        # No transaction remains open while Ollama runs.
        self.session.commit()

        digests: dict[str, str] = {}
        self._report(progress_callback, "Checking Ollama model metadata...")
        try:
            for item in self.client.list_models():
                name, digest = item.get("name"), item.get("digest")
                if isinstance(name, str) and isinstance(digest, str):
                    digests[name] = digest
        except OllamaError:
            # Generation calls below provide the authoritative connection error;
            # model digests remain explicitly unavailable for this run.
            pass
        self._report(progress_callback, "Starting model generation.")

        results: list[CandidateResult] = []
        for model_index, model in enumerate(models, start=1):
            results.append(self._run_candidate(
                model_index=model_index,
                model_count=len(models),
                model=model,
                chunks=chunks,
                contexts=contexts,
                all_chunk_count=all_chunk_count,
                translation_id=translation_id,
                document_id=document_id,
                source_locale=source_locale,
                target_locale=target_locale,
                task_kind=resolved_task_kind,
                domain=domain,
                reading_level=reading_level,
                template_text=template_text,
                prompt_key=prompt_key,
                prompt_version_id=prompt_version_id,
                prompt_template_hash=prompt_template_hash,
                config=config,
                digest=digests.get(model.model_name),
                progress_callback=progress_callback,
            ))
        return results

    def _run_candidate(
        self, *, model: ModelConfig, chunks: tuple[_SourceChunk, ...],
        contexts: dict[UUID, dict[str, Any]], all_chunk_count: int,
        translation_id: UUID, document_id: UUID, source_locale: str,
        target_locale: str, task_kind: str, domain: str, reading_level: str,
        template_text: str, prompt_key: str, prompt_version_id: UUID,
        prompt_template_hash: str, config: TranslationConfig, digest: str | None,
        model_index: int, model_count: int,
        progress_callback: Callable[[str], None] | None,
    ) -> CandidateResult:
        options: dict[str, Any] = {"temperature": config.temperature}
        if config.num_ctx is not None:
            options["num_ctx"] = config.num_ctx
        if config.num_predict is not None:
            options["num_predict"] = config.num_predict
        options.update(model.options)

        outputs: list[tuple[_SourceChunk, Generation, str]] = []
        prompt_manifest: list[dict[str, Any]] = []
        error: str | None = None
        self._report(progress_callback,
                     f"[{model_index}/{model_count}] {model.alias}: translating {len(chunks)} chunks.")
        for chunk_index, chunk in enumerate(chunks, start=1):
            context = contexts[chunk.id]
            rendered = self.prompt_manager.render(template_text, {
                "source_locale": source_locale,
                "target_locale": target_locale,
                "task_kind": task_kind,
                "domain": domain,
                "reading_level": reading_level,
                "glossary_terms": self._format_glossary(context["glossary"]),
                "translation_memory": self._format_memories(context["memories"]),
                "locator": json.dumps(chunk.locator, ensure_ascii=False, sort_keys=True),
                "source_text": chunk.source_text,
            })
            prompt_hash = sha256_text(rendered)
            prompt_manifest.append({
                "chunk_id": str(chunk.id),
                "stable_key": chunk.stable_key,
                "source_hash": chunk.source_hash,
                "rendered_prompt_hash": prompt_hash,
                "glossary_snapshot": context["glossary"],
                "translation_memory_snapshot": context["memories"],
            })
            self._report(progress_callback,
                         f"[{model_index}/{model_count}] {model.alias} chunk "
                         f"{chunk_index}/{len(chunks)} ({chunk.stable_key}): waiting for Ollama...")
            started = time.perf_counter()
            try:
                response = self.client.generate(model=model.model_name, prompt=rendered, options=options)
            except OllamaError as exc:
                error = str(exc)
                self._report(progress_callback,
                             f"[{model_index}/{model_count}] {model.alias} chunk "
                             f"{chunk_index}/{len(chunks)} failed: {error}")
                break
            outputs.append((chunk, response, prompt_hash))
            elapsed = time.perf_counter() - started
            self._report(progress_callback,
                         f"[{model_index}/{model_count}] {model.alias} chunk "
                         f"{chunk_index}/{len(chunks)} completed in {elapsed:.1f}s "
                         f"({len(response.response)} characters).")

        status = "completed" if len(outputs) == all_chunk_count else ("partial" if outputs else "failed")
        content_payload = [{"stable_key": chunk.stable_key, "target_text": response.response}
                           for chunk, response, _ in outputs]
        content_hash = hashlib.sha256(json.dumps(
            content_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")).hexdigest() if outputs else None
        response_manifest = [{
            "chunk_id": str(chunk.id),
            "model_returned": response.model,
            "created_at": response.created_at,
            "total_duration": response.total_duration,
            "prompt_eval_count": response.prompt_eval_count,
            "eval_count": response.eval_count,
        } for chunk, response, _ in outputs]

        model_run = ModelRun(
            provider="ollama",
            model_name=model.model_name,
            model_digest=digest,
            role="translator",
            prompt_hash=prompt_template_hash,
            options=options,
            runtime_manifest={
                "base_url": self.client.base_url,
                "endpoint": "/api/generate",
                "stream": False,
                "model_alias": model.alias,
                "model_digest_available": digest is not None,
                "responses": response_manifest,
                "error": error,
            },
        )
        self.session.add(model_run)
        self.session.flush()
        current_max = self.session.scalar(select(func.max(TranslationVersion.version_number)).where(
            TranslationVersion.translation_id == translation_id
        )) or 0
        version = TranslationVersion(
            translation_id=translation_id,
            version_number=current_max + 1,
            model_run_id=model_run.id,
            author_kind="model",
            status=status,
            content_hash=content_hash,
            run_manifest={
                "document_id": str(document_id),
                "source_locale": source_locale,
                "target_locale": target_locale,
                "task_kind": task_kind,
                "domain": domain,
                "model_alias": model.alias,
                "model_name": model.model_name,
                "prompt_key": prompt_key,
                "prompt_version_id": str(prompt_version_id),
                "prompt_template_hash": prompt_template_hash,
                "input_chunk_count": len(chunks),
                "document_chunk_count": all_chunk_count,
                "complete_document": len(chunks) == all_chunk_count,
                "chunks": prompt_manifest,
                "error": error,
            },
        )
        self.session.add(version)
        self.session.flush()
        for chunk, response, _prompt_hash in outputs:
            self.session.add(TranslationSegment(
                translation_version_id=version.id,
                chunk_id=chunk.id,
                target_text=response.response,
                target_hash=sha256_text(response.response),
            ))
        self.session.commit()
        self._report(progress_callback,
                     f"[{model_index}/{model_count}] {model.alias}: saved translation version "
                     f"{version.version_number} ({status}, {len(outputs)}/{all_chunk_count} chunks).")
        return CandidateResult(
            model_alias=model.alias,
            model_name=model.model_name,
            version_id=version.id,
            version_number=version.version_number,
            status=status,
            translated_chunks=len(outputs),
            total_chunks=all_chunk_count,
            error=error,
        )

    @staticmethod
    def _report(callback: Callable[[str], None] | None, message: str) -> None:
        if callback is not None:
            callback(message)

    def _load_glossary(self, project_id: UUID, source_locale: str, target_locale: str) -> list[dict[str, Any]]:
        rows = self.session.execute(
            select(GlossaryEntry).join(Glossary).where(
                Glossary.project_id == project_id,
                Glossary.source_locale == source_locale,
                Glossary.target_locale == target_locale,
            ).order_by(GlossaryEntry.source_term)
        ).scalars()
        return [{"id": str(row.id), "source": row.source_term, "target": row.target_term,
                 "notes": row.notes} for row in rows]

    def _load_memories(self, project_id: UUID, source_locale: str, target_locale: str) -> list[dict[str, Any]]:
        rows = self.session.scalars(select(TranslationMemory).where(
            TranslationMemory.project_id == project_id,
            TranslationMemory.source_locale == source_locale,
            TranslationMemory.target_locale == target_locale,
            TranslationMemory.approved.is_(True),
        ).order_by(TranslationMemory.created_at.desc()))
        return [{"id": str(row.id), "source": row.source_text, "target": row.target_text,
                 "source_hash": row.source_hash} for row in rows]

    @staticmethod
    def _context_for(chunk: _SourceChunk, glossary: list[dict[str, Any]],
                     memories: list[dict[str, Any]]) -> dict[str, Any]:
        source_folded = chunk.source_text.casefold()
        applicable_glossary = [entry for entry in glossary
                               if entry["source"].casefold() in source_folded]
        tokens = set(re.findall(r"\w+", source_folded, flags=re.UNICODE))
        ranked: list[tuple[float, dict[str, Any]]] = []
        for item in memories:
            if item["source_hash"] == chunk.source_hash:
                score = 1.0
            else:
                memory_tokens = set(re.findall(r"\w+", item["source"].casefold(), flags=re.UNICODE))
                union = tokens | memory_tokens
                score = len(tokens & memory_tokens) / len(union) if union else 0.0
            if score > 0.0:
                ranked.append((score, item))
        ranked.sort(key=lambda pair: (-pair[0], pair[1]["id"]))
        selected = [{**item, "similarity": round(score, 4)} for score, item in ranked[:3]]
        return {"glossary": applicable_glossary, "memories": selected}

    @staticmethod
    def _format_glossary(entries: list[dict[str, Any]]) -> str:
        return "\n".join(f"{item['source']} → {item['target']}" +
                          (f" ({item['notes']})" if item.get("notes") else "") for item in entries)

    @staticmethod
    def _format_memories(items: list[dict[str, Any]]) -> str:
        return "\n".join(f"{item['source']} → {item['target']}" for item in items)

