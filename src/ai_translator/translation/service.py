from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..db.models import (
    Chunk,
    Document,
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
class CandidateResult:
    model_alias: str
    model_name: str
    version_id: UUID
    version_number: int
    status: str
    translated_chunks: int
    total_chunks: int
    error: str | None = None
    scope_label: str = "document"


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
        book: str | None = None,
        chapter: int | None = None,
        based_on_translation_version_id: UUID | None = None,
        progress_callback: Callable[[str], None] | None = None,
    ) -> list[CandidateResult]:
        if not target_locale.strip():
            raise TranslationError("Target locale is required")
        if chunk_limit is not None and chunk_limit < 1:
            raise TranslationError("Chunk limit must be a positive integer")
        if chapter is not None and chapter < 1:
            raise TranslationError("Chapter number must be a positive integer")
        if chapter is not None and not book:
            raise TranslationError("Pass --book when selecting a chapter")
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
        document_chunks = list(self.session.scalars(
            select(Chunk).where(Chunk.document_id == document.id).order_by(Chunk.ordinal)
        ))
        if not document_chunks:
            raise TranslationError("Document has no translatable chunks")
        document_chunk_count = len(document_chunks)
        source_chunks = document_chunks
        scope_label = "document"
        scope_kind = "document"
        if book is not None:
            source_chunks = [chunk for chunk in source_chunks
                             if str((chunk.locator or {}).get("book", "")).casefold() == book.casefold()]
            if not source_chunks:
                raise TranslationError(f"Book {book!r} was not found in document {document.title!r}")
            scope_label = book
            scope_kind = "book"
        if chapter is not None:
            source_chunks = [chunk for chunk in source_chunks
                             if (chunk.locator or {}).get("chapter") == chapter]
            if not source_chunks:
                raise TranslationError(f"Chapter {chapter} of {book!r} has no source chunks")
            scope_label = f"{book} {chapter}"
            scope_kind = "chapter"
        scope_chunk_count = len(source_chunks)
        if chunk_limit is not None:
            source_chunks = source_chunks[:chunk_limit]
        chunks = tuple(_SourceChunk(c.id, c.stable_key, c.ordinal, c.source_text,
                                    c.source_hash, dict(c.locator or {})) for c in source_chunks)
        self._report(progress_callback,
                     f"Prepared {len(chunks)} of {scope_chunk_count} {scope_label} chunks "
                     f"({document_chunk_count} in the document) from {document.title!r}.")
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
        if based_on_translation_version_id is not None:
            parent_version = self.session.get(TranslationVersion, based_on_translation_version_id)
            if parent_version is None or parent_version.translation_id != translation_id:
                raise TranslationError("Based-on translation version must belong to this translation task")
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

        contexts = self._build_contexts(
            chunks=chunks, project_id=project_id, source_locale=source_locale,
            target_locale=target_locale,
        )
        glossary_count = len({item["id"] for context in contexts.values() for item in context["glossary"]})
        memory_count = len({item["id"] for context in contexts.values() for item in context["memories"]})
        self._report(progress_callback,
                     f"Prepared {glossary_count} glossary term(s) and "
                     f"{memory_count} distinct approved memory match(es) for these chunks.")

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
                scope_chunk_count=scope_chunk_count,
                document_chunk_count=document_chunk_count,
                scope_label=scope_label,
                scope_kind=scope_kind,
                scope_book=book,
                scope_chapter=chapter,
                parent_version_id=based_on_translation_version_id,
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
        contexts: dict[UUID, dict[str, Any]], scope_chunk_count: int,
        document_chunk_count: int, scope_label: str, scope_kind: str,
        scope_book: str | None, scope_chapter: int | None,
        parent_version_id: UUID | None,
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

        status = "completed" if len(outputs) == scope_chunk_count else ("partial" if outputs else "failed")
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
            parent_version_id=parent_version_id,
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
                "scope": {"kind": scope_kind, "label": scope_label,
                          "book": scope_book, "chapter": scope_chapter},
                "based_on_translation_version_id": (str(parent_version_id)
                                                     if parent_version_id else None),
                "scope_chunk_count": scope_chunk_count,
                "document_chunk_count": document_chunk_count,
                "complete_scope": len(outputs) == scope_chunk_count,
                "complete_document": len(outputs) == document_chunk_count,
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
        glossary_usage: dict[UUID, set[str]] = defaultdict(set)
        memory_usage: dict[UUID, set[str]] = defaultdict(set)
        for chunk, _response, _prompt_hash in outputs:
            reference = f"{document_id}:{chunk.stable_key}"
            for item in contexts[chunk.id]["glossary"]:
                glossary_usage[UUID(str(item["id"]))].add(reference)
            for item in contexts[chunk.id]["memories"]:
                memory_usage[UUID(str(item["id"]))].add(reference)
        GlossaryService(self.session).record_usage(glossary_usage)
        TranslationMemoryService(self.session).record_usage(memory_usage)
        self.session.commit()
        self._report(progress_callback,
                     f"[{model_index}/{model_count}] {model.alias}: saved translation version "
                     f"{version.version_number} ({status}, {len(outputs)}/{scope_chunk_count} "
                     f"{scope_label} chunks).")
        return CandidateResult(
            model_alias=model.alias,
            model_name=model.model_name,
            version_id=version.id,
            version_number=version.version_number,
            status=status,
            translated_chunks=len(outputs),
            total_chunks=scope_chunk_count,
            error=error,
            scope_label=scope_label,
        )

    @staticmethod
    def _report(callback: Callable[[str], None] | None, message: str) -> None:
        if callback is not None:
            callback(message)

    def _build_contexts(self, *, chunks: tuple[_SourceChunk, ...], project_id: UUID,
                        source_locale: str, target_locale: str) -> dict[UUID, dict[str, Any]]:
        glossary_service = GlossaryService(self.session)
        memory_service = TranslationMemoryService(self.session)
        glossary_entries = glossary_service.entries_for_scope(project_id, source_locale, target_locale)
        memory_entries = memory_service.list_for_project(
            project_id, source_locale=source_locale, target_locale=target_locale,
        )
        contexts: dict[UUID, dict[str, Any]] = {}
        for chunk in chunks:
            try:
                glossary = glossary_service.resolve_for_text(
                    project_id=project_id, source_locale=source_locale,
                    target_locale=target_locale, source_text=chunk.source_text,
                    entries=glossary_entries,
                )
                matches = memory_service.lookup(
                    project_id=project_id, source_locale=source_locale,
                    target_locale=target_locale, source_phrase=chunk.source_text, limit=3,
                    entries=memory_entries,
                )
            except (GlossaryConflictError, TranslationMemoryConflictError) as exc:
                raise TranslationError(str(exc)) from exc
            memories = [{
                "id": str(match.entry.id), "source": match.entry.source_text,
                "target": match.entry.target_text, "source_hash": match.entry.source_hash,
                "similarity": round(match.similarity, 4), "approved": match.entry.approved,
                "occurrence_count": match.entry.occurrence_count,
            } for match in matches]
            contexts[chunk.id] = {"glossary": glossary, "memories": memories}
        return contexts

    @staticmethod
    def _format_glossary(entries: list[dict[str, Any]]) -> str:
        return "\n".join(
            f"{item['source']} → {item['target']} "
            f"[priority={item['priority']}; glossary={item['glossary']}]" +
            (f" ({item['notes']})" if item.get("notes") else "")
            for item in entries
        )

    @staticmethod
    def _format_memories(items: list[dict[str, Any]]) -> str:
        return "\n".join(
            f"{item['source']} → {item['target']} [similarity={item['similarity']:.3f}]"
            for item in items
        )

