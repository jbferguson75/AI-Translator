from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

from sqlalchemy import DateTime, Uuid
from sqlalchemy import event, inspect
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """Declarative base shared by all persistence entities."""


class UUIDPrimaryKey:
    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)


class Timestamped:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )


@event.listens_for(Session, "before_flush")
def prevent_source_mutation(session: Session, _flush_context, _instances) -> None:
    """Reject ORM edits to source text/structure; DB triggers cover raw SQL."""
    immutable_fields = {
        "Document": {"source_text", "source_hash", "source_locale"},
        "Book": {"document_id", "title", "ordinal", "locator"},
        "Chapter": {"book_id", "title", "ordinal", "locator"},
        "Verse": {"chapter_id", "ordinal", "source_text", "source_hash"},
        "Chunk": {"document_id", "chapter_id", "verse_id", "stable_key", "ordinal",
                  "source_text", "source_hash", "locator"},
        "TranslationVersion": {"translation_id", "version_number", "parent_version_id",
                               "model_run_id", "author_kind", "status", "content_hash", "run_manifest"},
        "TranslationSegment": {"translation_version_id", "chunk_id", "target_text", "target_hash"},
        "EvaluationVersion": {"evaluation_id", "translation_version_id", "model_run_id",
                              "version_number", "status", "rubric_version", "plugin_manifest", "run_manifest"},
        "EvaluationResult": {"evaluation_version_id", "plugin_id", "metric_key", "scope_key",
                             "score", "confidence", "reasoning", "recommendations",
                             "detected_issues", "searchable_text", "value", "categorical_value",
                             "finding", "evidence"},
        "HumanReview": {"review_pass_id", "reviewer", "decision", "chunk_id", "comment",
                        "proposed_text", "resulting_translation_version_id"},
        "PromptVersion": {"prompt_key", "version_number", "template", "content_hash", "metadata_json"},
        "PublishedOutput": {"document_id", "translation_version_id", "target_locale",
                            "output_format", "content_hash", "artifact_uri", "published_by"},
    }
    for entity in session.dirty:
        names = immutable_fields.get(type(entity).__name__)
        if names and any(inspect(entity).attrs[name].history.has_changes() for name in names):
            raise ValueError("Original source text and structure are immutable; create a new document snapshot")
    append_only = {"TranslationVersion", "TranslationSegment", "EvaluationVersion",
                   "EvaluationResult", "HumanReview", "PromptVersion", "PublishedOutput",
                   "Book", "Chapter", "Verse", "Chunk"}
    if any(type(entity).__name__ in append_only for entity in session.deleted):
        raise ValueError("This record is append-only and cannot be deleted")

