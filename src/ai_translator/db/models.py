from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship, synonym

from .base import Base, Timestamped, UUIDPrimaryKey, utc_now


class Project(UUIDPrimaryKey, Timestamped, Base):
    __tablename__ = "projects"
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    default_source_locale: Mapped[str] = mapped_column(String(35), nullable=False,
                                                      default="en", server_default="en")
    settings: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    documents: Mapped[list[Document]] = relationship(back_populates="project", cascade="all, delete-orphan")
    experiments: Mapped[list[Experiment]] = relationship(back_populates="project")


class Document(UUIDPrimaryKey, Timestamped, Base):
    __tablename__ = "documents"
    __table_args__ = (Index("ix_documents_project_title", "project_id", "title"),)
    project_id: Mapped[UUID] = mapped_column(ForeignKey("projects.id", ondelete="RESTRICT"), nullable=False)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    domain: Mapped[str] = mapped_column(String(40), nullable=False, default="general")
    source_locale: Mapped[str] = mapped_column(String(35), nullable=False)
    source_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    source_text: Mapped[str] = mapped_column(Text, nullable=False)
    source_format: Mapped[str | None] = mapped_column(String(80))
    project: Mapped[Project] = relationship(back_populates="documents")
    books: Mapped[list[Book]] = relationship(back_populates="document", cascade="all, delete-orphan")
    translations: Mapped[list[Translation]] = relationship(back_populates="document")


class Book(UUIDPrimaryKey, Timestamped, Base):
    __tablename__ = "books"
    __table_args__ = (UniqueConstraint("document_id", "ordinal", name="uq_books_document_ordinal"),)
    document_id: Mapped[UUID] = mapped_column(ForeignKey("documents.id", ondelete="RESTRICT"), nullable=False)
    title: Mapped[str] = mapped_column(String(300), nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    locator: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    document: Mapped[Document] = relationship(back_populates="books")
    chapters: Mapped[list[Chapter]] = relationship(back_populates="book", cascade="all, delete-orphan")


class Chapter(UUIDPrimaryKey, Timestamped, Base):
    __tablename__ = "chapters"
    __table_args__ = (UniqueConstraint("book_id", "ordinal", name="uq_chapters_book_ordinal"),)
    book_id: Mapped[UUID] = mapped_column(ForeignKey("books.id", ondelete="RESTRICT"), nullable=False)
    title: Mapped[str | None] = mapped_column(String(300))
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    locator: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    book: Mapped[Book] = relationship(back_populates="chapters")
    verses: Mapped[list[Verse]] = relationship(back_populates="chapter", cascade="all, delete-orphan")
    chunks: Mapped[list[Chunk]] = relationship(back_populates="chapter")


class Verse(UUIDPrimaryKey, Timestamped, Base):
    __tablename__ = "verses"
    __table_args__ = (UniqueConstraint("chapter_id", "ordinal", name="uq_verses_chapter_ordinal"),)
    chapter_id: Mapped[UUID] = mapped_column(ForeignKey("chapters.id", ondelete="RESTRICT"), nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    source_text: Mapped[str] = mapped_column(Text, nullable=False)
    source_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    chapter: Mapped[Chapter] = relationship(back_populates="verses")
    chunks: Mapped[list[Chunk]] = relationship(back_populates="verse")


class Chunk(UUIDPrimaryKey, Timestamped, Base):
    """Immutable source span used as the alignment unit for generated text."""
    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("document_id", "stable_key", name="uq_chunks_document_stable_key"),
        Index("ix_chunks_document_ordinal", "document_id", "ordinal"),
    )
    document_id: Mapped[UUID] = mapped_column(ForeignKey("documents.id", ondelete="RESTRICT"), nullable=False)
    chapter_id: Mapped[UUID | None] = mapped_column(ForeignKey("chapters.id", ondelete="RESTRICT"))
    verse_id: Mapped[UUID | None] = mapped_column(ForeignKey("verses.id", ondelete="RESTRICT"))
    stable_key: Mapped[str] = mapped_column(String(300), nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    source_text: Mapped[str] = mapped_column(Text, nullable=False)
    source_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    locator: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    document: Mapped[Document] = relationship()
    chapter: Mapped[Chapter | None] = relationship(back_populates="chunks")
    verse: Mapped[Verse | None] = relationship(back_populates="chunks")


class Translation(UUIDPrimaryKey, Timestamped, Base):
    __tablename__ = "translations"
    __table_args__ = (Index("ix_translations_document_locale", "document_id", "target_locale"),)
    document_id: Mapped[UUID] = mapped_column(ForeignKey("documents.id", ondelete="RESTRICT"), nullable=False)
    target_locale: Mapped[str] = mapped_column(String(35), nullable=False)
    task_kind: Mapped[str] = mapped_column(String(40), nullable=False, default="translation")
    target_reading_level: Mapped[str | None] = mapped_column(String(80))
    domain_profile: Mapped[str | None] = mapped_column(String(80))
    document: Mapped[Document] = relationship(back_populates="translations")
    versions: Mapped[list[TranslationVersion]] = relationship(back_populates="translation", cascade="all, delete-orphan", order_by="TranslationVersion.version_number")


class ModelRun(UUIDPrimaryKey, Timestamped, Base):
    __tablename__ = "model_runs"
    provider: Mapped[str] = mapped_column(String(80), nullable=False, default="ollama")
    model_name: Mapped[str] = mapped_column(String(200), nullable=False)
    model_digest: Mapped[str | None] = mapped_column(String(200))
    role: Mapped[str] = mapped_column(String(30), nullable=False)
    prompt_hash: Mapped[str | None] = mapped_column(String(64))
    runtime_manifest: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    options: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)


class TranslationVersion(UUIDPrimaryKey, Timestamped, Base):
    __tablename__ = "translation_versions"
    __table_args__ = (
        UniqueConstraint("translation_id", "version_number", name="uq_translation_versions_number"),
        Index("ix_translation_versions_translation_created", "translation_id", "created_at"),
        Index("ix_translation_versions_artifact_kind", "artifact_kind", "created_at"),
        CheckConstraint(
            "artifact_kind IN ('ORIGINAL_TRANSLATION', 'REFINED_TRANSLATION', "
            "'SYNTHESIZED_TRANSLATION', 'HUMAN_EDIT', 'FINAL_APPROVED')",
            name="ck_translation_versions_artifact_kind",
        ),
    )
    translation_id: Mapped[UUID] = mapped_column(ForeignKey("translations.id", ondelete="RESTRICT"), nullable=False)
    version_number: Mapped[int] = mapped_column(Integer, nullable=False)
    parent_version_id: Mapped[UUID | None] = mapped_column(ForeignKey("translation_versions.id", ondelete="RESTRICT"))
    model_run_id: Mapped[UUID | None] = mapped_column(ForeignKey("model_runs.id", ondelete="RESTRICT"))
    author_kind: Mapped[str] = mapped_column(String(20), nullable=False, default="model")
    artifact_kind: Mapped[str] = mapped_column(
        String(40), nullable=False, default="ORIGINAL_TRANSLATION", server_default="ORIGINAL_TRANSLATION",
    )
    status: Mapped[str] = mapped_column(String(30), nullable=False, default="draft")
    content_hash: Mapped[str | None] = mapped_column(String(64))
    prompt_version_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("prompt_versions.id", ondelete="RESTRICT",
                   name="fk_translation_versions_prompt_version_id_prompt_versions")
    )
    run_manifest: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    translation: Mapped[Translation] = relationship(back_populates="versions")
    parent: Mapped[TranslationVersion | None] = relationship(remote_side="TranslationVersion.id", foreign_keys=[parent_version_id])
    model_run: Mapped[ModelRun | None] = relationship()
    prompt_version: Mapped[PromptVersion | None] = relationship()
    lineage_parents: Mapped[list[TranslationVersionParent]] = relationship(
        foreign_keys="TranslationVersionParent.child_version_id",
        back_populates="child", cascade="all, delete-orphan",
    )
    lineage_children: Mapped[list[TranslationVersionParent]] = relationship(
        foreign_keys="TranslationVersionParent.parent_version_id",
        back_populates="parent",
    )
    segments: Mapped[list[TranslationSegment]] = relationship(back_populates="translation_version", cascade="all, delete-orphan")
    glossary_suggestion_runs: Mapped[list[GlossarySuggestionRun]] = relationship(back_populates="translation_version")


class TranslationVersionParent(Base):
    """Append-only lineage edge; supports multiple source candidates per artifact."""
    __tablename__ = "translation_version_parents"
    __table_args__ = (
        Index("ix_translation_version_parents_parent", "parent_version_id", "child_version_id"),
        CheckConstraint(
            "relationship_kind IN ('parent', 'source_candidate')",
            name="ck_translation_version_parents_kind",
        ),
    )
    child_version_id: Mapped[UUID] = mapped_column(
        ForeignKey("translation_versions.id", ondelete="RESTRICT",
                   name="fk_translation_version_parents_child"), primary_key=True,
    )
    parent_version_id: Mapped[UUID] = mapped_column(
        ForeignKey("translation_versions.id", ondelete="RESTRICT",
                   name="fk_translation_version_parents_parent"), primary_key=True,
    )
    relationship_kind: Mapped[str] = mapped_column(String(30), primary_key=True, default="parent")
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now)
    child: Mapped[TranslationVersion] = relationship(
        foreign_keys=[child_version_id], back_populates="lineage_parents",
    )
    parent: Mapped[TranslationVersion] = relationship(
        foreign_keys=[parent_version_id], back_populates="lineage_children",
    )


class TranslationSegment(UUIDPrimaryKey, Base):
    __tablename__ = "translation_segments"
    __table_args__ = (UniqueConstraint("translation_version_id", "chunk_id", name="uq_translation_segment_chunk"),)
    translation_version_id: Mapped[UUID] = mapped_column(ForeignKey("translation_versions.id", ondelete="RESTRICT"), nullable=False)
    chunk_id: Mapped[UUID] = mapped_column(ForeignKey("chunks.id", ondelete="RESTRICT"), nullable=False)
    target_text: Mapped[str] = mapped_column(Text, nullable=False)
    target_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    translation_version: Mapped[TranslationVersion] = relationship(back_populates="segments")
    chunk: Mapped[Chunk] = relationship()


class Evaluation(UUIDPrimaryKey, Timestamped, Base):
    __tablename__ = "evaluations"
    translation_id: Mapped[UUID] = mapped_column(ForeignKey("translations.id", ondelete="RESTRICT"), nullable=False)
    evaluator_name: Mapped[str | None] = mapped_column(String(200))
    evaluations: Mapped[list[EvaluationVersion]] = relationship(back_populates="evaluation", cascade="all, delete-orphan")


class EvaluationVersion(UUIDPrimaryKey, Timestamped, Base):
    __tablename__ = "evaluation_versions"
    __table_args__ = (
        UniqueConstraint("evaluation_id", "version_number", name="uq_evaluation_versions_number"),
        Index("ix_evaluation_versions_eval_created", "evaluation_id", "created_at"),
    )
    evaluation_id: Mapped[UUID] = mapped_column(ForeignKey("evaluations.id", ondelete="RESTRICT"), nullable=False)
    translation_version_id: Mapped[UUID] = mapped_column(ForeignKey("translation_versions.id", ondelete="RESTRICT"), nullable=False)
    model_run_id: Mapped[UUID | None] = mapped_column(ForeignKey("model_runs.id", ondelete="RESTRICT"))
    version_number: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(30), nullable=False, default="completed")
    rubric_version: Mapped[str | None] = mapped_column(String(100))
    plugin_manifest: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    run_manifest: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    evaluation: Mapped[Evaluation] = relationship(back_populates="evaluations")
    translation_version: Mapped[TranslationVersion] = relationship()
    model_run: Mapped[ModelRun | None] = relationship()
    results: Mapped[list[EvaluationResult]] = relationship(back_populates="evaluation_version", cascade="all, delete-orphan")


class EvaluationResult(UUIDPrimaryKey, Base):
    __tablename__ = "evaluation_results"
    __table_args__ = (Index("ix_evaluation_results_plugin_metric", "plugin_id", "metric_key"),)
    evaluation_version_id: Mapped[UUID] = mapped_column(ForeignKey("evaluation_versions.id", ondelete="RESTRICT"), nullable=False)
    plugin_id: Mapped[str] = mapped_column(String(100), nullable=False, default="legacy")
    metric_key: Mapped[str] = mapped_column(String(200), nullable=False)
    scope_key: Mapped[str] = mapped_column(String(300), nullable=False, default="document")
    score: Mapped[float | None] = mapped_column(Float)
    confidence: Mapped[float | None] = mapped_column(Float)
    reasoning: Mapped[str | None] = mapped_column(Text)
    strengths: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    recommendations: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    detected_issues: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    issue_details: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False, default=list)
    searchable_text: Mapped[str] = mapped_column(Text, nullable=False, default="")
    value: Mapped[float | None] = mapped_column(Float)
    categorical_value: Mapped[str | None] = mapped_column(String(200))
    finding: Mapped[str | None] = mapped_column(Text)
    evidence: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    evaluation_version: Mapped[EvaluationVersion] = relationship(back_populates="results")


class ReviewPass(UUIDPrimaryKey, Timestamped, Base):
    __tablename__ = "review_passes"
    translation_version_id: Mapped[UUID] = mapped_column(ForeignKey("translation_versions.id", ondelete="RESTRICT"), nullable=False)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    status: Mapped[str] = mapped_column(String(30), nullable=False, default="open")
    assigned_to: Mapped[str | None] = mapped_column(String(200))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    translation_version: Mapped[TranslationVersion] = relationship()
    reviews: Mapped[list[HumanReview]] = relationship(back_populates="review_pass", cascade="all, delete-orphan")


class HumanReview(UUIDPrimaryKey, Timestamped, Base):
    __tablename__ = "human_reviews"
    __table_args__ = (
        CheckConstraint(
            "status IN ('pending', 'approved', 'rejected', 'needs_revision')",
            name="ck_human_reviews_status",
        ),
        CheckConstraint(
            "(status = 'approved' AND approval_date IS NOT NULL) OR "
            "(status != 'approved' AND approval_date IS NULL)",
            name="ck_human_reviews_approval_date",
        ),
        Index("ix_human_reviews_pass_chunk_created", "review_pass_id", "chunk_id", "created_at"),
        Index("ix_human_reviews_status_created", "status", "created_at"),
    )
    review_pass_id: Mapped[UUID] = mapped_column(ForeignKey("review_passes.id", ondelete="RESTRICT"), nullable=False)
    reviewer: Mapped[str] = mapped_column(String(200), nullable=False)
    decision: Mapped[str] = mapped_column(String(30), nullable=False)
    status: Mapped[str] = mapped_column(String(30), nullable=False, default="pending", server_default="pending")
    chunk_id: Mapped[UUID | None] = mapped_column(ForeignKey("chunks.id", ondelete="RESTRICT"))
    comment: Mapped[str | None] = mapped_column(Text)
    proposed_text: Mapped[str | None] = mapped_column(Text)
    review_notes: Mapped[str | None] = mapped_column(Text)
    correction: Mapped[str | None] = mapped_column(Text)
    override_reason: Mapped[str | None] = mapped_column(Text)
    approval_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    supersedes_review_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("human_reviews.id", ondelete="RESTRICT", name="fk_human_reviews_supersedes")
    )
    previous_event_hash: Mapped[str | None] = mapped_column(String(64))
    event_hash: Mapped[str] = mapped_column(String(64), nullable=False, server_default=text("''"))
    resulting_translation_version_id: Mapped[UUID | None] = mapped_column(ForeignKey("translation_versions.id", ondelete="RESTRICT"))
    review_pass: Mapped[ReviewPass] = relationship(back_populates="reviews")
    supersedes: Mapped[HumanReview | None] = relationship(
        remote_side="HumanReview.id", foreign_keys=[supersedes_review_id]
    )


class Glossary(UUIDPrimaryKey, Timestamped, Base):
    __tablename__ = "glossaries"
    project_id: Mapped[UUID] = mapped_column(ForeignKey("projects.id", ondelete="RESTRICT"), nullable=False)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    source_locale: Mapped[str] = mapped_column(String(35), nullable=False)
    target_locale: Mapped[str] = mapped_column(String(35), nullable=False)
    target_language = synonym("target_locale")
    project_scope = synonym("project_id")
    entries: Mapped[list[GlossaryEntry]] = relationship(back_populates="glossary", cascade="all, delete-orphan")


class GlossaryEntry(UUIDPrimaryKey, Timestamped, Base):
    __tablename__ = "glossary_entries"
    __table_args__ = (
        UniqueConstraint("glossary_id", "source_term", name="uq_glossary_source_term"),
        Index("ix_glossary_entries_term_priority", "source_term", "priority"),
    )
    glossary_id: Mapped[UUID] = mapped_column(ForeignKey("glossaries.id", ondelete="RESTRICT"), nullable=False)
    source_term: Mapped[str] = mapped_column(String(500), nullable=False)
    target_term: Mapped[str] = mapped_column(String(500), nullable=False)
    preferred_translation = synonym("target_term")
    priority: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    notes: Mapped[str | None] = mapped_column(Text)
    occurrence_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    source_references: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list,
                                                        server_default=text("'[]'"))
    glossary: Mapped[Glossary] = relationship(back_populates="entries")


class TranslationMemory(UUIDPrimaryKey, Timestamped, Base):
    __tablename__ = "translation_memories"
    __table_args__ = (Index("ix_translation_memory_locales", "source_locale", "target_locale"),)
    project_id: Mapped[UUID] = mapped_column(ForeignKey("projects.id", ondelete="RESTRICT"), nullable=False)
    source_locale: Mapped[str] = mapped_column(String(35), nullable=False)
    target_locale: Mapped[str] = mapped_column(String(35), nullable=False)
    source_text: Mapped[str] = mapped_column(Text, nullable=False)
    target_text: Mapped[str] = mapped_column(Text, nullable=False)
    source_phrase = synonym("source_text")
    preferred_translation = synonym("target_text")
    source_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    origin_translation_version_id: Mapped[UUID | None] = mapped_column(ForeignKey("translation_versions.id", ondelete="RESTRICT"))
    approved: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    occurrence_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    source_references: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list,
                                                        server_default=text("'[]'"))


class GlossarySuggestionRun(UUIDPrimaryKey, Timestamped, Base):
    __tablename__ = "glossary_suggestion_runs"
    __table_args__ = (
        UniqueConstraint("translation_version_id", "version_number", name="uq_glossary_suggestion_run_version"),
        Index("ix_glossary_suggestion_runs_version_status", "translation_version_id", "status"),
    )
    translation_version_id: Mapped[UUID] = mapped_column(
        ForeignKey("translation_versions.id", ondelete="RESTRICT"), nullable=False
    )
    model_run_id: Mapped[UUID | None] = mapped_column(ForeignKey("model_runs.id", ondelete="RESTRICT"))
    version_number: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(30), nullable=False, default="running")
    model_name: Mapped[str] = mapped_column(String(200), nullable=False)
    model_digest: Mapped[str | None] = mapped_column(String(200))
    prompt_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    raw_response: Mapped[str | None] = mapped_column(Text)
    run_manifest: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    translation_version: Mapped[TranslationVersion] = relationship(back_populates="glossary_suggestion_runs")
    suggestions: Mapped[list[GlossarySuggestion]] = relationship(back_populates="suggestion_run")


class GlossarySuggestion(UUIDPrimaryKey, Timestamped, Base):
    __tablename__ = "glossary_suggestions"
    __table_args__ = (
        Index("ix_glossary_suggestions_run_status", "suggestion_run_id", "status"),
        Index("ix_glossary_suggestions_project_term", "project_id", "source_term"),
    )
    suggestion_run_id: Mapped[UUID] = mapped_column(
        ForeignKey("glossary_suggestion_runs.id", ondelete="RESTRICT"), nullable=False
    )
    project_id: Mapped[UUID] = mapped_column(ForeignKey("projects.id", ondelete="RESTRICT"), nullable=False)
    source_locale: Mapped[str] = mapped_column(String(35), nullable=False)
    target_locale: Mapped[str] = mapped_column(String(35), nullable=False)
    source_term: Mapped[str] = mapped_column(String(500), nullable=False)
    preferred_translation: Mapped[str] = mapped_column(String(500), nullable=False)
    occurrence_count: Mapped[int] = mapped_column(Integer, nullable=False)
    source_references: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    reasoning: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(30), nullable=False, default="proposed")
    glossary_entry_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("glossary_entries.id", ondelete="RESTRICT")
    )
    resolved_by: Mapped[str | None] = mapped_column(String(200))
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    resolution_note: Mapped[str | None] = mapped_column(Text)
    suggestion_run: Mapped[GlossarySuggestionRun] = relationship(back_populates="suggestions")
    glossary_entry: Mapped[GlossaryEntry | None] = relationship()


class PromptVersion(UUIDPrimaryKey, Timestamped, Base):
    __tablename__ = "prompt_versions"
    __table_args__ = (UniqueConstraint("prompt_key", "version_number", name="uq_prompt_version_number"),)
    prompt_key: Mapped[str] = mapped_column(String(200), nullable=False)
    version_number: Mapped[int] = mapped_column(Integer, nullable=False)
    template: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)


class Experiment(UUIDPrimaryKey, Timestamped, Base):
    __tablename__ = "experiments"
    project_id: Mapped[UUID] = mapped_column(ForeignKey("projects.id", ondelete="RESTRICT"), nullable=False)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    configuration: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    runs: Mapped[list[ExperimentRun]] = relationship(back_populates="experiment", cascade="all, delete-orphan")
    project: Mapped[Project] = relationship(back_populates="experiments")


class ExperimentRun(UUIDPrimaryKey, Timestamped, Base):
    __tablename__ = "experiment_runs"
    experiment_id: Mapped[UUID] = mapped_column(ForeignKey("experiments.id", ondelete="RESTRICT"), nullable=False)
    translation_version_id: Mapped[UUID | None] = mapped_column(ForeignKey("translation_versions.id", ondelete="RESTRICT"))
    evaluation_version_id: Mapped[UUID | None] = mapped_column(ForeignKey("evaluation_versions.id", ondelete="RESTRICT"))
    label: Mapped[str | None] = mapped_column(String(200))
    parameters: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    experiment: Mapped[Experiment] = relationship(back_populates="runs")


class PublishedOutput(UUIDPrimaryKey, Timestamped, Base):
    __tablename__ = "published_outputs"
    __table_args__ = (Index("ix_published_outputs_document_locale", "document_id", "target_locale"),)
    document_id: Mapped[UUID] = mapped_column(ForeignKey("documents.id", ondelete="RESTRICT"), nullable=False)
    translation_version_id: Mapped[UUID] = mapped_column(ForeignKey("translation_versions.id", ondelete="RESTRICT"), nullable=False)
    target_locale: Mapped[str] = mapped_column(String(35), nullable=False)
    output_format: Mapped[str] = mapped_column(String(80), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    artifact_uri: Mapped[str] = mapped_column(Text, nullable=False)
    published_by: Mapped[str] = mapped_column(String(200), nullable=False)
    published_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now)


# Names are explicit in alembic migrations; these constraints make versioning
# and valid foreign key references enforceable in SQLite as well as other DBs.

