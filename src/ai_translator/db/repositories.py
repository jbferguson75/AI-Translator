from __future__ import annotations

from typing import Generic, TypeVar
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .models import (
    Document, Evaluation, EvaluationResult, EvaluationVersion, Project, Translation,
    TranslationVersion,
)

T = TypeVar("T")


class Repository(Generic[T]):
    """Small base repository. Callers own the session and transaction."""
    model: type[T]

    def __init__(self, session: Session):
        self.session = session

    def add(self, entity: T) -> T:
        self.session.add(entity)
        return entity

    def get(self, entity_id: UUID) -> T | None:
        return self.session.get(self.model, entity_id)

    def delete(self, entity: T) -> None:
        self.session.delete(entity)


class ProjectRepository(Repository[Project]):
    model = Project

    def by_name(self, name: str) -> list[Project]:
        return list(self.session.scalars(select(Project).where(Project.name == name).order_by(Project.created_at)))


class DocumentRepository(Repository[Document]):
    model = Document

    def list_all(self) -> list[Document]:
        return list(self.session.scalars(select(Document).order_by(Document.title, Document.created_at)))

    def for_project(self, project_id: UUID) -> list[Document]:
        return list(self.session.scalars(select(Document).where(Document.project_id == project_id).order_by(Document.title)))


class TranslationRepository(Repository[Translation]):
    model = Translation

    def versions(self, translation_id: UUID) -> list[TranslationVersion]:
        return list(self.session.scalars(select(TranslationVersion).where(
            TranslationVersion.translation_id == translation_id
        ).order_by(TranslationVersion.version_number)))

    def next_version_number(self, translation_id: UUID) -> int:
        versions = self.versions(translation_id)
        return (versions[-1].version_number + 1) if versions else 1


class EvaluationRepository(Repository[Evaluation]):
    model = Evaluation

    def versions(self, evaluation_id: UUID) -> list[EvaluationVersion]:
        return list(self.session.scalars(select(EvaluationVersion).where(
            EvaluationVersion.evaluation_id == evaluation_id
        ).order_by(EvaluationVersion.version_number)))

    def next_version_number(self, evaluation_id: UUID) -> int:
        versions = self.versions(evaluation_id)
        return (versions[-1].version_number + 1) if versions else 1

    def search_results(self, query: str, *, plugin_id: str | None = None,
                       limit: int = 100) -> list[EvaluationResult]:
        """Search reasoning, issues, and recommendations with case-insensitive substring matching."""
        term = query.strip().casefold()
        if not term:
            raise ValueError("Search query must not be empty")
        if limit < 1:
            raise ValueError("Search limit must be positive")
        escaped = term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        statement = select(EvaluationResult).where(
            func.lower(EvaluationResult.searchable_text).like(f"%{escaped}%", escape="\\")
        )
        if plugin_id:
            statement = statement.where(EvaluationResult.plugin_id == plugin_id)
        return list(self.session.scalars(statement.order_by(EvaluationResult.id).limit(limit)))

