from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..db.models import TranslationMemory


class TranslationMemoryConflictError(ValueError):
    """Raised when equally ranked memories prescribe different translations."""


@dataclass(frozen=True, slots=True)
class MemoryMatch:
    entry: TranslationMemory
    similarity: float


class TranslationMemoryService:
    """Manage, search, resolve and track project-scoped translation memories."""

    def __init__(self, session: Session):
        self.session = session

    def add(self, *, project_id: UUID, source_locale: str, target_locale: str,
            source_phrase: str, preferred_translation: str, approved: bool = False,
            source_reference: str | None = None) -> TranslationMemory:
        if not source_phrase.strip() or not preferred_translation.strip():
            raise ValueError("Source phrase and preferred translation are required")
        if not source_locale.strip() or not target_locale.strip():
            raise ValueError("Source and target locales are required")
        source = source_phrase.strip()
        entry = TranslationMemory(
            project_id=project_id, source_locale=source_locale.strip(), target_locale=target_locale.strip(),
            source_text=source, target_text=preferred_translation.strip(),
            source_hash=self.hash_source(source), approved=approved,
            source_references=[source_reference] if source_reference else [],
        )
        self.session.add(entry)
        self.session.flush()
        return entry

    def list_for_project(self, project_id: UUID, *, source_locale: str | None = None,
                         target_locale: str | None = None) -> list[TranslationMemory]:
        statement = select(TranslationMemory).where(TranslationMemory.project_id == project_id)
        if source_locale:
            statement = statement.where(TranslationMemory.source_locale == source_locale)
        if target_locale:
            statement = statement.where(TranslationMemory.target_locale == target_locale)
        return list(self.session.scalars(statement.order_by(
            TranslationMemory.source_locale, TranslationMemory.target_locale,
            TranslationMemory.source_text.collate("NOCASE"), TranslationMemory.created_at,
        )))

    def lookup(self, *, project_id: UUID, source_locale: str, target_locale: str,
               source_phrase: str, limit: int = 5,
               include_unapproved: bool = False,
               entries: list[TranslationMemory] | None = None) -> list[MemoryMatch]:
        if limit < 1:
            raise ValueError("Lookup limit must be positive")
        statement = select(TranslationMemory).where(
            TranslationMemory.project_id == project_id,
            TranslationMemory.source_locale == source_locale,
            TranslationMemory.target_locale == target_locale,
        )
        if not include_unapproved:
            statement = statement.where(TranslationMemory.approved.is_(True))
        source = source_phrase.strip()
        query_hash = self.hash_source(source)
        query_tokens = self._tokens(source)
        candidates = []
        source_entries = list(self.session.scalars(statement)) if entries is None else entries
        for entry in source_entries:
            if (entry.project_id != project_id or entry.source_locale != source_locale
                    or entry.target_locale != target_locale
                    or (not include_unapproved and not entry.approved)):
                continue
            if entry.source_hash == query_hash:
                similarity = 1.0
            else:
                tokens = self._tokens(entry.source_text)
                union = query_tokens | tokens
                similarity = (len(query_tokens & tokens) / len(union)) if union else 0.0
            if similarity > 0:
                candidates.append(MemoryMatch(entry, similarity))
        candidates.sort(key=lambda match: (
            -match.similarity, not match.entry.approved, -match.entry.occurrence_count,
            match.entry.source_text.casefold(), str(match.entry.id),
        ))
        return self._resolve_same_phrase(candidates)[:limit]

    def conflicts(self, project_id: UUID, *, source_locale: str | None = None,
                  target_locale: str | None = None) -> list[list[TranslationMemory]]:
        groups: dict[tuple[str, str, str], list[TranslationMemory]] = defaultdict(list)
        for entry in self.list_for_project(project_id, source_locale=source_locale, target_locale=target_locale):
            key = (entry.source_locale, entry.target_locale, entry.source_text.casefold())
            groups[key].append(entry)
        return [sorted(group, key=lambda entry: (-entry.approved, -entry.occurrence_count, str(entry.id)))
                for group in groups.values()
                if len({entry.target_text.casefold() for entry in group}) > 1]

    def set_approved(self, entry_id: UUID, approved: bool) -> TranslationMemory:
        entry = self.session.get(TranslationMemory, entry_id)
        if entry is None:
            raise ValueError(f"Translation memory {entry_id} does not exist")
        entry.approved = approved
        self.session.flush()
        return entry

    def remove(self, entry_id: UUID) -> bool:
        entry = self.session.get(TranslationMemory, entry_id)
        if entry is None:
            return False
        self.session.delete(entry)
        return True

    def record_usage(self, usage: dict[UUID, set[str]]) -> None:
        for entry_id, references in usage.items():
            entry = self.session.get(TranslationMemory, entry_id)
            if entry is None:
                continue
            entry.occurrence_count += len(references)
            entry.source_references = sorted(set(entry.source_references or []) | references)

    @staticmethod
    def hash_source(source: str) -> str:
        return hashlib.sha256(source.encode("utf-8")).hexdigest()

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return set(re.findall(r"\w+", text.casefold(), flags=re.UNICODE))

    @staticmethod
    def _resolve_same_phrase(matches: list[MemoryMatch]) -> list[MemoryMatch]:
        """Keep one preferred value per exact source phrase, rejecting top ties."""
        selected: dict[str, MemoryMatch] = {}
        for match in matches:
            key = match.entry.source_text.casefold()
            previous = selected.get(key)
            if previous is None:
                selected[key] = match
                continue
            current_rank = (match.similarity, match.entry.approved, match.entry.occurrence_count)
            previous_rank = (previous.similarity, previous.entry.approved, previous.entry.occurrence_count)
            if match.entry.target_text.casefold() == previous.entry.target_text.casefold():
                continue
            if current_rank == previous_rank:
                raise TranslationMemoryConflictError(
                    f"Translation memory conflict for {match.entry.source_text!r}: equally ranked entries "
                    "have different preferred translations; approve the intended entry and unapprove or remove "
                    "the competing entry"
                )
            if current_rank > previous_rank:
                selected[key] = match
        return sorted(selected.values(), key=lambda match: (
            -match.similarity, match.entry.source_text.casefold(), str(match.entry.id)
        ))
