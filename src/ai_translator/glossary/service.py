from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..db.models import Glossary, GlossaryEntry
from .errors import GlossaryConflictError


@dataclass(frozen=True, slots=True)
class GlossaryConflict:
    source_term: str
    priority: int
    entries: tuple[GlossaryEntry, ...]

    @property
    def translations(self) -> tuple[str, ...]:
        return tuple(sorted({entry.target_term for entry in self.entries}, key=str.casefold))


class GlossaryService:
    """Project-scoped terminology management and deterministic resolution."""

    def __init__(self, session: Session):
        self.session = session

    def create(self, *, project_id: UUID, name: str, source_locale: str,
               target_locale: str) -> Glossary:
        if not name.strip() or not source_locale.strip() or not target_locale.strip():
            raise ValueError("Glossary name and locales are required")
        glossary = Glossary(project_id=project_id, name=name.strip(),
                            source_locale=source_locale.strip(), target_locale=target_locale.strip())
        self.session.add(glossary)
        self.session.flush()
        return glossary

    def list_for_project(self, project_id: UUID) -> list[Glossary]:
        return list(self.session.scalars(select(Glossary).where(
            Glossary.project_id == project_id
        ).order_by(Glossary.name, Glossary.source_locale, Glossary.target_locale)))

    def add_entry(self, *, glossary_id: UUID, source_term: str,
                  preferred_translation: str, priority: int = 0,
                  notes: str | None = None) -> GlossaryEntry:
        if not source_term.strip() or not preferred_translation.strip():
            raise ValueError("Source term and preferred translation are required")
        if priority < 0:
            raise ValueError("Priority must be zero or greater")
        exists = self.session.scalar(select(Glossary.id).where(Glossary.id == glossary_id))
        if exists is None:
            raise ValueError(f"Glossary {glossary_id} does not exist")
        duplicate = self.session.scalar(select(GlossaryEntry.id).where(
            GlossaryEntry.glossary_id == glossary_id,
            func.lower(GlossaryEntry.source_term) == source_term.strip().casefold(),
        ))
        if duplicate is not None:
            raise ValueError("This source term already exists in that glossary; update its entry instead")
        entry = GlossaryEntry(glossary_id=glossary_id, source_term=source_term.strip(),
                              target_term=preferred_translation.strip(), priority=priority,
                              notes=notes.strip() if notes else None)
        self.session.add(entry)
        self.session.flush()
        return entry

    def update_entry(self, entry_id: UUID, *, preferred_translation: str | None = None,
                     priority: int | None = None, notes: str | None = None) -> GlossaryEntry:
        entry = self.session.get(GlossaryEntry, entry_id)
        if entry is None:
            raise ValueError(f"Glossary entry {entry_id} does not exist")
        if preferred_translation is not None:
            if not preferred_translation.strip():
                raise ValueError("Preferred translation cannot be blank")
            entry.target_term = preferred_translation.strip()
        if priority is not None:
            if priority < 0:
                raise ValueError("Priority must be zero or greater")
            entry.priority = priority
        if notes is not None:
            entry.notes = notes.strip() or None
        self.session.flush()
        return entry

    def entries(self, glossary_id: UUID) -> list[GlossaryEntry]:
        return list(self.session.scalars(select(GlossaryEntry).where(
            GlossaryEntry.glossary_id == glossary_id
        ).order_by(GlossaryEntry.source_term.collate("NOCASE"), GlossaryEntry.priority.desc())))

    def remove_entry(self, entry_id: UUID) -> bool:
        entry = self.session.get(GlossaryEntry, entry_id)
        if entry is None:
            return False
        self.session.delete(entry)
        return True

    def remove(self, glossary_id: UUID) -> bool:
        glossary = self.session.get(Glossary, glossary_id)
        if glossary is None:
            return False
        # Glossary entries are managed data, not append-only source records.
        for entry in list(glossary.entries):
            self.session.delete(entry)
        self.session.delete(glossary)
        return True

    def conflicts(self, *, project_id: UUID, source_locale: str,
                   target_locale: str) -> list[GlossaryConflict]:
        rows = self._entries_for_project(project_id, source_locale, target_locale)
        groups: dict[str, list[GlossaryEntry]] = defaultdict(list)
        for entry in rows:
            groups[entry.source_term.casefold()].append(entry)
        result = []
        for group in groups.values():
            priority = max(entry.priority for entry in group)
            winners = [entry for entry in group if entry.priority == priority]
            if len({entry.target_term.casefold() for entry in winners}) > 1:
                result.append(GlossaryConflict(winners[0].source_term, priority, tuple(winners)))
        return sorted(result, key=lambda item: item.source_term.casefold())

    def resolve_for_text(self, *, project_id: UUID, source_locale: str,
                         target_locale: str, source_text: str,
                         entries: list[GlossaryEntry] | None = None) -> list[dict[str, object]]:
        folded = source_text.casefold()
        glossary_entries = (entries if entries is not None else
                            self._entries_for_project(project_id, source_locale, target_locale))
        matched = [entry for entry in glossary_entries
                   if self._contains_term(folded, entry.source_term)]
        groups: dict[str, list[GlossaryEntry]] = defaultdict(list)
        for entry in matched:
            groups[entry.source_term.casefold()].append(entry)
        resolved: list[dict[str, object]] = []
        for group in groups.values():
            priority = max(entry.priority for entry in group)
            winners = [entry for entry in group if entry.priority == priority]
            translations = {entry.target_term.casefold() for entry in winners}
            if len(translations) > 1:
                names = sorted({entry.glossary.name for entry in winners})
                raise GlossaryConflictError(
                    f"Glossary conflict for {winners[0].source_term!r} at priority {priority} "
                    f"in {', '.join(names)}; adjust priorities or translations"
                )
            winner = sorted(winners, key=lambda entry: (entry.glossary.name.casefold(), str(entry.id)))[0]
            resolved.append({
                "id": str(winner.id), "source": winner.source_term, "target": winner.target_term,
                "notes": winner.notes, "priority": winner.priority, "glossary": winner.glossary.name,
            })
        return sorted(resolved, key=lambda item: (-len(str(item["source"])), str(item["source"]).casefold()))

    def record_usage(self, usage: dict[UUID, set[str]]) -> None:
        for entry_id, references in usage.items():
            entry = self.session.get(GlossaryEntry, entry_id)
            if entry is None:
                continue
            entry.occurrence_count += len(references)
            known = set(entry.source_references or [])
            entry.source_references = sorted(known | references)

    def _entries_for_project(self, project_id: UUID, source_locale: str,
                             target_locale: str) -> list[GlossaryEntry]:
        return list(self.session.scalars(
            select(GlossaryEntry).join(Glossary).where(
                Glossary.project_id == project_id,
                Glossary.source_locale == source_locale,
                Glossary.target_locale == target_locale,
            ).order_by(GlossaryEntry.source_term, GlossaryEntry.priority.desc(), Glossary.name)
        ))

    def entries_for_scope(self, project_id: UUID, source_locale: str,
                          target_locale: str) -> list[GlossaryEntry]:
        """Load a locale-pair glossary snapshot for reuse across source chunks."""
        return self._entries_for_project(project_id, source_locale, target_locale)

    @staticmethod
    def _contains_term(folded_source: str, source_term: str) -> bool:
        term = source_term.casefold().strip()
        if not term:
            return False
        return re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", folded_source, flags=re.UNICODE) is not None
