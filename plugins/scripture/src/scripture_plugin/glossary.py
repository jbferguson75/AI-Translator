from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path


@dataclass(frozen=True, slots=True)
class DoctrineTerm:
    source_term: str
    preferred_translation: str
    notes: str
    target_locale: str | None = None
    author: str | None = None

    def to_prompt_dict(self) -> dict[str, str | None]:
        return {
            "source_term": self.source_term,
            "preferred_translation": self.preferred_translation,
            "notes": self.notes,
            "target_locale": self.target_locale,
            "author": self.author,
        }


class DoctrineGlossary:
    """Plugin-owned doctrinal reference terms; core Glossary remains untouched."""

    def __init__(self, entries: tuple[DoctrineTerm, ...] = ()):
        self.entries = entries

    @classmethod
    def load(cls, path: str | Path | None = None) -> "DoctrineGlossary":
        if path is None:
            raw = json.loads(files("scripture_plugin").joinpath("data/doctrine_glossary.json").read_text("utf-8"))
        else:
            raw = json.loads(Path(path).expanduser().read_text("utf-8"))
        items = raw.get("entries") if isinstance(raw, dict) else None
        if not isinstance(items, list):
            raise ValueError("Doctrine glossary must contain an 'entries' array")
        entries: list[DoctrineTerm] = []
        for item in items:
            if not isinstance(item, dict):
                raise ValueError("Doctrine glossary entries must be objects")
            source = item.get("source_term")
            preferred = item.get("preferred_translation", "")
            notes = item.get("notes", "")
            locale = item.get("target_locale")
            author = item.get("author")
            if (not isinstance(source, str) or not source.strip()
                    or not isinstance(preferred, str) or not isinstance(notes, str)
                    or (locale is not None and not isinstance(locale, str))
                    or (author is not None and not isinstance(author, str))):
                raise ValueError(f"Invalid doctrine glossary entry: {item!r}")
            entries.append(DoctrineTerm(source.strip(), preferred.strip(), notes.strip(), locale, author))
        return cls(tuple(entries))

    def for_context(self, *, target_locale: str, author: str | None = None) -> tuple[DoctrineTerm, ...]:
        return tuple(entry for entry in self.entries
                     if (entry.target_locale is None or entry.target_locale.casefold() == target_locale.casefold())
                     and (entry.author is None or entry.author.casefold() == (author or "").casefold()))

    def as_prompt_records(self, *, target_locale: str, author: str | None = None) -> list[dict[str, str | None]]:
        return [entry.to_prompt_dict() for entry in self.for_context(target_locale=target_locale, author=author)]
