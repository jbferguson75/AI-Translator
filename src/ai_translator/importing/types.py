from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import UUID


@dataclass(frozen=True, slots=True)
class ParsedVerse:
    number: int
    text: str
    reference: str | None = None


@dataclass(frozen=True, slots=True)
class ParsedChapter:
    number: int
    verses: tuple[ParsedVerse, ...]
    reference: str | None = None


@dataclass(frozen=True, slots=True)
class ParsedBook:
    title: str
    number: int
    chapters: tuple[ParsedChapter, ...]
    metadata: dict | None = None


@dataclass(frozen=True, slots=True)
class ParsedScripture:
    title: str
    source_locale: str
    books: tuple[ParsedBook, ...]


@dataclass(frozen=True, slots=True)
class ParsedChunk:
    stable_key: str
    text: str
    ordinal: int


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    title: str
    source_locale: str
    source_text: str
    source_format: str
    chunks: tuple[ParsedChunk, ...]


@dataclass(frozen=True, slots=True)
class ImportReceipt:
    document_id: UUID
    title: str
    source_hash: str
    source_units: int


@dataclass(frozen=True, slots=True)
class SourceFile:
    path: Path
    raw_bytes: bytes
    text: str

