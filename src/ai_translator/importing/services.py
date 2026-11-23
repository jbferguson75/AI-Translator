from __future__ import annotations

import hashlib
from pathlib import Path

from sqlalchemy.orm import Session

from ..db.models import Book, Chapter, Chunk, Document, Project, Verse
from .errors import SourceParseError
from .parsers import ParserRegistry, ScriptureJSONParser
from .types import ImportReceipt, ParsedDocument, SourceFile
from .validation import SourceValidationService


class SourceImportService:
    """Parse, validate, and atomically persist immutable source snapshots."""

    def __init__(self, session: Session, *, registry: ParserRegistry | None = None,
                 scripture_parser: ScriptureJSONParser | None = None,
                 validator: SourceValidationService | None = None):
        self.session = session
        self.registry = registry or ParserRegistry()
        self.scripture_parser = scripture_parser or ScriptureJSONParser()
        self.validator = validator or SourceValidationService()

    @staticmethod
    def read_file(path: str | Path) -> SourceFile:
        source_path = Path(path).expanduser()
        try:
            raw_bytes = source_path.read_bytes()
        except OSError as exc:
            raise SourceParseError(f"Cannot read source file {source_path}: {exc}") from exc
        try:
            text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise SourceParseError("Source files must be valid UTF-8") from exc
        return SourceFile(path=source_path, raw_bytes=raw_bytes, text=text)

    def import_scripture(self, *, project_id, path: str | Path, source_locale: str | None = None,
                         title: str | None = None) -> ImportReceipt:
        project = self.session.get(Project, project_id)
        self.validator.project(project)
        source_file = self.read_file(path)
        parsed = self.scripture_parser.parse(
            source_file.text, title=title,
            source_locale=source_locale or project.default_source_locale,
        )
        self.validator.scripture(parsed)
        document = Document(
            project_id=project.id, title=parsed.title, domain="scripture",
            source_locale=parsed.source_locale, source_hash=self._hash(source_file.raw_bytes),
            source_text=source_file.text, source_format=self.scripture_parser.format_name,
        )
        self.session.add(document)
        self.session.flush()
        ordinal = 0
        for book in parsed.books:
            book_row = Book(document_id=document.id, title=book.title, ordinal=book.number,
                            locator={"book_number": book.number, **(book.metadata or {})})
            self.session.add(book_row)
            self.session.flush()
            for chapter in book.chapters:
                chapter_locator = {"book": book.title, "chapter": chapter.number}
                if chapter.reference is not None:
                    chapter_locator["reference"] = chapter.reference
                chapter_row = Chapter(book_id=book_row.id, ordinal=chapter.number,
                                      locator=chapter_locator)
                self.session.add(chapter_row)
                self.session.flush()
                for verse in chapter.verses:
                    verse_row = Verse(chapter_id=chapter_row.id, ordinal=verse.number,
                                      source_text=verse.text, source_hash=self._hash(verse.text.encode("utf-8")))
                    self.session.add(verse_row)
                    self.session.flush()
                    ordinal += 1
                    stable_key = f"{book.number}:{chapter.number}:{verse.number}"
                    self.session.add(Chunk(
                        document_id=document.id, chapter_id=chapter_row.id, verse_id=verse_row.id,
                        stable_key=stable_key, ordinal=ordinal, source_text=verse.text,
                        source_hash=self._hash(verse.text.encode("utf-8")),
                        locator={"book": book.title, "book_number": book.number,
                                 "chapter": chapter.number, "verse": verse.number,
                                 **({"reference": verse.reference} if verse.reference is not None else {})},
                    ))
        # The caller controls commit/rollback, allowing composition with a unit of work.
        return ImportReceipt(document.id, document.title, document.source_hash, ordinal)

    def import_document(self, *, project_id, path: str | Path, source_locale: str,
                        title: str | None = None, format_name: str | None = None,
                        domain: str = "general") -> ImportReceipt:
        project = self.session.get(Project, project_id)
        self.validator.project(project)
        source_file = self.read_file(path)
        parser = self.registry.by_name(format_name) if format_name else self.registry.by_suffix(source_file.path.suffix)
        parsed = parser.parse(source_file.text, title=title or source_file.path.stem,
                              source_locale=source_locale)
        self.validator.document(parsed)
        document = Document(
            project_id=project.id, title=parsed.title, domain=domain,
            source_locale=parsed.source_locale, source_hash=self._hash(source_file.raw_bytes),
            source_text=parsed.source_text, source_format=parsed.source_format,
        )
        self.session.add(document)
        self.session.flush()
        for chunk in parsed.chunks:
            self.session.add(Chunk(
                document_id=document.id, stable_key=chunk.stable_key, ordinal=chunk.ordinal,
                source_text=chunk.text, source_hash=self._hash(chunk.text.encode("utf-8")),
                locator={"kind": "paragraph", "ordinal": chunk.ordinal},
            ))
        return ImportReceipt(document.id, document.title, document.source_hash, len(parsed.chunks))

    @staticmethod
    def _hash(value: bytes) -> str:
        return hashlib.sha256(value).hexdigest()

