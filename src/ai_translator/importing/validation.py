from __future__ import annotations

from ..db.models import Project
from .errors import SourceValidationError
from .types import ParsedDocument, ParsedScripture


class SourceValidationService:
    """Validate parsed import data before any database writes occur."""

    @staticmethod
    def project(project: Project | None) -> None:
        if project is None:
            raise SourceValidationError("Project does not exist")

    @staticmethod
    def scripture(parsed: ParsedScripture) -> None:
        if not parsed.title.strip():
            raise SourceValidationError("Document title must not be empty")
        if not parsed.source_locale.strip():
            raise SourceValidationError("Source locale is required")
        if not parsed.books:
            raise SourceValidationError("At least one book is required")
        SourceValidationService._unique_numbers((b.number for b in parsed.books), "book")
        for book in parsed.books:
            SourceValidationService._unique_numbers((c.number for c in book.chapters), f"chapter in {book.title}")
            for chapter in book.chapters:
                SourceValidationService._unique_numbers(
                    (v.number for v in chapter.verses), f"verse in {book.title} {chapter.number}"
                )

    @staticmethod
    def document(parsed: ParsedDocument) -> None:
        if not parsed.title.strip():
            raise SourceValidationError("Document title must not be empty")
        if not parsed.source_locale.strip():
            raise SourceValidationError("Source locale is required")
        if not parsed.chunks:
            raise SourceValidationError("Document must contain at least one source chunk")
        if "".join(chunk.text for chunk in parsed.chunks) != parsed.source_text:
            raise SourceValidationError("Parsed chunks must preserve the original document text exactly")

    @staticmethod
    def _unique_numbers(numbers, label: str) -> None:
        values = list(numbers)
        if len(values) != len(set(values)):
            raise SourceValidationError(f"Duplicate {label} number")

