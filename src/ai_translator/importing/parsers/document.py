from __future__ import annotations

from typing import Protocol

from ..errors import SourceParseError
from ..types import ParsedChunk, ParsedDocument


class DocumentParser(Protocol):
    format_name: str

    def parse(self, text: str, *, title: str, source_locale: str) -> ParsedDocument: ...


class PlainTextParser:
    """Paragraph-aware text parser; concatenating chunks reproduces the input exactly."""

    format_name = "text-utf8-v1"

    def parse(self, text: str, *, title: str, source_locale: str) -> ParsedDocument:
        if not isinstance(text, str) or text == "":
            raise SourceParseError("Document text must not be empty")
        chunks = self._paragraphs(text)
        return ParsedDocument(title=title, source_locale=source_locale, source_text=text,
                              source_format=self.format_name, chunks=chunks)

    @staticmethod
    def _paragraphs(text: str) -> tuple[ParsedChunk, ...]:
        lines = text.splitlines(keepends=True)
        if not lines:
            lines = [text]
        groups: list[str] = []
        current = ""
        for line in lines:
            current += line
            if line.strip() == "":
                # Blank lines terminate the current meaningful unit and stay in
                # that unit so all original separators are retained verbatim.
                groups.append(current)
                current = ""
        if current:
            groups.append(current)
        if not groups:
            groups = [text]
        return tuple(ParsedChunk(stable_key=f"paragraph:{index}", text=part, ordinal=index)
                     for index, part in enumerate(groups, start=1))

