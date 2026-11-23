from __future__ import annotations

from ..errors import SourceParseError
from .document import DocumentParser, PlainTextParser


class ParserRegistry:
    """Format-to-parser registry; add parsers without changing import services."""

    def __init__(self, parsers: list[DocumentParser] | None = None):
        self._parsers: dict[str, DocumentParser] = {}
        for parser in parsers or [PlainTextParser()]:
            self.register(parser)

    def register(self, parser: DocumentParser) -> None:
        if parser.format_name in self._parsers:
            raise ValueError(f"Parser already registered for {parser.format_name!r}")
        self._parsers[parser.format_name] = parser

    def by_name(self, format_name: str) -> DocumentParser:
        try:
            return self._parsers[format_name]
        except KeyError as exc:
            supported = ", ".join(sorted(self._parsers)) or "none"
            raise SourceParseError(f"Unsupported document format {format_name!r}; supported: {supported}") from exc

    def by_suffix(self, suffix: str) -> DocumentParser:
        normalized = suffix.lower().lstrip(".")
        if normalized in {"txt", "text"}:
            return self.by_name("text-utf8-v1")
        raise SourceParseError(f"No document parser registered for .{normalized}")

