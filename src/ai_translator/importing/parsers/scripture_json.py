from __future__ import annotations

import json
from typing import Any

from ..errors import SourceParseError
from ..types import ParsedBook, ParsedChapter, ParsedScripture, ParsedVerse


class ScriptureJSONParser:
    """Parse the documented structured scripture JSON format without trimming verse text."""

    format_name = "scripture-json-v1"

    def parse(self, text: str, *, title: str | None = None,
              source_locale: str | None = None) -> ParsedScripture:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise SourceParseError(f"Invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}") from exc
        if not isinstance(payload, dict):
            raise SourceParseError("Scripture JSON root must be an object")
        raw_books = payload.get("books")
        if raw_books is None and "book" in payload:
            raw_books = [payload["book"]]
        if not isinstance(raw_books, list):
            raise SourceParseError("Scripture JSON must include a 'books' array")
        parsed_books: list[ParsedBook] = []
        for book_index, raw_book in enumerate(raw_books, start=1):
            location = f"books[{book_index - 1}]"
            self._require_object(raw_book, location)
            book_title = raw_book.get("title", raw_book.get("name", raw_book.get("book")))
            if not isinstance(book_title, str) or not book_title.strip():
                raise SourceParseError(f"{location}.title must be a non-empty string")
            book_number = self._number(raw_book.get("number", book_index), f"{location}.number")
            raw_chapters = raw_book.get("chapters")
            if not isinstance(raw_chapters, list) or not raw_chapters:
                raise SourceParseError(f"{location}.chapters must be a non-empty array")
            chapters: list[ParsedChapter] = []
            for chapter_index, raw_chapter in enumerate(raw_chapters, start=1):
                chapter_location = f"{location}.chapters[{chapter_index - 1}]"
                self._require_object(raw_chapter, chapter_location)
                chapter_number = self._number(
                    raw_chapter.get("number", raw_chapter.get("chapter", chapter_index)),
                    f"{chapter_location}.chapter",
                )
                raw_verses = raw_chapter.get("verses")
                if not isinstance(raw_verses, list) or not raw_verses:
                    raise SourceParseError(f"{chapter_location}.verses must be a non-empty array")
                verses: list[ParsedVerse] = []
                for verse_index, raw_verse in enumerate(raw_verses, start=1):
                    verse_location = f"{chapter_location}.verses[{verse_index - 1}]"
                    self._require_object(raw_verse, verse_location)
                    verse_number = self._number(
                        raw_verse.get("number", raw_verse.get("verse", verse_index)),
                        f"{verse_location}.verse",
                    )
                    verse_text = raw_verse.get("text", raw_verse.get("original_text"))
                    if not isinstance(verse_text, str) or verse_text == "":
                        raise SourceParseError(f"{verse_location}.text must be a non-empty string")
                    reference = raw_verse.get("reference")
                    if reference is not None and not isinstance(reference, str):
                        raise SourceParseError(f"{verse_location}.reference must be a string")
                    verses.append(ParsedVerse(number=verse_number, text=verse_text, reference=reference))
                chapter_reference = raw_chapter.get("reference")
                if chapter_reference is not None and not isinstance(chapter_reference, str):
                    raise SourceParseError(f"{chapter_location}.reference must be a string")
                chapters.append(ParsedChapter(number=chapter_number, verses=tuple(verses),
                                              reference=chapter_reference))
            metadata = {key: value for key, value in raw_book.items()
                        if key not in {"book", "title", "name", "number", "chapters"}}
            parsed_books.append(ParsedBook(title=book_title, number=book_number,
                                           chapters=tuple(chapters), metadata=metadata))
        parsed_title = title if title is not None else payload.get("title", "Scripture import")
        parsed_locale = source_locale if source_locale is not None else payload.get("source_locale", "")
        if not isinstance(parsed_title, str):
            raise SourceParseError("title must be a string")
        if not isinstance(parsed_locale, str):
            raise SourceParseError("source_locale must be a string")
        result = ParsedScripture(title=parsed_title, source_locale=parsed_locale,
                                 books=tuple(parsed_books))
        return result

    @staticmethod
    def _require_object(value: Any, location: str) -> None:
        if not isinstance(value, dict):
            raise SourceParseError(f"{location} must be an object")

    @staticmethod
    def _number(value: Any, location: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise SourceParseError(f"{location} must be a positive integer")
        return value

