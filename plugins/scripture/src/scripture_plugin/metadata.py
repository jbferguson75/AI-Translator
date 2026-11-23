from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path


@dataclass(frozen=True, slots=True)
class AuthorProfile:
    name: str
    voice_profile: str
    books: tuple[str, ...]


class ScriptureMetadata:
    """Book-to-author mapping and editorial voice cues, separate from core schema."""

    def __init__(self, profiles: dict[str, AuthorProfile]):
        self.profiles = profiles
        self._book_map = {
            self._normalize(book): profile
            for profile in profiles.values()
            for book in profile.books
        }

    @classmethod
    def load(cls, path: str | Path | None = None) -> "ScriptureMetadata":
        if path is None:
            raw = json.loads(files("scripture_plugin").joinpath("data/authors.json").read_text("utf-8"))
        else:
            raw = json.loads(Path(path).expanduser().read_text("utf-8"))
        authors = raw.get("authors") if isinstance(raw, dict) else None
        if not isinstance(authors, dict):
            raise ValueError("Author metadata must contain an 'authors' object")
        profiles: dict[str, AuthorProfile] = {}
        for name, item in authors.items():
            if (not isinstance(name, str) or not isinstance(item, dict)
                    or not isinstance(item.get("voice_profile"), str)
                    or not isinstance(item.get("books"), list)
                    or any(not isinstance(book, str) or not book.strip() for book in item["books"])):
                raise ValueError(f"Invalid author metadata entry: {name!r}")
            profiles[name] = AuthorProfile(name, item["voice_profile"], tuple(item["books"]))
        return cls(profiles)

    def author_for_book(self, book: str | None) -> AuthorProfile | None:
        if not book:
            return None
        direct = self._book_map.get(self._normalize(book))
        if direct:
            return direct
        # Number-prefix normalization also accepts "First Nephi" and "1 Nephi".
        aliases = {"first": "1", "second": "2", "third": "3", "fourth": "4"}
        normalized = self._normalize(book)
        for word, digit in aliases.items():
            normalized = normalized.replace(word + " ", digit + " ", 1)
        return self._book_map.get(normalized)

    @staticmethod
    def _normalize(value: str) -> str:
        return " ".join(value.casefold().replace(".", "").split())
