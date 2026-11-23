import json

import pytest
from sqlalchemy import create_engine, select, event
from sqlalchemy.orm import Session

from ai_translator.db.base import Base
from ai_translator.db.models import Book, Chapter, Chunk, Document, Project, Verse
from ai_translator.importing.errors import SourceParseError, SourceValidationError
from ai_translator.importing.services import SourceImportService


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:")

    @event.listens_for(engine, "connect")
    def enable_foreign_keys(connection, _record):
        connection.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(engine)
    with Session(engine) as db:
        project = Project(name="Import Test", default_source_locale="he")
        db.add(project)
        db.commit()
        yield db, project
    engine.dispose()


def test_scripture_import_preserves_verse_text_and_hierarchy(session, tmp_path):
    db, project = session
    verse_text = "  בְּרֵאשִׁית\nלְשׁוֹן  "
    body = {
        "title": "Genesis",
        "source_locale": "he",
        "books": [{"number": 1, "title": "Genesis", "chapters": [{
            "number": 1, "verses": [{"number": 1, "text": verse_text},
                                       {"number": 2, "text": "In the beginning"}],
        }]}],
    }
    raw = json.dumps(body, ensure_ascii=False, indent=2).encode("utf-8")
    path = tmp_path / "genesis.json"
    path.write_bytes(raw)

    receipt = SourceImportService(db).import_scripture(project_id=project.id, path=path)
    db.commit()

    document = db.get(Document, receipt.document_id)
    assert document.source_text == raw.decode("utf-8")
    assert document.source_hash == receipt.source_hash
    book = db.scalar(select(Book).where(Book.document_id == document.id))
    chapter = db.scalar(select(Chapter).where(Chapter.book_id == book.id))
    verses = list(db.scalars(select(Verse).where(Verse.chapter_id == chapter.id).order_by(Verse.ordinal)))
    chunks = list(db.scalars(select(Chunk).where(Chunk.document_id == document.id).order_by(Chunk.ordinal)))
    assert [verse.source_text for verse in verses] == [verse_text, "In the beginning"]
    assert [chunk.source_text for chunk in chunks] == [verse_text, "In the beginning"]
    assert chunks[0].verse_id == verses[0].id
    assert chunks[0].stable_key == "1:1:1"


def test_duplicate_scripture_number_is_rejected_before_writing(session, tmp_path):
    db, project = session
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps({
        "source_locale": "en", "books": [{"title": "Book", "chapters": [
            {"number": 1, "verses": [{"number": 1, "text": "one"},
                                      {"number": 1, "text": "duplicate"}]}
        ]}]
    }), encoding="utf-8")
    with pytest.raises(SourceValidationError, match="Duplicate verse"):
        SourceImportService(db).import_scripture(project_id=project.id, path=path)
    assert db.scalar(select(Document).where(Document.project_id == project.id)) is None


def test_scripture_import_accepts_book_of_mormon_schema_and_keeps_references(session, tmp_path):
    db, project = session
    payload = {
        "title": "The Book of Mormon",
        "books": [{
            "book": "1 Nephi", "full_title": "The First Book of Nephi", "lds_slug": "1-nephi",
            "chapters": [{"chapter": 1, "reference": "1 Nephi 1", "verses": [
                {"reference": "1 Nephi 1:1", "text": "  exact text\n", "verse": 1}
            ]}],
        }],
    }
    path = tmp_path / "bom.json"
    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    path.write_bytes(raw)
    receipt = SourceImportService(db).import_scripture(
        project_id=project.id, path=path, source_locale="en"
    )
    db.commit()

    document = db.get(Document, receipt.document_id)
    book = db.scalar(select(Book).where(Book.document_id == document.id))
    chapter = db.scalar(select(Chapter).where(Chapter.book_id == book.id))
    chunk = db.scalar(select(Chunk).where(Chunk.document_id == document.id))
    assert book.title == "1 Nephi"
    assert book.locator["full_title"] == "The First Book of Nephi"
    assert chapter.locator["reference"] == "1 Nephi 1"
    assert chunk.source_text == "  exact text\n"
    assert chunk.locator["reference"] == "1 Nephi 1:1"


def test_new_project_defaults_scripture_import_locale_to_english(session, tmp_path):
    db, _existing_project = session
    project = Project(name="English default")
    db.add(project)
    db.flush()
    assert project.default_source_locale == "en"
    path = tmp_path / "one-verse.json"
    path.write_text(json.dumps({"books": [{"title": "Test", "chapters": [
        {"verses": [{"text": "Exact text"}]}
    ]}]}), encoding="utf-8")

    receipt = SourceImportService(db).import_scripture(project_id=project.id, path=path)
    db.commit()
    assert db.get(Document, receipt.document_id).source_locale == "en"


def test_plain_text_import_chunks_reconstruct_source_exactly(session, tmp_path):
    db, project = session
    source = "First paragraph.\r\nStill first.\r\n\r\nSecond paragraph\n"
    path = tmp_path / "novel.txt"
    path.write_bytes(source.encode("utf-8"))
    receipt = SourceImportService(db).import_document(
        project_id=project.id, path=path, source_locale="en", domain="literary"
    )
    db.commit()
    document = db.get(Document, receipt.document_id)
    chunks = list(db.scalars(select(Chunk).where(Chunk.document_id == document.id).order_by(Chunk.ordinal)))
    assert document.source_text == source
    assert "".join(chunk.source_text for chunk in chunks) == source
    assert len(chunks) == 2
    assert [chunk.ordinal for chunk in chunks] == [1, 2]


def test_non_utf8_document_is_rejected_without_normalization(session, tmp_path):
    db, project = session
    path = tmp_path / "bad.txt"
    path.write_bytes(b"\xff\xfe")
    with pytest.raises(SourceParseError, match="UTF-8"):
        SourceImportService(db).import_document(project_id=project.id, path=path, source_locale="en")

