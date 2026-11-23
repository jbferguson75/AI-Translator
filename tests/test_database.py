from sqlalchemy import create_engine, event
from sqlalchemy.exc import IntegrityError
import pytest
from sqlalchemy.orm import Session

from ai_translator.db.base import Base
from ai_translator.db.models import (
    Chunk, Document, Evaluation, EvaluationVersion, Project, Translation,
    TranslationVersion,
)
from ai_translator.db.repositories import EvaluationRepository, TranslationRepository


def make_db():
    engine = create_engine("sqlite:///:memory:")
    @event.listens_for(engine, "connect")
    def fk_on(dbapi_connection, _):
        dbapi_connection.execute("PRAGMA foreign_keys=ON")
    Base.metadata.create_all(engine)
    return engine


def test_project_document_and_translation_versions():
    engine = make_db()
    with Session(engine) as session, session.begin():
        project = Project(name="Study")
        session.add(project)
        session.flush()
        doc = Document(project_id=project.id, title="Genesis", domain="scripture",
                       source_locale="he", source_hash="a" * 64, source_text="Source")
        session.add(doc)
        session.flush()
        translation = Translation(document_id=doc.id, target_locale="en")
        session.add(translation)
        session.flush()
        repo = TranslationRepository(session)
        for version in (1, 2):
            repo.add(TranslationVersion(translation_id=translation.id, version_number=version,
                                        author_kind="model", status="completed"))
        session.flush()
        assert repo.next_version_number(translation.id) == 3
        assert [v.version_number for v in repo.versions(translation.id)] == [1, 2]


def test_evaluations_are_independently_versioned():
    engine = make_db()
    with Session(engine) as session, session.begin():
        project = Project(name="Study")
        session.add(project)
        session.flush()
        doc = Document(project_id=project.id, title="Book", source_locale="fr",
                       source_hash="b" * 64, source_text="bonjour")
        session.add(doc)
        session.flush()
        translation = Translation(document_id=doc.id, target_locale="en")
        session.add(translation)
        session.flush()
        tv = TranslationVersion(translation_id=translation.id, version_number=1)
        evaluation = Evaluation(translation_id=translation.id, evaluator_name="reviewer model")
        session.add_all([tv, evaluation])
        session.flush()
        repo = EvaluationRepository(session)
        for version in (1, 2):
            repo.add(EvaluationVersion(evaluation_id=evaluation.id, translation_version_id=tv.id,
                                       version_number=version, run_manifest={"seed": version}))
        session.flush()
        assert repo.next_version_number(evaluation.id) == 3
        assert len(repo.versions(evaluation.id)) == 2


def test_source_chunks_cannot_be_updated_or_deleted():
    engine = make_db()
    with Session(engine) as session, session.begin():
        project = Project(name="Study")
        session.add(project)
        session.flush()
        doc = Document(project_id=project.id, title="Book", source_locale="en",
                       source_hash="c" * 64, source_text="unchanged")
        session.add(doc)
        session.flush()
        chunk = Chunk(document_id=doc.id, stable_key="p1", ordinal=1,
                      source_text="original", source_hash="d" * 64)
        session.add(chunk)
        session.flush()
        chunk_id = chunk.id
    with Session(engine) as session:
        chunk = session.get(Chunk, chunk_id)
        chunk.source_text = "changed"
        with pytest.raises(ValueError, match="immutable"):
            session.commit()
        session.rollback()


def test_unique_translation_version_number():
    engine = make_db()
    with Session(engine) as session, session.begin():
        project = Project(name="Study")
        session.add(project)
        session.flush()
        doc = Document(project_id=project.id, title="Book", source_locale="en",
                       source_hash="e" * 64, source_text="source")
        session.add(doc)
        session.flush()
        translation = Translation(document_id=doc.id, target_locale="es")
        session.add(translation)
        session.flush()
        session.add_all([
            TranslationVersion(translation_id=translation.id, version_number=1),
            TranslationVersion(translation_id=translation.id, version_number=1),
        ])
        try:
            session.flush()
            assert False, "duplicate version number should fail"
        except IntegrityError:
            session.rollback()

