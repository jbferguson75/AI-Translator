import json
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, select, event
from sqlalchemy.orm import Session

from ai_translator.db.base import Base
from ai_translator.db.models import (
    Chunk, Document, Glossary, GlossaryEntry, ModelRun, Project,
    PromptVersion, Translation, TranslationMemory, TranslationSegment,
    TranslationVersion,
)
from ai_translator.translation.config import ModelConfig, TranslationConfig
from ai_translator.translation.errors import OllamaError, TranslationConfigurationError
from ai_translator.translation.ollama import Generation, OllamaClient
from ai_translator.translation.service import TranslationService


@pytest.fixture
def translation_db():
    engine = create_engine("sqlite:///:memory:")

    @event.listens_for(engine, "connect")
    def foreign_keys(connection, _record):
        connection.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(engine)
    session = Session(engine, expire_on_commit=False)
    project = Project(name="Translation Tests", default_source_locale="en")
    session.add(project)
    session.flush()
    document = Document(project_id=project.id, title="Source", domain="scripture",
                        source_locale="en", source_hash="a" * 64,
                        source_text='{"immutable":"source"}', source_format="scripture-json-v1")
    session.add(document)
    session.flush()
    chunk = Chunk(document_id=document.id, stable_key="1:1:1", ordinal=1,
                  source_text="  In the beginning  ", source_hash="b" * 64,
                  locator={"reference": "Book 1:1"})
    session.add(chunk)
    session.commit()
    yield session, project, document, chunk
    session.close()
    engine.dispose()


def make_config(*, prompt_file: Path | None = None, target_locale="es",
                task_kind="translation", target_reading_level=None):
    models = {
        "qwen": ModelConfig("qwen", "qwen3:8b", {"temperature": 0.1}),
        "mistral": ModelConfig("mistral", "mistral:latest"),
        "gemma": ModelConfig("gemma", "gemma3:4b"),
    }
    return TranslationConfig(
        base_url="http://localhost:11434", timeout_seconds=1,
        candidate_models=("qwen", "mistral", "gemma"), models=models,
        target_locale=target_locale, task_kind=task_kind,
        target_reading_level=target_reading_level,
        temperature=0.2, prompt_file=prompt_file,
    )


class FakeOllama:
    base_url = "http://localhost:11434"

    def __init__(self, *, fail_model: str | None = None):
        self.fail_model = fail_model
        self.prompts = []

    def list_models(self):
        return [{"name": "qwen3:8b", "digest": "sha-qwen"},
                {"name": "mistral:latest", "digest": "sha-mistral"},
                {"name": "gemma3:4b", "digest": "sha-gemma"}]

    def generate(self, *, model, prompt, options=None):
        self.prompts.append((model, prompt, options))
        if model == self.fail_model:
            raise OllamaError("model unavailable")
        return Generation(model, f"{model} translation", "2026-11-23T00:00:00Z", 100, 12, 5)


def test_multiple_models_create_distinct_immutable_candidates(translation_db):
    session, _project, document, chunk = translation_db
    client = FakeOllama()
    candidates = TranslationService(session, client).translate_document(
        document_id=document.id, target_locale="es", config=make_config()
    )
    assert [candidate.status for candidate in candidates] == ["completed"] * 3
    assert [candidate.version_number for candidate in candidates] == [1, 2, 3]
    versions = list(session.scalars(select(TranslationVersion).order_by(TranslationVersion.version_number)))
    assert len(versions) == 3
    assert len(list(session.scalars(select(TranslationSegment)))) == 3
    assert len(list(session.scalars(select(ModelRun)))) == 3
    assert len(list(session.scalars(select(PromptVersion)))) == 1
    assert all(version.run_manifest["prompt_template_hash"] for version in versions)
    assert all(chunk.source_text == "  In the beginning  " for _ in versions)


def test_repeated_translation_adds_versions_without_overwriting(translation_db):
    session, _project, document, _chunk = translation_db
    service = TranslationService(session, FakeOllama())
    config = make_config()
    service.translate_document(document_id=document.id, target_locale="es", config=config,
                               model_selection=["qwen"])
    service.translate_document(document_id=document.id, target_locale="es", config=config,
                               model_selection=["qwen"])
    versions = list(session.scalars(select(TranslationVersion).order_by(TranslationVersion.version_number)))
    assert [version.version_number for version in versions] == [1, 2]
    assert versions[0].id != versions[1].id
    segments = list(session.scalars(select(TranslationSegment).order_by(TranslationSegment.id)))
    assert len(segments) == 2


def test_failed_model_is_recorded_and_does_not_block_other_candidates(translation_db):
    session, _project, document, _chunk = translation_db
    candidates = TranslationService(session, FakeOllama(fail_model="mistral:latest")).translate_document(
        document_id=document.id, target_locale="es", config=make_config(),
        model_selection=["qwen", "mistral"],
    )
    assert [candidate.status for candidate in candidates] == ["completed", "failed"]
    assert candidates[1].error == "model unavailable"
    versions = list(session.scalars(select(TranslationVersion).order_by(TranslationVersion.version_number)))
    assert versions[1].run_manifest["error"] == "model unavailable"
    assert versions[1].content_hash is None


def test_glossary_and_memory_are_injected_into_prompt_and_audit_manifest(translation_db):
    session, project, document, _chunk = translation_db
    glossary = Glossary(project_id=project.id, name="Terms", source_locale="en", target_locale="es")
    session.add(glossary)
    session.flush()
    session.add(GlossaryEntry(glossary_id=glossary.id, source_term="beginning",
                              target_term="principio", notes="preferred"))
    session.add(TranslationMemory(project_id=project.id, source_locale="en", target_locale="es",
                                  source_text="In the beginning", target_text="En el principio",
                                  source_hash="other", approved=True))
    session.commit()
    client = FakeOllama()
    result = TranslationService(session, client).translate_document(
        document_id=document.id, target_locale="es", config=make_config(), model_selection=["qwen"]
    )[0]
    prompt = client.prompts[0][1]
    assert "beginning → principio (preferred)" in prompt
    assert "In the beginning → En el principio" in prompt
    version = session.get(TranslationVersion, result.version_id)
    chunk_manifest = version.run_manifest["chunks"][0]
    assert chunk_manifest["glossary_snapshot"][0]["target"] == "principio"
    assert chunk_manifest["translation_memory_snapshot"][0]["target"] == "En el principio"


def test_toml_configuration_resolves_qwen_mistral_gemma_and_direct_tag(tmp_path):
    config_path = tmp_path / "models.toml"
    config_path.write_text('''
[ollama]
base_url = "http://localhost:11434"
[translation]
target_locale = "es"
task_kind = "simplification"
target_reading_level = "4th grade"
candidate_models = ["qwen", "mistral", "gemma"]
[models.qwen]
name = "qwen3:8b"
[models.mistral]
name = "mistral:latest"
[models.gemma]
name = "gemma3:4b"
''', encoding="utf-8")
    config = TranslationConfig.from_file(config_path)
    assert [model.model_name for model in config.resolve_models()] == [
        "qwen3:8b", "mistral:latest", "gemma3:4b"
    ]
    assert config.target_reading_level == "4th grade"
    assert config.task_kind == "simplification"
    assert config.resolve_models(["qwen3:8b"])[0].model_name == "qwen3:8b"


def test_ollama_client_uses_non_streaming_generate_api(monkeypatch):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps({"model": "qwen3:8b", "response": "Hola", "done": True}).encode()

    captured = {}

    def fake_urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        captured["payload"] = json.loads(request.data)
        return FakeResponse()

    monkeypatch.setattr("ai_translator.translation.ollama.urlopen", fake_urlopen)
    result = OllamaClient(timeout_seconds=17).generate(
        model="qwen3:8b", prompt="Translate this", options={"temperature": 0.2}
    )
    assert captured["request"].full_url == "http://localhost:11434/api/generate"
    assert captured["payload"]["stream"] is False
    assert captured["payload"]["options"] == {"temperature": 0.2}
    assert captured["timeout"] == 17
    assert result.response == "Hola"


def test_unknown_model_alias_has_actionable_configuration_error():
    config = make_config()
    with pytest.raises(TranslationConfigurationError, match="Unknown model alias"):
        config.resolve_models(["unknown-model"])


def test_fourth_grade_english_simplification_is_a_distinct_prompted_task(translation_db):
    session, _project, document, _chunk = translation_db
    config = make_config(target_locale="en", task_kind="simplification",
                         target_reading_level="4th grade")
    client = FakeOllama()
    service = TranslationService(session, client)
    service.translate_document(document_id=document.id, target_locale="en", config=config,
                               model_selection=["qwen"])
    service.translate_document(document_id=document.id, target_locale="en", config=config,
                               model_selection=["qwen"], target_reading_level="8th grade")

    tasks = list(session.scalars(select(Translation).order_by(Translation.target_reading_level)))
    assert [(task.target_locale, task.task_kind, task.target_reading_level) for task in tasks] == [
        ("en", "simplification", "4th grade"),
        ("en", "simplification", "8th grade"),
    ]
    assert "4th grade reading level" in client.prompts[0][1]
    assert "reading-level simplification task" in client.prompts[0][1]
