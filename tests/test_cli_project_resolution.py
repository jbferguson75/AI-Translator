import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from typer import BadParameter
from typer.testing import CliRunner

from ai_translator.db.base import Base
from ai_translator.db.models import Project
from ai_translator.interfaces.cli import app, resolve_project


def test_project_name_is_resolved_without_uuid_primary_key_lookup():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        project = Project(name="Book of Mormon - Simplified")
        session.add(project)
        session.flush()

        result = resolve_project(session, "Book of Mormon - Simplified")

        assert result.id == project.id
    engine.dispose()


def test_duplicate_project_names_require_uuid():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.add_all([Project(name="Shared"), Project(name="Shared")])
        session.flush()

        with pytest.raises(BadParameter, match="ambiguous"):
            resolve_project(session, "Shared")
    engine.dispose()


def test_cli_can_initialize_database_and_create_project(tmp_path):
    database_url = f"sqlite:///{tmp_path / 'cli.db'}"
    runner = CliRunner()
    upgraded = runner.invoke(app, ["db", "upgrade", "--database", database_url])
    assert upgraded.exit_code == 0, upgraded.output
    created = runner.invoke(app, ["project", "create", "Book of Mormon - Simplified",
                                  "--database", database_url])
    assert created.exit_code == 0, created.output
    assert "source locale en" in created.output


def test_missing_schema_reports_upgrade_command(tmp_path):
    database_url = f"sqlite:///{tmp_path / 'not-initialized.db'}"
    runner = CliRunner()
    result = runner.invoke(app, ["project", "create", "Example", "--database", database_url])
    assert result.exit_code != 0
    assert "Database schema is not initialized" in result.output
    assert "ai-translator db" in result.output
