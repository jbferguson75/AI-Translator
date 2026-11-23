from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text


def test_alembic_upgrade_creates_schema_and_immutability_triggers(tmp_path: Path):
    database = tmp_path / "migration.db"
    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", f"sqlite:///{database}")
    command.upgrade(config, "head")

    engine = create_engine(f"sqlite:///{database}")
    with engine.connect() as connection:
        tables = set(connection.execute(text(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )).scalars())
        assert {"projects", "documents", "translation_versions", "evaluation_versions"} <= tables
        triggers = set(connection.execute(text(
            "SELECT name FROM sqlite_master WHERE type='trigger'"
        )).scalars())
        assert "trg_documents_source_immutable" in triggers
        assert "trg_translation_versions_no_update" in triggers

