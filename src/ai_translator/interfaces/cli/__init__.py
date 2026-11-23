from __future__ import annotations

import typer
from pathlib import Path
from uuid import UUID
from sqlalchemy import inspect, select
from sqlalchemy.exc import IntegrityError, OperationalError
from alembic import command as alembic_command
from alembic.config import Config as AlembicConfig
from ...db.base import Base
from ...db.models import Project
from ...db.document_lifecycle import DocumentRemovalService
from ...db.repositories import DocumentRepository, EvaluationRepository
from ...db.session import make_engine, make_session_factory
from ...importing.errors import SourceImportError
from ...importing.services import SourceImportService
from ...translation.config import TranslationConfig
from ...translation.errors import TranslationError
from ...translation.ollama import OllamaClient
from ...translation.service import TranslationService
from ...evaluation.config import EvaluationConfig
from ...evaluation.errors import EvaluationError
from ...evaluation.registry import default_registry
from ...evaluation.service import EvaluationService

app = typer.Typer(no_args_is_help=True, help="Local-first translation platform")
import_app = typer.Typer(no_args_is_help=True, help="Import immutable source material")
db_app = typer.Typer(no_args_is_help=True, help="Manage the local database")
project_app = typer.Typer(no_args_is_help=True, help="Manage translation projects")
document_app = typer.Typer(no_args_is_help=True, help="List and inspect source documents")
evaluation_app = typer.Typer(no_args_is_help=True, help="Search stored evaluation results")
evaluators_app = typer.Typer(no_args_is_help=True, help="Inspect registered evaluation plugins")
app.add_typer(import_app, name="import")
app.add_typer(db_app, name="db")
app.add_typer(project_app, name="project")
app.add_typer(document_app, name="document")
app.add_typer(evaluation_app, name="evaluation")
app.add_typer(evaluators_app, name="evaluators")


def _import_using_session(project: str, database: str):
    engine = make_engine(database)
    factory = make_session_factory(engine)
    session = factory()
    try:
        project_row = resolve_project(session, project)
        return session, engine, project_row
    except OperationalError as exc:
        session.close()
        engine.dispose()
        if "no such table" in str(exc).lower():
            raise typer.BadParameter(
                f"Database schema is not initialized. Run 'ai-translator db upgrade --database {database}' first."
            ) from exc
        raise
    except Exception:
        session.close()
        engine.dispose()
        raise


def resolve_project(session, identifier: str) -> Project:
    """Resolve UUIDs as primary keys and all other values as project names."""
    try:
        project_id = UUID(identifier)
    except (ValueError, AttributeError):
        project_id = None
    if project_id is not None:
        project = session.get(Project, project_id)
        if project is not None:
            return project
    matches = list(session.scalars(select(Project).where(Project.name == identifier)))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise typer.BadParameter(f"No project found for id or name {identifier!r}")
    raise typer.BadParameter(f"Project name {identifier!r} is ambiguous; pass its UUID")


@db_app.command("upgrade")
def upgrade_database(
    database: str = typer.Option("sqlite:///ai_translator.db", "--database"),
) -> None:
    """Create or upgrade the local schema to the latest Alembic revision."""
    repository_root = Path(__file__).resolve().parents[4]
    alembic_config = AlembicConfig(str(repository_root / "alembic.ini"))
    alembic_config.set_main_option("script_location", str(repository_root / "migrations"))
    alembic_config.set_main_option("sqlalchemy.url", database.replace("%", "%%"))
    if _stamp_existing_initial_schema(database):
        typer.echo("Recognized existing initial schema and recorded its Alembic baseline")
    alembic_command.upgrade(alembic_config, "head")
    typer.echo(f"Database is up to date: {database}")


def _stamp_existing_initial_schema(database: str) -> bool:
    engine = make_engine(database)
    try:
        with engine.connect() as connection:
            inspector = inspect(connection)
            table_names = set(inspector.get_table_names())
            expected_tables = set(Base.metadata.tables)
            if not (table_names & expected_tables):
                return False
            if "alembic_version" in table_names:
                versions = connection.exec_driver_sql("SELECT version_num FROM alembic_version").scalars().all()
                if versions:
                    return False
            if not expected_tables.issubset(table_names):
                raise typer.BadParameter(
                    "Database contains an incomplete unversioned schema; refusing to run migrations automatically"
                )
            new_result_columns = {"plugin_id", "score", "confidence", "reasoning", "recommendations",
                                  "detected_issues", "searchable_text"}
            at_head = False
            for table_name in expected_tables:
                actual_columns = {column["name"] for column in inspector.get_columns(table_name)}
                expected_columns = {column.name for column in Base.metadata.tables[table_name].columns}
                if table_name == "evaluation_results":
                    baseline_columns = expected_columns - new_result_columns
                    if actual_columns == expected_columns:
                        at_head = True
                    elif actual_columns != baseline_columns:
                        raise typer.BadParameter(
                            "Unversioned evaluation_results table does not match a known schema; "
                            "refusing to stamp it"
                        )
                    continue
                if actual_columns != expected_columns:
                    raise typer.BadParameter(
                        f"Unversioned database table {table_name!r} does not match the initial schema; "
                        "refusing to run migrations automatically"
                    )
            required_triggers = {
                *(f"trg_{table}_no_{action}" for table in (
                    "books", "chapters", "verses", "chunks", "translation_versions",
                    "translation_segments", "evaluation_versions", "evaluation_results",
                    "human_reviews", "prompt_versions", "published_outputs",
                ) for action in ("update", "delete")),
                "trg_documents_source_immutable", "trg_documents_source_no_delete",
            }
            triggers = set(connection.exec_driver_sql(
                "SELECT name FROM sqlite_master WHERE type='trigger'"
            ).scalars())
            if not required_triggers.issubset(triggers):
                raise typer.BadParameter(
                    "Unversioned database is missing source immutability triggers; refusing to stamp its schema"
                )
            revision = "0002_eval_payload" if at_head else "ba0de84b1038"
            if at_head:
                indexes = {index["name"] for index in inspector.get_indexes("evaluation_results")}
                if "ix_evaluation_results_plugin_metric" not in indexes:
                    raise typer.BadParameter(
                        "Unversioned evaluation schema is missing its searchable-results index; refusing to stamp it"
                    )
            if "alembic_version" not in table_names:
                connection.exec_driver_sql(
                    "CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL PRIMARY KEY)"
                )
            connection.exec_driver_sql(
                "INSERT INTO alembic_version (version_num) VALUES (?)", (revision,)
            )
            connection.commit()
            return True
    finally:
        engine.dispose()


@project_app.command("create")
def create_project(
    name: str = typer.Argument(..., help="Project name"),
    source_locale: str = typer.Option("en", "--source-locale", "-l"),
    database: str = typer.Option("sqlite:///ai_translator.db", "--database"),
) -> None:
    """Create a project after initializing the database schema."""
    engine = make_engine(database)
    session = make_session_factory(engine)()
    try:
        existing = list(session.scalars(select(Project).where(Project.name == name)))
        if existing:
            raise typer.BadParameter(f"Project {name!r} already exists; use its UUID")
        project = Project(name=name, default_source_locale=source_locale)
        session.add(project)
        session.commit()
        typer.echo(f"Created project {project.name!r} (id {project.id}, source locale {project.default_source_locale})")
    except OperationalError as exc:
        session.rollback()
        if "no such table" in str(exc).lower():
            raise typer.BadParameter(
                f"Database schema is not initialized. Run 'ai-translator db upgrade --database {database}' first."
            ) from exc
        raise
    finally:
        session.close()
        engine.dispose()


@document_app.command("list")
def list_documents(
    project: str | None = typer.Option(None, "--project", "-p", help="Filter by project UUID or unique name"),
    database: str = typer.Option("sqlite:///ai_translator.db", "--database"),
) -> None:
    """List imported source documents and their UUIDs."""
    engine = make_engine(database)
    session = make_session_factory(engine)()
    try:
        project_row = resolve_project(session, project) if project else None
        repository = DocumentRepository(session)
        documents = (repository.for_project(project_row.id) if project_row else repository.list_all())
        if not documents:
            typer.echo("No documents found.")
            return
        for document in documents:
            typer.echo(
                f"{document.id}\t{document.title}\t{document.source_locale}\t"
                f"{document.domain}\tproject={document.project.name}"
            )
    except OperationalError as exc:
        if "no such table" in str(exc).lower():
            raise typer.BadParameter(
                f"Database schema is not initialized. Run 'ai-translator db upgrade --database {database}' first."
            ) from exc
        raise
    finally:
        session.close()
        engine.dispose()


@document_app.command("remove")
def remove_document(
    document_id: str = typer.Argument(..., help="Document UUID to remove with all dependent data"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt"),
    database: str = typer.Option("sqlite:///ai_translator.db", "--database"),
) -> None:
    """Permanently remove one document and dependent translations and reviews."""
    try:
        parsed_id = UUID(document_id)
    except ValueError as exc:
        raise typer.BadParameter("document_id must be a UUID") from exc
    if not yes:
        typer.confirm(
            f"Permanently remove document {parsed_id} and all dependent translation/evaluation data?",
            abort=True,
        )
    engine = make_engine(database)
    session = make_session_factory(engine)()
    try:
        removed = DocumentRemovalService(session).remove_document(parsed_id)
        if removed:
            typer.echo(f"Removed document {parsed_id} and its dependent data.")
        else:
            typer.echo(f"No document found with UUID {parsed_id}.", err=True)
            raise typer.Exit(code=1)
    except OperationalError as exc:
        session.rollback()
        if "no such table" in str(exc).lower():
            raise typer.BadParameter(
                f"Database schema is not initialized. Run 'ai-translator db upgrade --database {database}' first."
            ) from exc
        raise
    except IntegrityError as exc:
        session.rollback()
        raise typer.BadParameter(
            "Removal was rolled back because related records still violate a foreign-key constraint."
        ) from exc
    finally:
        session.close()
        engine.dispose()


@document_app.command("clear")
def clear_documents(
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt"),
    database: str = typer.Option("sqlite:///ai_translator.db", "--database"),
) -> None:
    """Permanently remove every document and all dependent data."""
    if not yes:
        typer.confirm(
            "Permanently remove ALL documents and their dependent translation/evaluation data?",
            abort=True,
        )
    engine = make_engine(database)
    session = make_session_factory(engine)()
    try:
        count = DocumentRemovalService(session).clear_documents()
        typer.echo(f"Removed {count} document(s) and their dependent data.")
    except OperationalError as exc:
        session.rollback()
        if "no such table" in str(exc).lower():
            raise typer.BadParameter(
                f"Database schema is not initialized. Run 'ai-translator db upgrade --database {database}' first."
            ) from exc
        raise
    except IntegrityError as exc:
        session.rollback()
        raise typer.BadParameter(
            "Removal was rolled back because related records still violate a foreign-key constraint."
        ) from exc
    finally:
        session.close()
        engine.dispose()


@import_app.command("scripture")
def import_scripture(
    path: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    project: str = typer.Option(..., "--project", "-p", help="Project UUID or unique name"),
    source_locale: str | None = typer.Option(None, "--source-locale", help="Overrides source_locale in JSON"),
    title: str | None = typer.Option(None, "--title"),
    database: str = typer.Option("sqlite:///ai_translator.db", "--database"),
) -> None:
    try:
        session, engine, project_row = _import_using_session(project, database)
        try:
            receipt = SourceImportService(session).import_scripture(
                project_id=project_row.id, path=path, source_locale=source_locale, title=title
            )
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
            engine.dispose()
        typer.echo(f"Imported scripture {receipt.title!r}: {receipt.source_units} verses (document {receipt.document_id})")
    except SourceImportError as exc:
        raise typer.BadParameter(str(exc)) from exc


@import_app.command("document")
def import_document(
    path: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    project: str = typer.Option(..., "--project", "-p", help="Project UUID or unique name"),
    source_locale: str = typer.Option(..., "--source-locale", "-l"),
    title: str | None = typer.Option(None, "--title"),
    domain: str = typer.Option("general", "--domain"),
    format_name: str | None = typer.Option(None, "--format", help="Registered parser name"),
    database: str = typer.Option("sqlite:///ai_translator.db", "--database"),
) -> None:
    try:
        session, engine, project_row = _import_using_session(project, database)
        try:
            receipt = SourceImportService(session).import_document(
                project_id=project_row.id, path=path, source_locale=source_locale,
                title=title, domain=domain, format_name=format_name,
            )
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
            engine.dispose()
        typer.echo(f"Imported document {receipt.title!r}: {receipt.source_units} chunks (document {receipt.document_id})")
    except SourceImportError as exc:
        raise typer.BadParameter(str(exc)) from exc


@app.command("translate")
def translate_document(
    document_id: str = typer.Argument(..., help="Source document UUID"),
    target_locale: str | None = typer.Option(None, "--target-locale", "-l"),
    reading_level: str | None = typer.Option(None, "--reading-level",
                                             help="Target reading level, e.g. '4th grade'"),
    task_kind: str | None = typer.Option(None, "--task-kind",
                                         help="Overrides translation.task_kind in config"),
    models: str | None = typer.Option(None, "--models", help="Comma-separated configured aliases or Ollama tags"),
    config_file: Path = typer.Option(Path("ai-translator.toml"), "--config", "-c"),
    chunk_limit: int | None = typer.Option(None, "--limit", min=1,
                                           help="Translate only the first N source chunks (stores partial candidates)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show progress while preparing and translating chunks"),
    database: str = typer.Option("sqlite:///ai_translator.db", "--database"),
) -> None:
    try:
        config = TranslationConfig.from_file(config_file)
        locale = target_locale or config.target_locale
        if not locale:
            raise TranslationError("Set translation.target_locale in config or pass --target-locale")
        try:
            parsed_document_id = UUID(document_id)
        except ValueError as exc:
            raise TranslationError("document_id must be a document UUID") from exc
        engine = make_engine(database)
        session = make_session_factory(engine)()
        try:
            client = OllamaClient(config.base_url, timeout_seconds=config.timeout_seconds)
            selection = [model.strip() for model in models.split(",") if model.strip()] if models else None
            candidates = TranslationService(session, client).translate_document(
                document_id=parsed_document_id,
                target_locale=locale,
                config=config,
                model_selection=selection,
                chunk_limit=chunk_limit,
                task_kind=task_kind,
                target_reading_level=reading_level,
                progress_callback=(lambda message: typer.echo(message, err=True)) if verbose else None,
            )
        finally:
            session.close()
            engine.dispose()
        for candidate in candidates:
            typer.echo(
                f"{candidate.model_alias} ({candidate.model_name}): {candidate.status}, "
                f"version {candidate.version_number}, {candidate.translated_chunks}/{candidate.total_chunks} chunks "
                f"[id {candidate.version_id}]"
            )
            if candidate.error:
                typer.echo(f"  error: {candidate.error}", err=True)
    except TranslationError as exc:
        raise typer.BadParameter(str(exc)) from exc


@app.command("models")
def list_ollama_models(
    config_file: Path = typer.Option(Path("ai-translator.toml"), "--config", "-c"),
) -> None:
    """Show installed Ollama models and configuration aliases."""
    try:
        config = TranslationConfig.from_file(config_file)
        client = OllamaClient(config.base_url, timeout_seconds=config.timeout_seconds)
        for item in client.list_models():
            typer.echo(f"{item.get('name', 'unknown')}  {item.get('digest', '')}")
    except TranslationError as exc:
        raise typer.BadParameter(str(exc)) from exc


@evaluators_app.command("list")
def list_evaluators() -> None:
    """List evaluator plugins available in this installation."""
    registry = default_registry()
    for plugin_id in registry.ids():
        plugin = registry.get(plugin_id)
        typer.echo(f"{plugin.plugin_id}\t{plugin.display_name}\tversion {plugin.version}")


@app.command("evaluate")
def evaluate_translation(
    translation_version_id: str = typer.Argument(..., help="Immutable translation version UUID"),
    config_file: Path = typer.Option(Path("ai-translator-evaluation.toml"), "--config", "-c"),
    plugins: str | None = typer.Option(None, "--plugins", help="Comma-separated evaluator plugin IDs"),
    models: str | None = typer.Option(None, "--models", help="Comma-separated configured evaluator aliases"),
    chunk_limit: int | None = typer.Option(None, "--limit", min=1,
                                           help="Evaluate only the first N translated chunks"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show progress while evaluating batches"),
    database: str = typer.Option("sqlite:///ai_translator.db", "--database"),
) -> None:
    try:
        config = EvaluationConfig.from_file(config_file)
        try:
            version_id = UUID(translation_version_id)
        except ValueError as exc:
            raise EvaluationError("translation_version_id must be a UUID") from exc
        engine = make_engine(database)
        session = make_session_factory(engine)()
        try:
            service = EvaluationService(session, OllamaClient(
                config.base_url, timeout_seconds=config.timeout_seconds
            ))
            candidates = service.evaluate_translation_version(
                translation_version_id=version_id,
                config=config,
                plugin_selection=[value.strip() for value in plugins.split(",") if value.strip()] if plugins else None,
                model_selection=[value.strip() for value in models.split(",") if value.strip()] if models else None,
                chunk_limit=chunk_limit,
                progress_callback=(lambda message: typer.echo(message, err=True)) if verbose else None,
            )
        finally:
            session.close()
            engine.dispose()
        for candidate in candidates:
            score = "n/a" if candidate.score is None else f"{candidate.score:.2f}"
            confidence = "n/a" if candidate.confidence is None else f"{candidate.confidence:.3f}"
            typer.echo(
                f"{candidate.plugin_id} / {candidate.model_alias}: {candidate.status}, "
                f"score {score}, confidence {confidence}, {candidate.issue_count} issues, "
                f"version {candidate.version_number} [{candidate.evaluation_version_id}]"
            )
            if candidate.error:
                typer.echo(f"  error: {candidate.error}", err=True)
    except EvaluationError as exc:
        raise typer.BadParameter(str(exc)) from exc


@evaluation_app.command("search")
def search_evaluations(
    query: str = typer.Argument(..., help="Substring to find in reasoning, issues, or recommendations"),
    plugin: str | None = typer.Option(None, "--plugin"),
    limit: int = typer.Option(50, "--limit", min=1, max=1000),
    database: str = typer.Option("sqlite:///ai_translator.db", "--database"),
) -> None:
    engine = make_engine(database)
    session = make_session_factory(engine)()
    try:
        results = EvaluationRepository(session).search_results(query, plugin_id=plugin, limit=limit)
        for result in results:
            issues = "; ".join(result.detected_issues)
            typer.echo(
                f"{result.plugin_id} score={result.score} confidence={result.confidence}\n"
                f"Reasoning: {result.reasoning or ''}\nIssues: {issues}\n"
                f"Recommendations: {'; '.join(result.recommendations)}\n"
            )
        typer.echo(f"{len(results)} result(s)")
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    except OperationalError as exc:
        if "no such table" in str(exc).lower():
            raise typer.BadParameter("Database schema is not initialized; run 'ai-translator db upgrade' first") from exc
        raise
    finally:
        session.close()
        engine.dispose()


def run() -> None:
    app()


if __name__ == "__main__":
    run()

