from __future__ import annotations

import typer
import json
from pathlib import Path
from datetime import datetime
from uuid import UUID
from sqlalchemy import inspect, select
from sqlalchemy.exc import IntegrityError, OperationalError
from alembic import command as alembic_command
from alembic.config import Config as AlembicConfig
from ...db.base import Base
from ...db.models import (
    Experiment, Glossary, GlossaryEntry, GlossarySuggestionRun, Project, TranslationMemory,
    TranslationVersion,
)
from ...db.document_lifecycle import DocumentRemovalService
from ...db.repositories import DocumentRepository, EvaluationRepository
from ...db.session import make_engine, make_session_factory
from ...importing.errors import SourceImportError
from ...importing.services import SourceImportService
from ...translation.config import TranslationConfig
from ...translation.errors import TranslationError
from ...translation.ollama import OllamaClient
from ...translation.service import TranslationService
from ...translation.refinement import TranslationRefinementService
from ...translation.iteration import TranslationIterationService
from ...translation.reporting import (comparison_report, convergence_report, improvement_report,
                                      lineage_report, side_by_side_report)
from ...evaluation.config import EvaluationConfig
from ...evaluation.errors import EvaluationError
from ...evaluation.registry import default_registry
from ...evaluation.service import EvaluationService
from ...glossary.service import GlossaryService
from ...glossary.suggestions import GlossarySuggestionService
from ...translation_memory.service import TranslationMemoryService
from ...review.service import HumanReviewService

app = typer.Typer(no_args_is_help=True, help="Local-first translation platform")
import_app = typer.Typer(no_args_is_help=True, help="Import immutable source material")
db_app = typer.Typer(no_args_is_help=True, help="Manage the local database")
project_app = typer.Typer(no_args_is_help=True, help="Manage translation projects")
document_app = typer.Typer(no_args_is_help=True, help="List and inspect source documents")
evaluation_app = typer.Typer(no_args_is_help=True, help="Search stored evaluation results")
evaluators_app = typer.Typer(no_args_is_help=True, help="Inspect registered evaluation plugins")
glossary_app = typer.Typer(no_args_is_help=True, help="Manage project translation glossaries")
memory_app = typer.Typer(no_args_is_help=True, help="Manage and search translation memories")
human_review_app = typer.Typer(no_args_is_help=True, help="Review, approve, and audit translation versions")
review_app = typer.Typer(no_args_is_help=True, help="Compare reviewed translation candidates")
report_app = typer.Typer(no_args_is_help=True, help="Translation lineage and quality reports")
app.add_typer(import_app, name="import")
app.add_typer(db_app, name="db")
app.add_typer(project_app, name="project")
app.add_typer(document_app, name="document")
app.add_typer(evaluation_app, name="evaluation")
app.add_typer(evaluators_app, name="evaluators")
app.add_typer(glossary_app, name="glossary")
app.add_typer(memory_app, name="memory")
app.add_typer(human_review_app, name="human-review")
app.add_typer(review_app, name="review")
app.add_typer(report_app, name="report")


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


def _run_db_action(database: str, action):
    engine = make_engine(database)
    session = make_session_factory(engine)()
    try:
        result = action(session)
        session.commit()
        return result
    except OperationalError as exc:
        session.rollback()
        if "no such table" in str(exc).lower() or "no such column" in str(exc).lower():
            raise typer.BadParameter(
                f"Database schema is not initialized or is outdated. Run 'ai-translator db upgrade --database {database}' first."
            ) from exc
        raise
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
        engine.dispose()


def _parse_uuid(value: str, label: str) -> UUID:
    try:
        return UUID(value)
    except ValueError as exc:
        raise typer.BadParameter(f"{label} must be a UUID") from exc


@human_review_app.command("create")
def create_human_review_pass(
    translation_version_id: str = typer.Argument(..., help="Immutable AI or human translation version UUID"),
    name: str = typer.Option("Human review", "--name"),
    assigned_to: str | None = typer.Option(None, "--assigned-to"),
    created_by: str = typer.Option("local-user", "--created-by"),
    database: str = typer.Option("sqlite:///ai_translator.db", "--database"),
) -> None:
    """Create a review pass with one pending item for every translated chunk."""
    version_id = _parse_uuid(translation_version_id, "translation_version_id")
    try:
        review_pass, item_count = _run_db_action(database, lambda session: HumanReviewService(session).create_pass(
            translation_version_id=version_id, name=name, assigned_to=assigned_to, created_by=created_by,
        ))
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"Created review pass {review_pass.id}: {item_count} pending item(s) for version {version_id}")


@human_review_app.command("passes")
def list_human_review_passes(
    translation_version_id: str | None = typer.Option(None, "--translation-version"),
    status: str | None = typer.Option(None, "--status", help="Filter by pass status, e.g. open or completed"),
    database: str = typer.Option("sqlite:///ai_translator.db", "--database"),
) -> None:
    """List review passes and their source translation versions."""
    version_id = _parse_uuid(translation_version_id, "translation_version_id") if translation_version_id else None
    rows = _run_db_action(database, lambda session: [
        f"{item.id}\t{item.status}\t{item.name}\tassigned_to={item.assigned_to or '-'}\t"
        f"translation_version={item.translation_version_id}\tcreated={item.created_at.isoformat()}"
        for item in HumanReviewService(session).list_passes(translation_version_id=version_id, status=status)
    ])
    if not rows:
        typer.echo("No human review passes found.")
    for row in rows:
        typer.echo(row)


@human_review_app.command("items")
def list_human_review_items(
    review_pass_id: str = typer.Argument(...),
    status: str | None = typer.Option(None, "--status", help="pending, approved, rejected, or needs_revision"),
    database: str = typer.Option("sqlite:///ai_translator.db", "--database"),
) -> None:
    """List current review items, source text, candidate text, and decision state."""
    pass_id = _parse_uuid(review_pass_id, "review_pass_id")
    if status is not None and status not in {"pending", "approved", "rejected", "needs_revision"}:
        raise typer.BadParameter("status must be pending, approved, rejected, or needs_revision")
    try:
        rows = _run_db_action(database, lambda session: [
            f"{item.review_event_id}\t{item.status}\t{item.stable_key}\t"
            f"chunk={item.chunk_id}\n  SOURCE: {item.source_text}\n  TARGET: {item.candidate_text}"
            for item in HumanReviewService(session).items(pass_id, status=status)
        ])
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if not rows:
        typer.echo("No review items match that filter.")
    for row in rows:
        typer.echo(row)


@human_review_app.command("decide")
def decide_human_review(
    review_event_id: str = typer.Argument(..., help="Current pending/current event UUID from human-review items"),
    status: str = typer.Option(..., "--status", help="pending, approved, rejected, or needs_revision"),
    reviewer: str = typer.Option(..., "--reviewer"),
    review_notes: str | None = typer.Option(None, "--notes"),
    correction: str | None = typer.Option(None, "--correction", help="Corrected target text for this chunk"),
    override_reason: str | None = typer.Option(None, "--override-reason"),
    approval_date: str | None = typer.Option(None, "--approval-date", help="ISO-8601 timestamp; defaults to now on approval"),
    database: str = typer.Option("sqlite:///ai_translator.db", "--database"),
) -> None:
    """Append a review decision; approved results create a new human-authored version."""
    event_id = _parse_uuid(review_event_id, "review_event_id")
    parsed_approval_date = None
    if approval_date:
        try:
            parsed_approval_date = datetime.fromisoformat(approval_date.replace("Z", "+00:00"))
        except ValueError as exc:
            raise typer.BadParameter("approval_date must be an ISO-8601 timestamp") from exc
    try:
        event = _run_db_action(database, lambda session: HumanReviewService(session).decide(
            review_event_id=event_id, status=status, reviewer=reviewer,
            review_notes=review_notes, correction=correction,
            override_reason=override_reason, approval_date=parsed_approval_date,
        ))
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(
        f"Recorded {event.status} by {event.reviewer}; review event {event.id}; "
        f"approved_at={event.approval_date.isoformat() if event.approval_date else '-'}; "
        f"human_translation_version={event.resulting_translation_version_id or '-'}; "
        f"audit_hash={event.event_hash}"
    )


@human_review_app.command("history")
def human_review_history(
    review_event_id: str = typer.Argument(...),
    database: str = typer.Option("sqlite:///ai_translator.db", "--database"),
) -> None:
    """Show append-only event history for one review item."""
    event_id = _parse_uuid(review_event_id, "review_event_id")
    try:
        rows = _run_db_action(database, lambda session: [
            f"{item.id}\t{item.status}\treviewer={item.reviewer}\tcreated={item.created_at.isoformat()}\t"
            f"approval_date={item.approval_date.isoformat() if item.approval_date else '-'}\t"
            f"supersedes={item.supersedes_review_id or '-'}\tparent_hash={item.previous_event_hash or '-'}\t"
            f"event_hash={item.event_hash}\n  notes={item.review_notes or '-'}\n  correction={item.correction or '-'}\n"
            f"  override_reason={item.override_reason or '-'}\n"
            f"  human_translation_version={item.resulting_translation_version_id or '-'}"
            for item in HumanReviewService(session).history(event_id)
        ])
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    for row in rows:
        typer.echo(row)


@human_review_app.command("audit")
def verify_human_review_audit(
    review_pass_id: str = typer.Argument(...),
    database: str = typer.Option("sqlite:///ai_translator.db", "--database"),
) -> None:
    """Verify event ordering, supersedes links, and the per-item SHA-256 chain."""
    pass_id = _parse_uuid(review_pass_id, "review_pass_id")
    try:
        result = _run_db_action(database, lambda session: HumanReviewService(session).verify_audit(pass_id))
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"Review pass {pass_id}: {result.event_count} event(s), audit {'valid' if result.valid else 'INVALID'}")
    for error in result.errors:
        typer.echo(f"  {error}", err=True)
    if not result.valid:
        raise typer.Exit(code=1)


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
            suggestion_tables = {"glossary_suggestion_runs", "glossary_suggestions"}
            lineage_tables = {"translation_version_parents"}
            present_suggestion_tables = suggestion_tables & table_names
            if not (table_names & expected_tables):
                return False
            if "alembic_version" in table_names:
                versions = connection.exec_driver_sql("SELECT version_num FROM alembic_version").scalars().all()
                if versions:
                    return False
            if len(present_suggestion_tables) not in {0, len(suggestion_tables)}:
                raise typer.BadParameter(
                    "Unversioned database contains only part of the glossary-suggestion schema; refusing to stamp it"
                )
            required_existing_tables = expected_tables - (suggestion_tables - present_suggestion_tables) \
                - (lineage_tables - (lineage_tables & table_names))
            if not required_existing_tables.issubset(table_names):
                raise typer.BadParameter(
                    "Database contains an incomplete unversioned schema; refusing to run migrations automatically"
                )
            new_result_columns = {"plugin_id", "score", "confidence", "reasoning", "recommendations",
                                  "detected_issues", "searchable_text", "strengths", "issue_details"}
            evaluation_payload_present = False
            structured_feedback_present = False
            lineage_present = False
            knowledge_usage_present: dict[str, bool] = {"glossary_entries": False,
                                                        "translation_memories": False}
            knowledge_usage_columns = {
                "glossary_entries": {"priority", "occurrence_count", "source_references"},
                "translation_memories": {"occurrence_count", "source_references"},
            }
            human_review_columns = {
                "status", "approval_date", "review_notes", "correction", "override_reason",
                "supersedes_review_id", "previous_event_hash", "event_hash",
            }
            human_review_workflow_present = False
            for table_name in expected_tables:
                if table_name not in table_names:
                    continue
                actual_columns = {column["name"] for column in inspector.get_columns(table_name)}
                expected_columns = {column.name for column in Base.metadata.tables[table_name].columns}
                if table_name == "translation_version_parents":
                    if actual_columns != expected_columns:
                        raise typer.BadParameter("Unversioned lineage table does not match the current schema")
                    lineage_present = True
                    continue
                if table_name == "translation_versions":
                    lineage_columns = {"artifact_kind", "prompt_version_id"}
                    if actual_columns == expected_columns:
                        pass
                    elif actual_columns == expected_columns - lineage_columns:
                        pass  # recognized pre-lineage schema; Alembic revision 0007 adds these columns
                    else:
                        raise typer.BadParameter("Unversioned translation_versions table does not match a known schema")
                    continue
                if table_name == "evaluation_results":
                    baseline_columns = expected_columns - new_result_columns
                    pre_structured_columns = expected_columns - {"strengths", "issue_details"}
                    if actual_columns == expected_columns:
                        evaluation_payload_present = True
                        structured_feedback_present = True
                    elif actual_columns == pre_structured_columns:
                        evaluation_payload_present = True
                    elif actual_columns != baseline_columns:
                        raise typer.BadParameter(
                            "Unversioned evaluation_results table does not match a known schema; "
                            "refusing to stamp it"
                        )
                    continue
                if table_name in knowledge_usage_columns:
                    new_columns = knowledge_usage_columns[table_name]
                    baseline_columns = expected_columns - new_columns
                    if actual_columns == expected_columns:
                        knowledge_usage_present[table_name] = True
                    elif actual_columns != baseline_columns:
                        raise typer.BadParameter(
                            f"Unversioned table {table_name!r} does not match a known schema; refusing to stamp it"
                        )
                    continue
                if table_name == "human_reviews":
                    baseline_columns = expected_columns - human_review_columns
                    if actual_columns == expected_columns:
                        human_review_workflow_present = True
                    elif actual_columns != baseline_columns:
                        raise typer.BadParameter(
                            "Unversioned human_reviews table does not match a known schema; refusing to stamp it"
                        )
                    continue
                if actual_columns != expected_columns:
                    raise typer.BadParameter(
                        f"Unversioned database table {table_name!r} does not match the initial schema; "
                        "refusing to run migrations automatically"
                    )
            if len(set(knowledge_usage_present.values())) != 1 or (
                any(knowledge_usage_present.values()) and not evaluation_payload_present
            ):
                raise typer.BadParameter(
                    "Unversioned database contains a partial glossary/memory migration; refusing to stamp it"
                )
            required_triggers = {
                *(f"trg_{table}_no_{action}" for table in (
                    "books", "chapters", "verses", "chunks", "translation_versions",
                    "translation_segments", "evaluation_versions", "evaluation_results",
                    "human_reviews", "prompt_versions", "published_outputs",
                ) for action in ("update", "delete")),
                "trg_documents_source_immutable", "trg_documents_source_no_delete",
            }
            if human_review_workflow_present:
                required_triggers.update({
                    "trg_human_reviews_validate_insert", "trg_human_reviews_supersedes_scope",
                })
            if lineage_present:
                required_triggers.update({"trg_translation_version_parents_no_update",
                                          "trg_translation_version_parents_no_delete"})
            triggers = set(connection.exec_driver_sql(
                "SELECT name FROM sqlite_master WHERE type='trigger'"
            ).scalars())
            if not required_triggers.issubset(triggers):
                raise typer.BadParameter(
                    "Unversioned database is missing source immutability triggers; refusing to stamp its schema"
                )
            if evaluation_payload_present:
                indexes = {index["name"] for index in inspector.get_indexes("evaluation_results")}
                if "ix_evaluation_results_plugin_metric" not in indexes:
                    raise typer.BadParameter(
                        "Unversioned evaluation schema is missing its searchable-results index; refusing to stamp it"
                    )
            usage_is_current = all(knowledge_usage_present.values())
            if usage_is_current:
                indexes = {index["name"] for index in inspector.get_indexes("glossary_entries")}
                if "ix_glossary_entries_term_priority" not in indexes:
                    raise typer.BadParameter(
                        "Unversioned glossary schema is missing its term-priority index; refusing to stamp it"
                    )
            if evaluation_payload_present:
                revision = "0004_glossary_memory_usage" if usage_is_current else "0003_restore_delete_guards"
            else:
                revision = "ba0de84b1038"
            if present_suggestion_tables:
                if not evaluation_payload_present or not usage_is_current:
                    raise typer.BadParameter(
                        "Glossary-suggestion tables exist without the preceding schema revisions; refusing to stamp"
                    )
                for table_name in suggestion_tables:
                    actual_columns = {column["name"] for column in inspector.get_columns(table_name)}
                    expected_columns = {column.name for column in Base.metadata.tables[table_name].columns}
                    if actual_columns != expected_columns:
                        raise typer.BadParameter(
                            f"Unversioned {table_name!r} does not match the glossary-suggestion schema"
                        )
                run_indexes = {index["name"] for index in inspector.get_indexes("glossary_suggestion_runs")}
                suggestion_indexes = {index["name"] for index in inspector.get_indexes("glossary_suggestions")}
                if ("ix_glossary_suggestion_runs_version_status" not in run_indexes
                        or not {"ix_glossary_suggestions_run_status", "ix_glossary_suggestions_project_term"}
                        .issubset(suggestion_indexes)):
                    raise typer.BadParameter("Glossary-suggestion schema is missing required indexes")
                revision = "0005_glossary_suggestion_workflow"
            if human_review_workflow_present:
                if not present_suggestion_tables:
                    raise typer.BadParameter(
                        "Human-review workflow columns exist without preceding schema revisions; refusing to stamp"
                    )
                review_indexes = {index["name"] for index in inspector.get_indexes("human_reviews")}
                if not {"ix_human_reviews_pass_chunk_created", "ix_human_reviews_status_created"}.issubset(review_indexes):
                    raise typer.BadParameter("Human-review workflow is missing required indexes")
                revision = "0006_human_review_workflow"
            if lineage_present and structured_feedback_present and revision == "0006_human_review_workflow":
                revision = "0007_translation_lineage_feedback"
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


@glossary_app.command("create")
def create_glossary(
    name: str = typer.Argument(...),
    project: str = typer.Option(..., "--project", "-p", help="Project UUID or unique name"),
    source_locale: str = typer.Option("en", "--source-locale"),
    target_locale: str = typer.Option(..., "--target-locale", "-l"),
    database: str = typer.Option("sqlite:///ai_translator.db", "--database"),
) -> None:
    """Create another terminology glossary within a project."""
    def action(session):
        project_row = resolve_project(session, project)
        glossary = GlossaryService(session).create(
            project_id=project_row.id, name=name,
            source_locale=source_locale, target_locale=target_locale,
        )
        return f"Created glossary {glossary.name!r} ({glossary.id})"
    try:
        typer.echo(_run_db_action(database, action))
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


@glossary_app.command("list")
def list_glossaries(
    project: str = typer.Option(..., "--project", "-p", help="Project UUID or unique name"),
    database: str = typer.Option("sqlite:///ai_translator.db", "--database"),
) -> None:
    """List all glossaries in a project."""
    def action(session):
        project_row = resolve_project(session, project)
        rows = GlossaryService(session).list_for_project(project_row.id)
        return [f"{row.id}\t{row.name}\t{row.source_locale} → {row.target_locale}" for row in rows]
    try:
        rows = _run_db_action(database, action)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if not rows:
        typer.echo("No glossaries found.")
    for row in rows:
        typer.echo(row)


@glossary_app.command("add-entry")
def add_glossary_entry(
    glossary_id: str = typer.Argument(...),
    source_term: str = typer.Argument(...),
    preferred_translation: str = typer.Argument(...),
    priority: int = typer.Option(0, "--priority", min=0,
                                 help="Higher priority wins when project glossaries disagree"),
    notes: str | None = typer.Option(None, "--notes"),
    database: str = typer.Option("sqlite:///ai_translator.db", "--database"),
) -> None:
    """Add a term and its preferred translation."""
    glossary_uuid = _parse_uuid(glossary_id, "glossary_id")
    def action(session):
        entry = GlossaryService(session).add_entry(
            glossary_id=glossary_uuid, source_term=source_term,
            preferred_translation=preferred_translation, priority=priority, notes=notes,
        )
        return f"Added glossary entry {entry.id}: {entry.source_term} → {entry.target_term}"
    try:
        typer.echo(_run_db_action(database, action))
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


@glossary_app.command("entries")
def list_glossary_entries(
    glossary_id: str = typer.Argument(...),
    database: str = typer.Option("sqlite:///ai_translator.db", "--database"),
) -> None:
    """List terms, priority, and observed usage for a glossary."""
    glossary_uuid = _parse_uuid(glossary_id, "glossary_id")
    rows = _run_db_action(database, lambda session: [
        f"{entry.id}\t{entry.source_term} → {entry.target_term}\tpriority={entry.priority}\t"
        f"uses={entry.occurrence_count}\treferences={', '.join(entry.source_references or [])}\t"
        f"notes={entry.notes or ''}"
        for entry in GlossaryService(session).entries(glossary_uuid)
    ])
    for row in rows:
        typer.echo(row)
    if not rows:
        typer.echo("No entries found.")


@glossary_app.command("update-entry")
def update_glossary_entry(
    entry_id: str = typer.Argument(...),
    preferred_translation: str | None = typer.Option(None, "--preferred-translation"),
    priority: int | None = typer.Option(None, "--priority", min=0),
    notes: str | None = typer.Option(None, "--notes"),
    database: str = typer.Option("sqlite:///ai_translator.db", "--database"),
) -> None:
    """Change a preferred term, priority, or notes to resolve conflicts."""
    entry_uuid = _parse_uuid(entry_id, "entry_id")
    def action(session):
        entry = GlossaryService(session).update_entry(
            entry_uuid, preferred_translation=preferred_translation, priority=priority, notes=notes,
        )
        return f"Updated {entry.source_term}: {entry.target_term} (priority {entry.priority})"
    try:
        typer.echo(_run_db_action(database, action))
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


@glossary_app.command("remove-entry")
def remove_glossary_entry(
    entry_id: str = typer.Argument(...),
    database: str = typer.Option("sqlite:///ai_translator.db", "--database"),
) -> None:
    """Remove a glossary entry."""
    entry_uuid = _parse_uuid(entry_id, "entry_id")
    removed = _run_db_action(database, lambda session: GlossaryService(session).remove_entry(entry_uuid))
    typer.echo("Glossary entry removed." if removed else "No glossary entry found.")


@glossary_app.command("remove")
def remove_glossary(
    glossary_id: str = typer.Argument(...),
    database: str = typer.Option("sqlite:///ai_translator.db", "--database"),
) -> None:
    """Remove a glossary and all of its entries."""
    glossary_uuid = _parse_uuid(glossary_id, "glossary_id")
    removed = _run_db_action(database, lambda session: GlossaryService(session).remove(glossary_uuid))
    typer.echo("Glossary removed." if removed else "No glossary found.")


@glossary_app.command("conflicts")
def glossary_conflicts(
    project: str = typer.Option(..., "--project", "-p"),
    source_locale: str = typer.Option("en", "--source-locale"),
    target_locale: str = typer.Option(..., "--target-locale", "-l"),
    database: str = typer.Option("sqlite:///ai_translator.db", "--database"),
) -> None:
    """Show equal-priority glossary entries with different translations."""
    def action(session):
        project_row = resolve_project(session, project)
        return [
            f"{conflict.source_term!r} (priority {conflict.priority}): " + "; ".join(
                f"{entry.target_term!r} in {entry.glossary.name} [{entry.id}]"
                for entry in conflict.entries
            )
            for conflict in GlossaryService(session).conflicts(
                project_id=project_row.id, source_locale=source_locale, target_locale=target_locale,
            )
        ]
    try:
        rows = _run_db_action(database, action)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if not rows:
        typer.echo("No unresolved top-priority glossary conflicts.")
    for row in rows:
        typer.echo(row)


@glossary_app.command("suggest")
def suggest_glossary_entries(
    translation_version_id: str = typer.Argument(..., help="Draft translation version UUID"),
    config_file: Path = typer.Option(Path("ai-translator.toml"), "--config", "-c"),
    model: str | None = typer.Option(None, "--model", help="Configured translator model alias or Ollama model tag"),
    chunk_limit: int | None = typer.Option(None, "--limit", min=1,
                                           help="Analyze only the first N translated chunks"),
    database: str = typer.Option("sqlite:///ai_translator.db", "--database"),
) -> None:
    """Analyze a draft version and persist repeat-term suggestions for human review."""
    try:
        version_uuid = _parse_uuid(translation_version_id, "translation_version_id")
        config = TranslationConfig.from_file(config_file)
        selected_models = config.resolve_models([model] if model else None)
        selected_model = selected_models[0]
        engine = make_engine(database)
        session = make_session_factory(engine)()
        try:
            result = GlossarySuggestionService(
                session, OllamaClient(config.base_url, timeout_seconds=config.timeout_seconds)
            ).suggest(
                translation_version_id=version_uuid,
                model=selected_model,
                temperature=config.temperature,
                num_ctx=config.num_ctx,
                chunk_limit=chunk_limit,
            )
        finally:
            session.close()
            engine.dispose()
        typer.echo(
            f"Glossary suggestion run {result.version_number}: {result.status}, "
            f"{result.suggestion_count} suggestion(s), model {result.model_name} [id {result.run_id}]"
        )
        if result.error:
            typer.echo(f"  error: {result.error}", err=True)
    except TranslationError as exc:
        raise typer.BadParameter(str(exc)) from exc
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


@glossary_app.command("suggestions")
def list_glossary_suggestions(
    translation_version: str | None = typer.Option(None, "--translation-version"),
    project: str | None = typer.Option(None, "--project", "-p"),
    status: str = typer.Option("proposed", "--status",
                               help="Filter: proposed, approved, or rejected"),
    database: str = typer.Option("sqlite:///ai_translator.db", "--database"),
) -> None:
    """List glossary suggestions awaiting review or previously decided."""
    if status not in {"proposed", "approved", "rejected"}:
        raise typer.BadParameter("status must be proposed, approved, or rejected")
    version_uuid = _parse_uuid(translation_version, "translation_version") if translation_version else None
    def action(session):
        project_id = resolve_project(session, project).id if project else None
        rows = GlossarySuggestionService(session, OllamaClient()).list_suggestions(
            translation_version_id=version_uuid, project_id=project_id, status=status,
        )
        return [
            f"{row.id}\t{row.status}\t{row.source_term} → {row.preferred_translation}\t"
            f"confidence={row.confidence:.2f}\toccurrences={row.occurrence_count}\t"
            f"references={', '.join(row.source_references)}\treason={row.reasoning}"
            for row in rows
        ]
    try:
        rows = _run_db_action(database, action)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if not rows:
        typer.echo(f"No {status} glossary suggestions found.")
    for row in rows:
        typer.echo(row)


@glossary_app.command("suggestion-runs")
def list_glossary_suggestion_runs(
    translation_version: str = typer.Argument(...),
    database: str = typer.Option("sqlite:///ai_translator.db", "--database"),
) -> None:
    """List repeatable analysis runs and their status for a translation version."""
    version_uuid = _parse_uuid(translation_version, "translation_version")
    rows = _run_db_action(database, lambda session: [
        f"{run.id}\tversion={run.version_number}\t{run.status}\tmodel={run.model_name}\t"
        f"suggestions={len(run.suggestions)}\terror={run.run_manifest.get('error', '')}"
        for run in session.scalars(select(GlossarySuggestionRun).where(
            GlossarySuggestionRun.translation_version_id == version_uuid
        ).order_by(GlossarySuggestionRun.version_number))
    ])
    if not rows:
        typer.echo("No glossary suggestion runs found.")
    for row in rows:
        typer.echo(row)


@glossary_app.command("approve-suggestion")
def approve_glossary_suggestion(
    suggestion_id: str = typer.Argument(...),
    glossary_id: str = typer.Option(..., "--glossary", help="Destination glossary UUID"),
    priority: int = typer.Option(10, "--priority", min=0),
    preferred_translation: str | None = typer.Option(None, "--preferred-translation",
                                                       help="Override the model suggestion"),
    reviewer: str = typer.Option("local-user", "--reviewer"),
    note: str | None = typer.Option(None, "--note"),
    database: str = typer.Option("sqlite:///ai_translator.db", "--database"),
) -> None:
    """Approve a suggestion and add it to a matching glossary."""
    suggestion_uuid = _parse_uuid(suggestion_id, "suggestion_id")
    glossary_uuid = _parse_uuid(glossary_id, "glossary_id")
    def action(session):
        suggestion = GlossarySuggestionService(session, OllamaClient()).approve(
            suggestion_id=suggestion_uuid, glossary_id=glossary_uuid, priority=priority,
            reviewer=reviewer, preferred_translation=preferred_translation, note=note,
        )
        return f"Approved {suggestion.source_term} → {suggestion.preferred_translation}; glossary entry {suggestion.glossary_entry_id}"
    try:
        typer.echo(_run_db_action(database, action))
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


@glossary_app.command("reject-suggestion")
def reject_glossary_suggestion(
    suggestion_id: str = typer.Argument(...),
    reviewer: str = typer.Option("local-user", "--reviewer"),
    note: str | None = typer.Option(None, "--note"),
    database: str = typer.Option("sqlite:///ai_translator.db", "--database"),
) -> None:
    """Reject a proposed glossary term and record who reviewed it."""
    suggestion_uuid = _parse_uuid(suggestion_id, "suggestion_id")
    def action(session):
        suggestion = GlossarySuggestionService(session, OllamaClient()).reject(
            suggestion_id=suggestion_uuid, reviewer=reviewer, note=note,
        )
        return f"Rejected {suggestion.source_term} → {suggestion.preferred_translation}"
    try:
        typer.echo(_run_db_action(database, action))
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


@glossary_app.command("revise")
def revise_translation_with_glossary(
    translation_version_id: str = typer.Argument(..., help="Draft translation version UUID"),
    config_file: Path = typer.Option(Path("ai-translator.toml"), "--config", "-c"),
    models: str | None = typer.Option(None, "--models", help="Comma-separated configured aliases or Ollama tags"),
    chunk_limit: int | None = typer.Option(None, "--limit", min=1),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    database: str = typer.Option("sqlite:///ai_translator.db", "--database"),
) -> None:
    """Create new translation version(s) using currently approved glossary entries."""
    try:
        draft_id = _parse_uuid(translation_version_id, "translation_version_id")
        config = TranslationConfig.from_file(config_file)
        selection = [value.strip() for value in models.split(",") if value.strip()] if models else None
        engine = make_engine(database)
        session = make_session_factory(engine)()
        try:
            draft = session.get(TranslationVersion, draft_id)
            if draft is None:
                raise TranslationError(f"Translation version {draft_id} does not exist")
            translation = draft.translation
            scope = (draft.run_manifest or {}).get("scope", {})
            candidates = TranslationService(
                session, OllamaClient(config.base_url, timeout_seconds=config.timeout_seconds)
            ).translate_document(
                document_id=translation.document_id,
                target_locale=translation.target_locale,
                config=config,
                model_selection=selection,
                chunk_limit=chunk_limit,
                task_kind=translation.task_kind,
                target_reading_level=translation.target_reading_level,
                book=scope.get("book"),
                chapter=scope.get("chapter"),
                based_on_translation_version_id=draft.id,
                progress_callback=(lambda message: typer.echo(message, err=True)) if verbose else None,
            )
        finally:
            session.close()
            engine.dispose()
        for candidate in candidates:
            typer.echo(
                f"{candidate.model_alias} ({candidate.model_name}): {candidate.status}, "
                f"version {candidate.version_number}, {candidate.translated_chunks}/{candidate.total_chunks} "
                f"{candidate.scope_label} chunks [id {candidate.version_id}] (based on {draft_id})"
            )
            if candidate.error:
                typer.echo(f"  error: {candidate.error}", err=True)
    except TranslationError as exc:
        raise typer.BadParameter(str(exc)) from exc
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


@memory_app.command("add")
def add_memory(
    source_phrase: str = typer.Argument(...),
    preferred_translation: str = typer.Argument(...),
    project: str = typer.Option(..., "--project", "-p", help="Project UUID or unique name"),
    source_locale: str = typer.Option("en", "--source-locale"),
    target_locale: str = typer.Option(..., "--target-locale", "-l"),
    source_reference: str | None = typer.Option(None, "--source-reference"),
    approved: bool = typer.Option(False, "--approved", help="Allow automatic translation prompt injection"),
    database: str = typer.Option("sqlite:///ai_translator.db", "--database"),
) -> None:
    """Add a phrase pair to the project's translation memory."""
    def action(session):
        project_row = resolve_project(session, project)
        memory = TranslationMemoryService(session).add(
            project_id=project_row.id, source_locale=source_locale, target_locale=target_locale,
            source_phrase=source_phrase, preferred_translation=preferred_translation,
            approved=approved, source_reference=source_reference,
        )
        return f"Added memory {memory.id}: approved={memory.approved}"
    try:
        typer.echo(_run_db_action(database, action))
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


@memory_app.command("list")
def list_memories(
    project: str = typer.Option(..., "--project", "-p"),
    source_locale: str | None = typer.Option(None, "--source-locale"),
    target_locale: str | None = typer.Option(None, "--target-locale", "-l"),
    database: str = typer.Option("sqlite:///ai_translator.db", "--database"),
) -> None:
    """List phrase pairs and their approval and usage state."""
    def action(session):
        project_row = resolve_project(session, project)
        return [
            f"{entry.id}\t{entry.source_locale} → {entry.target_locale}\t"
            f"{entry.source_text} → {entry.target_text}\tapproved={entry.approved}\t"
            f"uses={entry.occurrence_count}\treferences={', '.join(entry.source_references or [])}"
            for entry in TranslationMemoryService(session).list_for_project(
                project_row.id, source_locale=source_locale, target_locale=target_locale,
            )
        ]
    try:
        rows = _run_db_action(database, action)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if not rows:
        typer.echo("No translation memories found.")
    for row in rows:
        typer.echo(row)


@memory_app.command("lookup")
def lookup_memories(
    source_phrase: str = typer.Argument(...),
    project: str = typer.Option(..., "--project", "-p"),
    source_locale: str = typer.Option("en", "--source-locale"),
    target_locale: str = typer.Option(..., "--target-locale", "-l"),
    limit: int = typer.Option(5, "--limit", min=1, max=100),
    include_unapproved: bool = typer.Option(False, "--include-unapproved"),
    database: str = typer.Option("sqlite:///ai_translator.db", "--database"),
) -> None:
    """Find exact and token-overlap translation memory matches."""
    def action(session):
        project_row = resolve_project(session, project)
        matches = TranslationMemoryService(session).lookup(
            project_id=project_row.id, source_locale=source_locale,
            target_locale=target_locale, source_phrase=source_phrase,
            limit=limit, include_unapproved=include_unapproved,
        )
        return [
            f"{match.entry.id}\tsimilarity={match.similarity:.3f}\tapproved={match.entry.approved}\t"
            f"uses={match.entry.occurrence_count}\t{match.entry.source_text} → {match.entry.target_text}"
            for match in matches
        ]
    try:
        rows = _run_db_action(database, action)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if not rows:
        typer.echo("No matching memories found.")
    for row in rows:
        typer.echo(row)


@memory_app.command("conflicts")
def memory_conflicts(
    project: str = typer.Option(..., "--project", "-p"),
    source_locale: str | None = typer.Option(None, "--source-locale"),
    target_locale: str | None = typer.Option(None, "--target-locale", "-l"),
    database: str = typer.Option("sqlite:///ai_translator.db", "--database"),
) -> None:
    """List repeated source phrases with different preferred translations."""
    def action(session):
        project_row = resolve_project(session, project)
        groups = TranslationMemoryService(session).conflicts(
            project_row.id, source_locale=source_locale, target_locale=target_locale,
        )
        return [
            f"{group[0].source_text!r}: " + "; ".join(
                f"{entry.target_text!r} approved={entry.approved} uses={entry.occurrence_count} [{entry.id}]"
                for entry in group
            ) for group in groups
        ]
    try:
        rows = _run_db_action(database, action)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if not rows:
        typer.echo("No translation memory conflicts.")
    for row in rows:
        typer.echo(row)


@memory_app.command("approve")
def approve_memory(
    memory_id: str = typer.Argument(...),
    database: str = typer.Option("sqlite:///ai_translator.db", "--database"),
) -> None:
    """Approve a memory for automatic prompt injection."""
    memory_uuid = _parse_uuid(memory_id, "memory_id")
    try:
        entry = _run_db_action(database, lambda session: _memory_approval_action(
            session, memory_uuid, True
        ))
        typer.echo(f"Approved memory {entry}")
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


@memory_app.command("unapprove")
def unapprove_memory(
    memory_id: str = typer.Argument(...),
    database: str = typer.Option("sqlite:///ai_translator.db", "--database"),
) -> None:
    """Exclude a memory from automatic prompt injection."""
    memory_uuid = _parse_uuid(memory_id, "memory_id")
    try:
        entry = _run_db_action(database, lambda session: _memory_approval_action(
            session, memory_uuid, False
        ))
        typer.echo(f"Unapproved memory {entry}")
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _memory_approval_action(session, memory_id: UUID, approved: bool) -> str:
    entry = TranslationMemoryService(session).set_approved(memory_id, approved)
    return str(entry.id)


@memory_app.command("remove")
def remove_memory(
    memory_id: str = typer.Argument(...),
    database: str = typer.Option("sqlite:///ai_translator.db", "--database"),
) -> None:
    """Remove a translation memory entry, for example to resolve a conflict."""
    memory_uuid = _parse_uuid(memory_id, "memory_id")
    removed = _run_db_action(database, lambda session: TranslationMemoryService(session).remove(memory_uuid))
    typer.echo("Translation memory removed." if removed else "No translation memory found.")


@memory_app.command("update")
def update_memory(
    memory_id: str = typer.Argument(...),
    preferred_translation: str = typer.Option(..., "--preferred-translation"),
    database: str = typer.Option("sqlite:///ai_translator.db", "--database"),
) -> None:
    """Change a memory's preferred translation."""
    memory_uuid = _parse_uuid(memory_id, "memory_id")
    def action(session):
        entry = session.get(TranslationMemory, memory_uuid)
        if entry is None:
            raise ValueError(f"Translation memory {memory_uuid} does not exist")
        if not preferred_translation.strip():
            raise ValueError("Preferred translation cannot be blank")
        entry.target_text = preferred_translation.strip()
        return str(entry.id)
    try:
        typer.echo(f"Updated memory { _run_db_action(database, action) }")
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


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
    book: str | None = typer.Option(None, "--book", help="Limit scripture translation to this book title"),
    chapter: int | None = typer.Option(None, "--chapter", min=1,
                                       help="Limit scripture translation to this chapter (requires --book)"),
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
                book=book,
                chapter=chapter,
                progress_callback=(lambda message: typer.echo(message, err=True)) if verbose else None,
            )
        finally:
            session.close()
            engine.dispose()
        for candidate in candidates:
            typer.echo(
                f"{candidate.model_alias} ({candidate.model_name}): {candidate.status}, "
                f"version {candidate.version_number}, {candidate.translated_chunks}/{candidate.total_chunks} "
                f"{candidate.scope_label} chunks "
                f"[id {candidate.version_id}]"
            )
            if candidate.error:
                typer.echo(f"  error: {candidate.error}", err=True)
    except TranslationError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _run_refinement_workflow(
    *,
    operation: str,
    version_ids: list[UUID],
    translation_config_path: Path,
    evaluation_config_path: Path,
    translation_models: str | None,
    evaluator_models: str | None,
    plugins: str | None,
    database: str,
    verbose: bool,
):
    try:
        translation_config = TranslationConfig.from_file(translation_config_path)
        evaluation_config = EvaluationConfig.from_file(evaluation_config_path)
    except (TranslationError, EvaluationError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    selected_translation_models = (
        [value.strip() for value in translation_models.split(",") if value.strip()]
        if translation_models else None
    )
    selected_evaluator_models = (
        [value.strip() for value in evaluator_models.split(",") if value.strip()]
        if evaluator_models else None
    )
    selected_plugins = [value.strip() for value in plugins.split(",") if value.strip()] if plugins else None
    engine = make_engine(database)
    session = make_session_factory(engine)()
    client = OllamaClient(translation_config.base_url, timeout_seconds=translation_config.timeout_seconds)
    callback = (lambda message: typer.echo(message, err=True)) if verbose else None
    try:
        service = TranslationRefinementService(session, client)
        if operation == "refinement":
            candidates = service.refine_version(
                translation_version_id=version_ids[0], config=translation_config,
                model_selection=selected_translation_models, progress_callback=callback,
            )
        else:
            candidates = service.synthesize_versions(
                translation_version_ids=version_ids, config=translation_config,
                model_selection=selected_translation_models, progress_callback=callback,
            )
        evaluation_client = OllamaClient(
            evaluation_config.base_url, timeout_seconds=evaluation_config.timeout_seconds,
        )
        evaluations = {}
        for candidate in candidates:
            if candidate.status in {"completed", "partial"}:
                registry, candidate_plugins = _evaluation_registry_for_version(
                    session, candidate.version_id, selected_plugins, evaluation_config,
                )
                evaluations[candidate.version_id] = EvaluationService(
                    session, evaluation_client, registry,
                ).evaluate_translation_version(
                    translation_version_id=candidate.version_id,
                    config=evaluation_config,
                    plugin_selection=candidate_plugins,
                    model_selection=selected_evaluator_models,
                    progress_callback=callback,
                )
        return candidates, evaluations
    except (TranslationError, EvaluationError) as exc:
        session.rollback()
        raise typer.BadParameter(str(exc)) from exc
    finally:
        session.close()
        engine.dispose()


def _evaluation_registry_for_version(session, version_id: UUID, selected_plugins, evaluation_config):
    """Use the Scripture plugin registry for Scripture documents when installed."""
    version = session.get(TranslationVersion, version_id)
    if version is None or version.translation.document.domain.casefold() != "scripture":
        return None, selected_plugins
    try:
        from scripture_plugin.registry import scripture_registry
    except ImportError:
        # The plugin is independently installable; core workflows remain usable
        # without it, while the Scripture CLI remains available when installed.
        return None, selected_plugins
    entries = GlossaryService(session).entries_for_scope(
        version.translation.document.project_id,
        version.translation.document.source_locale,
        version.translation.target_locale,
    )
    glossary_terms = tuple({
        "source_term": entry.source_term, "preferred_translation": entry.target_term,
        "target_locale": version.translation.target_locale, "notes": entry.notes or "",
        "glossary": entry.glossary.name,
    } for entry in entries)
    registry = scripture_registry(glossary_terms=glossary_terms)
    base_plugins = selected_plugins if selected_plugins is not None else list(evaluation_config.plugin_ids)
    scripture_plugins = [plugin_id for plugin_id in registry.ids() if plugin_id.startswith("scripture.")]
    return registry, list(dict.fromkeys([*base_plugins, *scripture_plugins]))


def _print_refinement_results(candidates, evaluations) -> None:
    for candidate in candidates:
        typer.echo(
            f"{candidate.operation} / {candidate.model_alias} ({candidate.model_name}): "
            f"{candidate.status}, version {candidate.version_number}, "
            f"{candidate.translated_chunks}/{candidate.total_chunks} chunks [id {candidate.version_id}]"
        )
        if candidate.error:
            typer.echo(f"  error: {candidate.error}", err=True)
        for evaluated in evaluations.get(candidate.version_id, []):
            score = "n/a" if evaluated.score is None else f"{evaluated.score:.2f}"
            typer.echo(
                f"  evaluated {evaluated.plugin_id} / {evaluated.model_alias}: "
                f"{evaluated.status}, score {score}, {evaluated.issue_count} issue(s), "
                f"evaluation version {evaluated.evaluation_version_id}"
            )


@app.command("refine")
def refine_translation(
    translation_version_id: str = typer.Argument(..., help="Candidate version with saved evaluation feedback"),
    config_file: Path = typer.Option(Path("ai-translator.toml"), "--config", "-c"),
    evaluation_config_file: Path = typer.Option(
        Path("ai-translator-evaluation.toml"), "--evaluation-config",
    ),
    models: str | None = typer.Option(None, "--models", help="Comma-separated configured revision model aliases"),
    evaluator_models: str | None = typer.Option(
        None, "--evaluator-models", help="Comma-separated configured evaluator model aliases",
    ),
    plugins: str | None = typer.Option(None, "--plugins", help="Comma-separated evaluator plugin IDs"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    database: str = typer.Option("sqlite:///ai_translator.db", "--database"),
) -> None:
    """Revise one candidate from its own saved evaluation feedback, then reevaluate it."""
    version_id = _parse_uuid(translation_version_id, "translation_version_id")
    candidates, evaluations = _run_refinement_workflow(
        operation="refinement", version_ids=[version_id],
        translation_config_path=config_file, evaluation_config_path=evaluation_config_file,
        translation_models=models, evaluator_models=evaluator_models, plugins=plugins,
        database=database, verbose=verbose,
    )
    _print_refinement_results(candidates, evaluations)


@app.command("synthesize")
def synthesize_translation(
    translation_version_ids: list[str] = typer.Argument(
        ..., help="At least two evaluated candidate translation version UUIDs",
    ),
    config_file: Path = typer.Option(Path("ai-translator.toml"), "--config", "-c"),
    evaluation_config_file: Path = typer.Option(
        Path("ai-translator-evaluation.toml"), "--evaluation-config",
    ),
    models: str | None = typer.Option(None, "--models", help="Comma-separated configured synthesis model aliases"),
    evaluator_models: str | None = typer.Option(
        None, "--evaluator-models", help="Comma-separated configured evaluator model aliases",
    ),
    plugins: str | None = typer.Option(None, "--plugins", help="Comma-separated evaluator plugin IDs"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    database: str = typer.Option("sqlite:///ai_translator.db", "--database"),
) -> None:
    """Synthesize a source-grounded candidate from multiple evaluated versions and reevaluate it."""
    version_ids = [_parse_uuid(value, "translation_version_id") for value in translation_version_ids]
    candidates, evaluations = _run_refinement_workflow(
        operation="synthesis", version_ids=version_ids,
        translation_config_path=config_file, evaluation_config_path=evaluation_config_file,
        translation_models=models, evaluator_models=evaluator_models, plugins=plugins,
        database=database, verbose=verbose,
    )
    _print_refinement_results(candidates, evaluations)


@app.command("iterate")
def iterate_translation(
    translation_version_id: str = typer.Argument(..., help="Evaluated baseline translation version UUID"),
    config_file: Path = typer.Option(Path("ai-translator.toml"), "--config", "-c"),
    evaluation_config_file: Path = typer.Option(Path("ai-translator-evaluation.toml"), "--evaluation-config"),
    max_iterations: int | None = typer.Option(None, "--max-iterations", min=1,
                                               help="Override project translation_iteration.max_iterations"),
    minimum_score_increase: float | None = typer.Option(None, "--minimum-score-increase", min=0),
    minimum_confidence_increase: float | None = typer.Option(None, "--minimum-confidence-increase", min=0),
    models: str | None = typer.Option(None, "--models"),
    evaluator_models: str | None = typer.Option(None, "--evaluator-models"),
    plugins: str | None = typer.Option(None, "--plugins"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    database: str = typer.Option("sqlite:///ai_translator.db", "--database"),
) -> None:
    """Run bounded refine/evaluate cycles, recording every candidate in an experiment."""
    try:
        version_id = _parse_uuid(translation_version_id, "translation_version_id")
        translation_config = TranslationConfig.from_file(config_file)
        evaluation_config = EvaluationConfig.from_file(evaluation_config_file)
        select_list = lambda value: [part.strip() for part in value.split(",") if part.strip()] if value else None
        callback = (lambda message: typer.echo(message, err=True)) if verbose else None
        engine = make_engine(database)
        session = make_session_factory(engine)()
        try:
            iteration_registry, iteration_plugins = _evaluation_registry_for_version(
                session, version_id, select_list(plugins), evaluation_config,
            )
            outcome = TranslationIterationService(
                session,
                OllamaClient(translation_config.base_url, timeout_seconds=translation_config.timeout_seconds),
                OllamaClient(evaluation_config.base_url, timeout_seconds=evaluation_config.timeout_seconds),
            ).run(
                baseline_version_id=version_id, translation_config=translation_config,
                evaluation_config=evaluation_config, max_iterations=max_iterations,
                minimum_score_increase=minimum_score_increase,
                minimum_confidence_increase=minimum_confidence_increase,
                translation_models=select_list(models), evaluator_models=select_list(evaluator_models),
                plugins=iteration_plugins, evaluation_registry=iteration_registry, progress=callback,
            )
        finally:
            session.close()
            engine.dispose()
        typer.echo(f"Experiment {outcome.experiment_id}: {outcome.iterations} iteration(s), "
                   f"selected version {outcome.selected_version_id}, score {outcome.score:.2f}, "
                   f"confidence {outcome.confidence:.3f}; converged={outcome.converged} ({outcome.reason})")
    except (TranslationError, EvaluationError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc


def _emit_json_report(report: dict, output: Path | None = None) -> None:
    rendered = json.dumps(report, ensure_ascii=False, indent=2, default=str)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
        typer.echo(f"Report written to {output}")
    else:
        typer.echo(rendered)


@review_app.command("compare")
def compare_translation_versions(
    version_ids: list[str] = typer.Argument(..., help="Translation version UUIDs to compare"),
    database: str = typer.Option("sqlite:///ai_translator.db", "--database"),
) -> None:
    """Compare post-evaluation results; the report does not automatically select a winner."""
    try:
        ids = [_parse_uuid(value, "translation_version_id") for value in version_ids]
        engine = make_engine(database)
        session = make_session_factory(engine)()
        try:
            _emit_json_report(comparison_report(session, ids))
        finally:
            session.close()
            engine.dispose()
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


@report_app.command("lineage")
def report_translation_lineage(
    version_id: str = typer.Argument(...),
    output: Path | None = typer.Option(None, "--output", "-o"),
    database: str = typer.Option("sqlite:///ai_translator.db", "--database"),
) -> None:
    """Show parent and source-candidate artifacts for a translation version."""
    engine = make_engine(database)
    session = make_session_factory(engine)()
    try:
        _emit_json_report(lineage_report(session, _parse_uuid(version_id, "version_id")), output)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    finally:
        session.close()
        engine.dispose()


@report_app.command("convergence")
def report_translation_convergence(
    experiment_id: str = typer.Argument(...),
    output: Path | None = typer.Option(None, "--output", "-o"),
    database: str = typer.Option("sqlite:///ai_translator.db", "--database"),
) -> None:
    """Show experiment iteration history and convergence configuration."""
    engine = make_engine(database)
    session = make_session_factory(engine)()
    try:
        _emit_json_report(convergence_report(session, _parse_uuid(experiment_id, "experiment_id")), output)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    finally:
        session.close()
        engine.dispose()


@report_app.command("improvement")
def report_translation_improvement(
    version_id: str = typer.Argument(...),
    baseline: str | None = typer.Option(None, "--baseline", help="Explicit baseline version UUID"),
    output: Path | None = typer.Option(None, "--output", "-o"),
    database: str = typer.Option("sqlite:///ai_translator.db", "--database"),
) -> None:
    """Compare a refined candidate with its recorded parent or an explicit baseline."""
    engine = make_engine(database)
    session = make_session_factory(engine)()
    try:
        baseline_id = _parse_uuid(baseline, "baseline") if baseline else None
        _emit_json_report(improvement_report(
            session, _parse_uuid(version_id, "version_id"), baseline_id,
        ), output)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    finally:
        session.close()
        engine.dispose()


@report_app.command("side-by-side")
def report_side_by_side_translations(
    version_ids: list[str] = typer.Argument(..., help="Translation version UUIDs for aligned comparison"),
    output: Path | None = typer.Option(None, "--output", "-o", help="Markdown file; omit to print to stdout"),
    database: str = typer.Option("sqlite:///ai_translator.db", "--database"),
) -> None:
    """Show source segments beside each selected translation version."""
    engine = make_engine(database)
    session = make_session_factory(engine)()
    try:
        report = side_by_side_report(
            session, [_parse_uuid(value, "version_id") for value in version_ids],
        )
        if output is None:
            typer.echo(report)
        else:
            destination = output.expanduser()
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(report, encoding="utf-8")
            typer.echo(f"Side-by-side report written to {destination}")
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    finally:
        session.close()
        engine.dispose()


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

