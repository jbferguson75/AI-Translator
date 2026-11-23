from __future__ import annotations

from pathlib import Path
from uuid import UUID

import typer

from ai_translator.db.session import make_engine, make_session_factory
from ai_translator.evaluation.config import EvaluationConfig
from ai_translator.evaluation.errors import EvaluationError
from ai_translator.translation.ollama import OllamaClient

from .glossary import DoctrineGlossary
from .metadata import ScriptureMetadata
from .review_report import HierarchicalConsistencyReport
from .review_service import HierarchicalReviewService, LEVEL_NAMES, ReviewSummary

app = typer.Typer(no_args_is_help=True, help="Hierarchical Scripture review")


def _run_review(translation_version: str, level: int, *, config_file: Path, models: str | None,
                book: str | None, chapter: int | None, verse: int | None,
                database: str, verbose: bool, output: Path | None,
                author_metadata: Path | None, doctrine_glossary: Path | None) -> None:
    try:
        version_id = UUID(translation_version)
        config = EvaluationConfig.from_file(config_file)
        metadata = ScriptureMetadata.load(author_metadata)
        doctrine = DoctrineGlossary.load(doctrine_glossary)
    except (ValueError, OSError, EvaluationError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    model_selection = [item.strip() for item in models.split(",") if item.strip()] if models else None
    engine = make_engine(database)
    session = make_session_factory(engine)()
    try:
        service = HierarchicalReviewService(
            session, OllamaClient(config.base_url, timeout_seconds=config.timeout_seconds),
            metadata=metadata, doctrine_glossary=doctrine,
        )
        summary = service.run(
            translation_version_id=version_id, level=level, config=config,
            model_selection=model_selection, book=book, chapter=chapter, verse=verse,
            progress_callback=(lambda message: typer.echo(message, err=True)) if verbose else None,
        )
        report_content = (HierarchicalConsistencyReport().render(session, version_id, summary.run_id)
                          if output else None)
    except (ValueError, OSError, EvaluationError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    finally:
        session.close()
        engine.dispose()
    _print_summary(summary)
    if output is not None and report_content is not None:
        path = output.expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report_content, encoding="utf-8")
        typer.echo(f"Consistency report written to {path}")


@app.command("verse")
def verse_review(
    translation_version: str = typer.Argument(..., help="Immutable translation version UUID"),
    book: str | None = typer.Option(None, "--book"),
    chapter: int | None = typer.Option(None, "--chapter", min=1),
    verse: int | None = typer.Option(None, "--verse", min=1),
    config: Path = typer.Option(Path("ai-translator-evaluation.toml"), "--config", "-c"),
    models: str | None = typer.Option(None, "--models"),
    author_metadata: Path | None = typer.Option(None, "--author-metadata"),
    doctrine_glossary: Path | None = typer.Option(None, "--doctrine-glossary"),
    report: Path | None = typer.Option(None, "--report", help="Write a consistency report for this review run"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    database: str = typer.Option("sqlite:///ai_translator.db", "--database"),
) -> None:
    """Review every verse, or narrow to a book/chapter/verse."""
    if verse is not None and chapter is None:
        raise typer.BadParameter("--verse requires --chapter")
    if chapter is not None and not book:
        raise typer.BadParameter("--chapter requires --book")
    _run_review(translation_version, 1, config_file=config, models=models, book=book,
                chapter=chapter, verse=verse, database=database, verbose=verbose, output=report,
                author_metadata=author_metadata, doctrine_glossary=doctrine_glossary)


@app.command("chapter")
def chapter_review(
    translation_version: str = typer.Argument(...),
    book: str = typer.Option(..., "--book"),
    chapter: int = typer.Option(..., "--chapter", min=1),
    config: Path = typer.Option(Path("ai-translator-evaluation.toml"), "--config", "-c"),
    models: str | None = typer.Option(None, "--models"),
    author_metadata: Path | None = typer.Option(None, "--author-metadata"),
    doctrine_glossary: Path | None = typer.Option(None, "--doctrine-glossary"),
    report: Path | None = typer.Option(None, "--report"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    database: str = typer.Option("sqlite:///ai_translator.db", "--database"),
) -> None:
    """Review one chapter for consistency, flow, context, and pronouns."""
    _run_review(translation_version, 2, config_file=config, models=models, book=book,
                chapter=chapter, verse=None, database=database, verbose=verbose, output=report,
                author_metadata=author_metadata, doctrine_glossary=doctrine_glossary)


@app.command("book")
def book_review(
    translation_version: str = typer.Argument(...),
    book: str = typer.Option(..., "--book"),
    config: Path = typer.Option(Path("ai-translator-evaluation.toml"), "--config", "-c"),
    models: str | None = typer.Option(None, "--models"),
    author_metadata: Path | None = typer.Option(None, "--author-metadata"),
    doctrine_glossary: Path | None = typer.Option(None, "--doctrine-glossary"),
    report: Path | None = typer.Option(None, "--report"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    database: str = typer.Option("sqlite:///ai_translator.db", "--database"),
) -> None:
    """Review a whole book for author voice and terminology consistency."""
    _run_review(translation_version, 3, config_file=config, models=models, book=book,
                chapter=None, verse=None, database=database, verbose=verbose, output=report,
                author_metadata=author_metadata, doctrine_glossary=doctrine_glossary)


@app.command("global")
def global_review(
    translation_version: str = typer.Argument(...),
    config: Path = typer.Option(Path("ai-translator-evaluation.toml"), "--config", "-c"),
    models: str | None = typer.Option(None, "--models"),
    author_metadata: Path | None = typer.Option(None, "--author-metadata"),
    doctrine_glossary: Path | None = typer.Option(None, "--doctrine-glossary"),
    report: Path | None = typer.Option(None, "--report"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    database: str = typer.Option("sqlite:///ai_translator.db", "--database"),
) -> None:
    """Review the whole document and compare project-wide terminology usage."""
    _run_review(translation_version, 4, config_file=config, models=models, book=None,
                chapter=None, verse=None, database=database, verbose=verbose, output=report,
                author_metadata=author_metadata, doctrine_glossary=doctrine_glossary)


@app.command("all")
def review_all_levels(
    translation_version: str = typer.Argument(...),
    levels: str = typer.Option("1,2,3,4", "--levels", help="Comma-separated review levels"),
    config_file: Path = typer.Option(Path("ai-translator-evaluation.toml"), "--config", "-c"),
    models: str | None = typer.Option(None, "--models"),
    author_metadata: Path | None = typer.Option(None, "--author-metadata"),
    doctrine_glossary: Path | None = typer.Option(None, "--doctrine-glossary"),
    report: Path | None = typer.Option(None, "--report"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    database: str = typer.Option("sqlite:///ai_translator.db", "--database"),
) -> None:
    """Run all four review levels across the full translation version."""
    try:
        version_id = UUID(translation_version)
        selected_levels = tuple(int(value.strip()) for value in levels.split(",") if value.strip())
        if not selected_levels or any(value not in LEVEL_NAMES for value in selected_levels):
            raise ValueError("levels must contain values from 1 to 4")
        if len(set(selected_levels)) != len(selected_levels):
            raise ValueError("levels must not contain duplicates")
        config = EvaluationConfig.from_file(config_file)
        metadata = ScriptureMetadata.load(author_metadata)
        doctrine = DoctrineGlossary.load(doctrine_glossary)
    except (ValueError, OSError, EvaluationError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    model_selection = [item.strip() for item in models.split(",") if item.strip()] if models else None
    engine = make_engine(database)
    session = make_session_factory(engine)()
    report_content = None
    try:
        service = HierarchicalReviewService(
            session, OllamaClient(config.base_url, timeout_seconds=config.timeout_seconds),
            metadata=metadata, doctrine_glossary=doctrine,
        )
        summary = service.run_all(
            translation_version_id=version_id, levels=selected_levels,
            config=config, model_selection=model_selection,
            progress_callback=(lambda message: typer.echo(message, err=True)) if verbose else None,
        )
        if report:
            report_content = HierarchicalConsistencyReport().render(session, version_id, summary.run_id)
    except (ValueError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    finally:
        session.close()
        engine.dispose()
    _print_summary(summary)
    if report and report_content:
        path = report.expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report_content, encoding="utf-8")
        typer.echo(f"Consistency report written to {path}")


@app.command("consistency-report")
def consistency_report(
    translation_version: str = typer.Argument(...),
    run_id: str | None = typer.Option(None, "--run-id", help="Specific hierarchical review run UUID"),
    output: Path | None = typer.Option(None, "--output", "-o"),
    database: str = typer.Option("sqlite:///ai_translator.db", "--database"),
) -> None:
    """Render the saved scoped scores, issues, and terminology drift findings."""
    try:
        version_id = UUID(translation_version)
        parsed_run = UUID(run_id) if run_id else None
    except ValueError as exc:
        raise typer.BadParameter("translation_version and run_id must be UUIDs") from exc
    engine = make_engine(database)
    session = make_session_factory(engine)()
    try:
        content = HierarchicalConsistencyReport().render(session, version_id, parsed_run)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    finally:
        session.close()
        engine.dispose()
    if output:
        path = output.expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        typer.echo(f"Consistency report written to {path}")
    else:
        typer.echo(content)


def _print_summary(summary: ReviewSummary) -> None:
    typer.echo(f"Review run {summary.run_id}; translation version {summary.translation_version_id}")
    for level, score in summary.level_scores.items():
        typer.echo(f"Level {level} ({LEVEL_NAMES[level]}): {score if score is not None else 'n/a'}")
    typer.echo(f"Overall score: {summary.overall_score if summary.overall_score is not None else 'n/a'}")
    for result in summary.candidates:
        score = "n/a" if result.score is None else f"{result.score:.2f}"
        typer.echo(
            f"  {result.level_name} {result.scope_label} / {result.plugin_id} / {result.model_alias}: "
            f"{result.status}, score {score}, {result.issue_count} issue(s), "
            f"evaluation version {result.version_number} [{result.evaluation_version_id}]"
        )
        if result.error:
            typer.echo(f"    error: {result.error}", err=True)

