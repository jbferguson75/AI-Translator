from __future__ import annotations

from pathlib import Path
from uuid import UUID

import typer
from ai_translator.db.models import Translation, TranslationVersion
from ai_translator.db.session import make_engine, make_session_factory
from ai_translator.evaluation.config import EvaluationConfig
from ai_translator.evaluation.service import EvaluationService
from ai_translator.glossary.service import GlossaryService
from ai_translator.translation.errors import OllamaError
from ai_translator.translation.ollama import OllamaClient

from .glossary import DoctrineGlossary
from .metadata import ScriptureMetadata
from .registry import scripture_registry
from .report import SCRIPTURE_PLUGIN_IDS, ScriptureReviewReport

app = typer.Typer(no_args_is_help=True, help="Scripture-specific translation evaluation and review")


@app.command("evaluators")
def list_evaluators() -> None:
    """List Scripture and core evaluators available through this plugin."""
    registry = scripture_registry()
    for plugin_id in registry.ids():
        plugin = registry.get(plugin_id)
        typer.echo(f"{plugin_id}\t{plugin.display_name}\tversion {plugin.version}")


@app.command("evaluate")
def evaluate(
    translation_version_id: str = typer.Argument(..., help="Immutable translation version UUID"),
    config_file: Path = typer.Option(Path("ai-translator-evaluation.toml"), "--config", "-c"),
    plugins: str | None = typer.Option(None, "--plugins", help="Comma-separated Scripture evaluator IDs"),
    models: str | None = typer.Option(None, "--models", help="Comma-separated configured evaluator aliases"),
    limit: int | None = typer.Option(None, "--limit", min=1, help="Evaluate only the first N segments"),
    author_metadata: Path | None = typer.Option(None, "--author-metadata", help="Override bundled author JSON"),
    doctrine_glossary: Path | None = typer.Option(None, "--doctrine-glossary", help="Override bundled doctrine JSON"),
    report: Path | None = typer.Option(None, "--report", help="Write a Markdown review report after evaluation"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    database: str = typer.Option("sqlite:///ai_translator.db", "--database"),
) -> None:
    """Run Scripture evaluators through the platform's existing evaluation service."""
    try:
        version_id = UUID(translation_version_id)
    except ValueError as exc:
        raise typer.BadParameter("translation_version_id must be a UUID") from exc
    try:
        config = EvaluationConfig.from_file(config_file)
        metadata = ScriptureMetadata.load(author_metadata)
        doctrine = DoctrineGlossary.load(doctrine_glossary)
    except (ValueError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    plugin_ids = ([value.strip() for value in plugins.split(",") if value.strip()]
                  if plugins else list(SCRIPTURE_PLUGIN_IDS))
    invalid = sorted(set(plugin_ids) - set(SCRIPTURE_PLUGIN_IDS))
    if invalid:
        raise typer.BadParameter("Unknown Scripture evaluator ID(s): " + ", ".join(invalid))
    model_selection = [value.strip() for value in models.split(",") if value.strip()] if models else None

    engine = make_engine(database)
    session = make_session_factory(engine)()
    report_text = None
    try:
        version = session.get(TranslationVersion, version_id)
        if version is None:
            raise typer.BadParameter(f"Translation version {version_id} does not exist")
        translation: Translation = version.translation
        if translation.document.domain.casefold() != "scripture":
            raise typer.BadParameter("This command accepts only documents whose domain is 'scripture'")
        glossary_records = _load_glossary_terms(session, translation)
        registry = scripture_registry(
            metadata=metadata, doctrine_glossary=doctrine, glossary_terms=glossary_records,
        )
        client = OllamaClient(config.base_url, timeout_seconds=config.timeout_seconds)
        candidates = EvaluationService(session, client, registry).evaluate_translation_version(
            translation_version_id=version_id,
            config=config,
            plugin_selection=plugin_ids,
            model_selection=model_selection,
            chunk_limit=limit,
            progress_callback=(lambda message: typer.echo(message, err=True)) if verbose else None,
        )
        if report is not None:
            report_text = ScriptureReviewReport(
                metadata=metadata, doctrine_glossary=doctrine,
            ).render(session, version_id, glossary_records)
    except OllamaError as exc:
        raise typer.BadParameter(str(exc)) from exc
    finally:
        session.close()
        engine.dispose()
    for candidate in candidates:
        score = "n/a" if candidate.score is None else f"{candidate.score:.2f}"
        confidence = "n/a" if candidate.confidence is None else f"{candidate.confidence:.3f}"
        typer.echo(
            f"{candidate.plugin_id} / {candidate.model_alias}: {candidate.status}; score {score}; "
            f"confidence {confidence}; issues {candidate.issue_count}; version "
            f"{candidate.version_number} [{candidate.evaluation_version_id}]"
        )
        if candidate.error:
            typer.echo(f"  error: {candidate.error}", err=True)
    if report_text is not None and report is not None:
        output = report.expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report_text, encoding="utf-8")
        typer.echo(f"Scripture review report written to {output}")


@app.command("review-report")
def review_report(
    translation_version_id: str = typer.Argument(..., help="Immutable translation version UUID"),
    output: Path | None = typer.Option(None, "--output", "-o", help="Markdown path; omit to print to stdout"),
    author_metadata: Path | None = typer.Option(None, "--author-metadata"),
    doctrine_glossary: Path | None = typer.Option(None, "--doctrine-glossary"),
    database: str = typer.Option("sqlite:///ai_translator.db", "--database"),
) -> None:
    """Build a report from saved evaluator results plus a deterministic drift scan."""
    try:
        version_id = UUID(translation_version_id)
        metadata = ScriptureMetadata.load(author_metadata)
        doctrine = DoctrineGlossary.load(doctrine_glossary)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    engine = make_engine(database)
    session = make_session_factory(engine)()
    try:
        version = session.get(TranslationVersion, version_id)
        if version is None:
            raise typer.BadParameter(f"Translation version {version_id} does not exist")
        translation = version.translation
        if translation.document.domain.casefold() != "scripture":
            raise typer.BadParameter("This command accepts only documents whose domain is 'scripture'")
        terms = _load_glossary_terms(session, translation)
        content = ScriptureReviewReport(metadata=metadata, doctrine_glossary=doctrine).render(
            session, version_id, terms,
        )
    finally:
        session.close()
        engine.dispose()
    if output is None:
        typer.echo(content)
    else:
        path = output.expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        typer.echo(f"Scripture review report written to {path}")


def _load_glossary_terms(session, translation: Translation) -> tuple[dict[str, str], ...]:
    entries = GlossaryService(session).entries_for_scope(
        translation.document.project_id,
        translation.document.source_locale,
        translation.target_locale,
    )
    result: list[dict[str, str]] = []
    for entry in entries:
        result.append({
            "source_term": entry.source_term,
            "preferred_translation": entry.target_term,
            "target_locale": translation.target_locale,
            "notes": entry.notes or "",
            "priority": str(entry.priority),
            "glossary": entry.glossary.name,
        })
    return tuple(result)


def run() -> None:
    app()


if __name__ == "__main__":
    run()
