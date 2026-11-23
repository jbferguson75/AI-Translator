# Database layer

## Layout

```text
src/ai_translator/db/
  base.py          Declarative base, UTC timestamps, source immutability guard
  models.py        SQLAlchemy 2.x entities and relationships
  repositories.py  Session-scoped repository classes
  session.py       SQLite engine and session factory
migrations/
  env.py           Alembic metadata and online/offline setup
  versions/        Checked-in schema revisions
tests/
  test_database.py
  test_migrations.py
```

## Setup

Install the project and development dependencies, then create/update the local database and a project:

```sh
python3.15 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
ai-translator db upgrade
ai-translator project create "Book of Mormon - Simplified" --source-locale en
```

Use `make_engine()` and `make_session_factory()` from `ai_translator.db.session` to create sessions. New projects default to source locale `en`; set `Project.default_source_locale` for a different project default. A service owns the transaction; repositories receive its `Session` and never commit independently. For a different database path, pass the same `--database` URL to both CLI commands or set `sqlalchemy.url` in `alembic.ini`.

Create later schema changes with `alembic revision --autogenerate -m "description"`, review the generated migration, and apply it with `alembic upgrade head`. Never edit an already applied revision. The initial migration adds SQLite triggers that reject edits/deletes to source structure and append-only translation, evaluation, review, prompt, and publication records. Source corrections are imported as new document snapshots.

## Versioning rules

- `Translation` identifies a document, target locale, and task. Its complete `TranslationVersion` snapshots use a per-translation monotonic version number and optional parent link.
- `Evaluation` identifies a translation evaluation series. Each `EvaluationVersion` points at the exact translation version it evaluated and has an independent version number and run manifest.
- Segment-level output belongs to a translation version. Review decisions are immutable `HumanReview` events and can point to a resulting human-authored translation version.
- The repository does not commit transactions. Keep model inference outside database write transactions and persist finished artifacts in short transactions.

Glossaries are project-scoped and may contain multiple locale-pair sets. Glossary entry priority, translation-memory approval, occurrence counts, and source references support deterministic injection and conflict review. Alembic revision `0004_glossary_memory_usage` adds those usage fields; see [GLOSSARY_MEMORY.md](GLOSSARY_MEMORY.md) for management commands and conflict policy.
