"""Restore append-only DELETE guards for immutable records.

Revision ID: 0003_restore_delete_guards
Revises: 0002_eval_payload
"""
from alembic import op
import sqlalchemy as sa

revision = "0003_restore_delete_guards"
down_revision = "0002_eval_payload"
branch_labels = None
depends_on = None


_IMMUTABLE_TABLES = (
    "books", "chapters", "verses", "chunks", "translation_versions",
    "translation_segments", "evaluation_versions", "evaluation_results",
    "human_reviews", "prompt_versions", "published_outputs",
)


def upgrade() -> None:
    for table in _IMMUTABLE_TABLES:
        op.execute(sa.text(
            f"CREATE TRIGGER IF NOT EXISTS trg_{table}_no_delete BEFORE DELETE ON {table} "
            "BEGIN SELECT RAISE(ABORT, 'record is immutable; append a new version or event'); END"
        ))
    op.execute(sa.text(
        "CREATE TRIGGER IF NOT EXISTS trg_documents_source_no_delete BEFORE DELETE ON documents "
        "BEGIN SELECT RAISE(ABORT, 'source snapshots are immutable'); END"
    ))


def downgrade() -> None:
    # These guards are also part of the original schema's integrity contract.
    # Keep them installed when downgrading the repair revision.
    pass
