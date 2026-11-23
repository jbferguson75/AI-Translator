"""Add terminology priority and translation memory usage metadata.

Revision ID: 0004_glossary_memory_usage
Revises: 0003_restore_delete_guards
"""
from alembic import op
import sqlalchemy as sa

revision = "0004_glossary_memory_usage"
down_revision = "0003_restore_delete_guards"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("glossary_entries", sa.Column(
        "priority", sa.Integer(), server_default="0", nullable=False
    ))
    op.add_column("glossary_entries", sa.Column(
        "occurrence_count", sa.Integer(), server_default="0", nullable=False
    ))
    op.add_column("glossary_entries", sa.Column(
        "source_references", sa.JSON(), server_default=sa.text("'[]'"), nullable=False
    ))
    op.create_index("ix_glossary_entries_term_priority", "glossary_entries", ["source_term", "priority"])
    op.add_column("translation_memories", sa.Column(
        "occurrence_count", sa.Integer(), server_default="0", nullable=False
    ))
    op.add_column("translation_memories", sa.Column(
        "source_references", sa.JSON(), server_default=sa.text("'[]'"), nullable=False
    ))


def downgrade() -> None:
    op.drop_column("translation_memories", "source_references")
    op.drop_column("translation_memories", "occurrence_count")
    op.drop_index("ix_glossary_entries_term_priority", table_name="glossary_entries")
    op.drop_column("glossary_entries", "source_references")
    op.drop_column("glossary_entries", "occurrence_count")
    op.drop_column("glossary_entries", "priority")
