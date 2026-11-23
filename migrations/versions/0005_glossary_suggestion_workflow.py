"""Persist versioned glossary suggestion and review runs.

Revision ID: 0005_glossary_suggestion_workflow
Revises: 0004_glossary_memory_usage
"""
from alembic import op
import sqlalchemy as sa

revision = "0005_glossary_suggestion_workflow"
down_revision = "0004_glossary_memory_usage"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "glossary_suggestion_runs",
        sa.Column("translation_version_id", sa.Uuid(), nullable=False),
        sa.Column("model_run_id", sa.Uuid(), nullable=True),
        sa.Column("version_number", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=30), nullable=False),
        sa.Column("model_name", sa.String(length=200), nullable=False),
        sa.Column("model_digest", sa.String(length=200), nullable=True),
        sa.Column("prompt_hash", sa.String(length=64), nullable=False),
        sa.Column("prompt", sa.Text(), nullable=False),
        sa.Column("raw_response", sa.Text(), nullable=True),
        sa.Column("run_manifest", sa.JSON(), nullable=False),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["model_run_id"], ["model_runs.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["translation_version_id"], ["translation_versions.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("translation_version_id", "version_number",
                            name="uq_glossary_suggestion_run_version"),
    )
    op.create_index("ix_glossary_suggestion_runs_version_status", "glossary_suggestion_runs",
                    ["translation_version_id", "status"])
    op.create_table(
        "glossary_suggestions",
        sa.Column("suggestion_run_id", sa.Uuid(), nullable=False),
        sa.Column("project_id", sa.Uuid(), nullable=False),
        sa.Column("source_locale", sa.String(length=35), nullable=False),
        sa.Column("target_locale", sa.String(length=35), nullable=False),
        sa.Column("source_term", sa.String(length=500), nullable=False),
        sa.Column("preferred_translation", sa.String(length=500), nullable=False),
        sa.Column("occurrence_count", sa.Integer(), nullable=False),
        sa.Column("source_references", sa.JSON(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("reasoning", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=30), nullable=False),
        sa.Column("glossary_entry_id", sa.Uuid(), nullable=True),
        sa.Column("resolved_by", sa.String(length=200), nullable=True),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("resolution_note", sa.Text(), nullable=True),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["glossary_entry_id"], ["glossary_entries.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["suggestion_run_id"], ["glossary_suggestion_runs.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_glossary_suggestions_run_status", "glossary_suggestions",
                    ["suggestion_run_id", "status"])
    op.create_index("ix_glossary_suggestions_project_term", "glossary_suggestions",
                    ["project_id", "source_term"])


def downgrade() -> None:
    op.drop_index("ix_glossary_suggestions_project_term", table_name="glossary_suggestions")
    op.drop_index("ix_glossary_suggestions_run_status", table_name="glossary_suggestions")
    op.drop_table("glossary_suggestions")
    op.drop_index("ix_glossary_suggestion_runs_version_status", table_name="glossary_suggestion_runs")
    op.drop_table("glossary_suggestion_runs")
