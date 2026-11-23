"""Add complete, searchable plugin evaluation payloads.

Revision ID: 0002_eval_payload
Revises: ba0de84b1038
"""
from alembic import op
import sqlalchemy as sa

revision = "0002_eval_payload"
down_revision = "ba0de84b1038"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # SQLite supports additive ALTER TABLE here. Avoid a batch table rebuild so
    # the source/evaluation append-only triggers remain attached to the table.
    op.add_column("evaluation_results", sa.Column(
        "plugin_id", sa.String(length=100), server_default="legacy", nullable=False
    ))
    op.add_column("evaluation_results", sa.Column("score", sa.Float(), nullable=True))
    op.add_column("evaluation_results", sa.Column("confidence", sa.Float(), nullable=True))
    op.add_column("evaluation_results", sa.Column("reasoning", sa.Text(), nullable=True))
    op.add_column("evaluation_results", sa.Column(
        "recommendations", sa.JSON(), server_default=sa.text("'[]'"), nullable=False
    ))
    op.add_column("evaluation_results", sa.Column(
        "detected_issues", sa.JSON(), server_default=sa.text("'[]'"), nullable=False
    ))
    op.add_column("evaluation_results", sa.Column(
        "searchable_text", sa.Text(), server_default="", nullable=False
    ))
    op.create_index("ix_evaluation_results_plugin_metric", "evaluation_results",
                    ["plugin_id", "metric_key"])


def downgrade() -> None:
    op.execute(sa.text("DROP TRIGGER IF EXISTS trg_evaluation_results_no_update"))
    op.execute(sa.text("DROP TRIGGER IF EXISTS trg_evaluation_results_no_delete"))
    with op.batch_alter_table("evaluation_results") as batch_op:
        batch_op.drop_index("ix_evaluation_results_plugin_metric")
        batch_op.drop_column("searchable_text")
        batch_op.drop_column("detected_issues")
        batch_op.drop_column("recommendations")
        batch_op.drop_column("reasoning")
        batch_op.drop_column("confidence")
        batch_op.drop_column("score")
        batch_op.drop_column("plugin_id")
    op.execute(sa.text(
        "CREATE TRIGGER trg_evaluation_results_no_update BEFORE UPDATE ON evaluation_results "
        "BEGIN SELECT RAISE(ABORT, 'record is immutable; append a new evaluation version'); END"
    ))
    op.execute(sa.text(
        "CREATE TRIGGER trg_evaluation_results_no_delete BEFORE DELETE ON evaluation_results "
        "BEGIN SELECT RAISE(ABORT, 'record is immutable; append a new evaluation version'); END"
    ))

