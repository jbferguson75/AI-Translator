"""Add artifact lineage and structured evaluator feedback.

Revision ID: 0007_translation_lineage_feedback
Revises: 0006_human_review_workflow
"""
from __future__ import annotations

import json
from uuid import UUID

import sqlalchemy as sa
from alembic import op

revision = "0007_translation_lineage_feedback"
down_revision = "0006_human_review_workflow"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # The tables are append-only. Drop SQLite guards while adding/backfilling
    # metadata; batch mode rebuilds the table to add the optional prompt FK.
    op.execute(sa.text("DROP TRIGGER IF EXISTS trg_translation_versions_no_update"))
    op.execute(sa.text("DROP TRIGGER IF EXISTS trg_translation_versions_no_delete"))
    with op.batch_alter_table("translation_versions") as batch:
        batch.add_column(sa.Column(
            "artifact_kind", sa.String(length=40), nullable=False,
            server_default="ORIGINAL_TRANSLATION",
        ))
        batch.add_column(sa.Column("prompt_version_id", sa.Uuid(), nullable=True))
        batch.create_foreign_key(
            "fk_translation_versions_prompt_version_id_prompt_versions",
            "prompt_versions", ["prompt_version_id"], ["id"], ondelete="RESTRICT",
        )
        batch.create_check_constraint(
            "ck_translation_versions_artifact_kind",
            "artifact_kind IN ('ORIGINAL_TRANSLATION', 'REFINED_TRANSLATION', "
            "'SYNTHESIZED_TRANSLATION', 'HUMAN_EDIT', 'FINAL_APPROVED')",
        )

    op.create_index(
        "ix_translation_versions_artifact_kind", "translation_versions", ["artifact_kind", "created_at"],
    )
    op.create_table(
        "translation_version_parents",
        sa.Column("child_version_id", sa.Uuid(), nullable=False),
        sa.Column("parent_version_id", sa.Uuid(), nullable=False),
        sa.Column("relationship_kind", sa.String(length=30), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["child_version_id"], ["translation_versions.id"], ondelete="RESTRICT",
            name="fk_translation_version_parents_child",
        ),
        sa.ForeignKeyConstraint(
            ["parent_version_id"], ["translation_versions.id"], ondelete="RESTRICT",
            name="fk_translation_version_parents_parent",
        ),
        sa.PrimaryKeyConstraint("child_version_id", "parent_version_id", "relationship_kind"),
        sa.CheckConstraint(
            "relationship_kind IN ('parent', 'source_candidate')",
            name="ck_translation_version_parents_kind",
        ),
    )
    op.create_index(
        "ix_translation_version_parents_parent", "translation_version_parents",
        ["parent_version_id", "child_version_id"],
    )

    bind = op.get_bind()
    versions = sa.Table("translation_versions", sa.MetaData(), autoload_with=bind)
    rows = list(bind.execute(sa.select(versions)).mappings())
    for row in rows:
        manifest = row["run_manifest"] or {}
        if isinstance(manifest, str):
            try:
                manifest = json.loads(manifest)
            except json.JSONDecodeError:
                manifest = {}
        operation = manifest.get("operation")
        if row["author_kind"] == "human":
            artifact_kind = "FINAL_APPROVED"
        elif operation == "synthesis":
            artifact_kind = "SYNTHESIZED_TRANSLATION"
        elif operation == "refinement":
            artifact_kind = "REFINED_TRANSLATION"
        else:
            artifact_kind = "ORIGINAL_TRANSLATION"
        prompt_id = manifest.get("prompt_version_id")
        values = {"artifact_kind": artifact_kind}
        if prompt_id:
            values["prompt_version_id"] = UUID(str(prompt_id))
        bind.execute(versions.update().where(versions.c.id == row["id"]).values(**values))
        parent_id = row["parent_version_id"]
        if parent_id:
            bind.execute(sa.text(
                "INSERT OR IGNORE INTO translation_version_parents "
                "(child_version_id, parent_version_id, relationship_kind, ordinal, created_at) "
                "VALUES (:child, :parent, 'parent', 0, :created)"
            ), {"child": row["id"], "parent": parent_id, "created": row["created_at"]})
        input_ids = manifest.get("input_translation_version_ids", [])
        if isinstance(input_ids, list):
            for ordinal, candidate_id in enumerate(input_ids):
                if str(candidate_id).replace("-", "") == str(row["id"]).replace("-", ""):
                    continue
                bind.execute(sa.text(
                    "INSERT OR IGNORE INTO translation_version_parents "
                    "(child_version_id, parent_version_id, relationship_kind, ordinal, created_at) "
                    "VALUES (:child, :parent, 'source_candidate', :ordinal, :created)"
                ), {"child": row["id"], "parent": UUID(str(candidate_id)).hex, "ordinal": ordinal,
                    "created": row["created_at"]})

    # Add machine-readable strengths and structured issue metadata while
    # preserving the legacy string issue array for compatibility/search.
    op.execute(sa.text("DROP TRIGGER IF EXISTS trg_evaluation_results_no_update"))
    op.add_column("evaluation_results", sa.Column(
        "strengths", sa.JSON(), nullable=False, server_default="[]",
    ))
    op.add_column("evaluation_results", sa.Column(
        "issue_details", sa.JSON(), nullable=False, server_default="[]",
    ))
    results = sa.Table("evaluation_results", sa.MetaData(), autoload_with=bind)
    for row in bind.execute(sa.select(results)).mappings():
        issues = row["detected_issues"] or []
        if isinstance(issues, str):
            try:
                issues = json.loads(issues)
            except json.JSONDecodeError:
                issues = []
        details = [{"type": "unspecified", "severity": "unspecified", "description": issue}
                   for issue in issues if isinstance(issue, str)]
        bind.execute(results.update().where(results.c.id == row["id"]).values(
            strengths=[], issue_details=details,
        ))

    op.execute(sa.text(
        "UPDATE evaluation_results SET searchable_text = searchable_text || char(10) || "
        "COALESCE((SELECT group_concat(json_extract(value, '$.description'), char(10)) "
        "FROM json_each(evaluation_results.issue_details)), '') || char(10) || "
        "COALESCE((SELECT group_concat(value, char(10)) "
        "FROM json_each(evaluation_results.strengths)), '')"
    ))
    op.execute(sa.text(
        "CREATE TRIGGER trg_translation_versions_no_update BEFORE UPDATE ON translation_versions "
        "BEGIN SELECT RAISE(ABORT, 'record is immutable; append a new version'); END"
    ))
    op.execute(sa.text(
        "CREATE TRIGGER trg_translation_versions_no_delete BEFORE DELETE ON translation_versions "
        "BEGIN SELECT RAISE(ABORT, 'record is immutable; append a new version'); END"
    ))
    op.execute(sa.text(
        "CREATE TRIGGER trg_translation_version_parents_no_update BEFORE UPDATE ON translation_version_parents "
        "BEGIN SELECT RAISE(ABORT, 'lineage records are immutable'); END"
    ))
    op.execute(sa.text(
        "CREATE TRIGGER trg_translation_version_parents_no_delete BEFORE DELETE ON translation_version_parents "
        "BEGIN SELECT RAISE(ABORT, 'lineage records are immutable'); END"
    ))
    op.execute(sa.text(
        "CREATE TRIGGER trg_evaluation_results_no_update BEFORE UPDATE ON evaluation_results "
        "BEGIN SELECT RAISE(ABORT, 'record is immutable; append a new version'); END"
    ))


def downgrade() -> None:
    for trigger in (
        "trg_translation_version_parents_no_update", "trg_translation_version_parents_no_delete",
        "trg_translation_versions_no_update", "trg_translation_versions_no_delete",
        "trg_evaluation_results_no_update",
    ):
        op.execute(sa.text(f"DROP TRIGGER IF EXISTS {trigger}"))
    op.drop_column("evaluation_results", "issue_details")
    op.drop_column("evaluation_results", "strengths")
    op.drop_index("ix_translation_version_parents_parent", table_name="translation_version_parents")
    op.drop_table("translation_version_parents")
    op.drop_index("ix_translation_versions_artifact_kind", table_name="translation_versions")
    with op.batch_alter_table("translation_versions") as batch:
        batch.drop_constraint(
            "ck_translation_versions_artifact_kind", type_="check",
        )
        batch.drop_constraint(
            "fk_translation_versions_prompt_version_id_prompt_versions", type_="foreignkey",
        )
        batch.drop_column("prompt_version_id")
        batch.drop_column("artifact_kind")
    op.execute(sa.text(
        "CREATE TRIGGER trg_translation_versions_no_update BEFORE UPDATE ON translation_versions "
        "BEGIN SELECT RAISE(ABORT, 'record is immutable; append a new version'); END"
    ))
    op.execute(sa.text(
        "CREATE TRIGGER trg_translation_versions_no_delete BEFORE DELETE ON translation_versions "
        "BEGIN SELECT RAISE(ABORT, 'record is immutable; append a new version'); END"
    ))
    op.execute(sa.text(
        "CREATE TRIGGER trg_evaluation_results_no_update BEFORE UPDATE ON evaluation_results "
        "BEGIN SELECT RAISE(ABORT, 'record is immutable; append a new version'); END"
    ))
