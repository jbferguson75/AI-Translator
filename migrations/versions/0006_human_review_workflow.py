"""Add versioned human decisions and an immutable review audit chain.

Revision ID: 0006_human_review_workflow
Revises: 0005_glossary_suggestion_workflow
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from uuid import UUID

import sqlalchemy as sa
from alembic import op

revision = "0006_human_review_workflow"
down_revision = "0005_glossary_suggestion_workflow"
branch_labels = None
depends_on = None


def _iso(value):
    if value is None:
        return None
    if not isinstance(value, datetime):
        value = datetime.fromisoformat(str(value))
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds")


def _uuid(value):
    return str(UUID(str(value))) if value is not None else None


def _digest(record: dict) -> str:
    payload = {
        "id": _uuid(record["id"]),
        "review_pass_id": _uuid(record["review_pass_id"]),
        "reviewer": record["reviewer"],
        "decision": record["decision"],
        "status": record["status"],
        "approval_date": _iso(record["approval_date"]),
        "chunk_id": _uuid(record["chunk_id"]),
        "comment": record["comment"],
        "proposed_text": record["proposed_text"],
        "review_notes": record["review_notes"],
        "correction": record["correction"],
        "override_reason": record["override_reason"],
        "resulting_translation_version_id": (
            _uuid(record["resulting_translation_version_id"])
        ),
        "supersedes_review_id": (
            _uuid(record["supersedes_review_id"])
        ),
        "previous_event_hash": record["previous_event_hash"],
        "created_at": _iso(record["created_at"]),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def upgrade() -> None:
    # Historical rows are append-only. Temporarily remove the SQLite UPDATE
    # guard only while deriving their new fields, then reinstall it below.
    op.execute(sa.text("DROP TRIGGER IF EXISTS trg_human_reviews_no_update"))
    op.add_column("human_reviews", sa.Column(
        "status", sa.String(length=30), nullable=False, server_default="pending",
    ))
    op.add_column("human_reviews", sa.Column("approval_date", sa.DateTime(timezone=True), nullable=True))
    op.add_column("human_reviews", sa.Column("review_notes", sa.Text(), nullable=True))
    op.add_column("human_reviews", sa.Column("correction", sa.Text(), nullable=True))
    op.add_column("human_reviews", sa.Column("override_reason", sa.Text(), nullable=True))
    # SQLite cannot ALTER TABLE to add a foreign-key constraint. Batch mode
    # rebuilds the table while preserving existing rows and indexes.
    with op.batch_alter_table("human_reviews") as batch:
        batch.add_column(sa.Column(
            "supersedes_review_id", sa.Uuid(),
            sa.ForeignKey("human_reviews.id", ondelete="RESTRICT", name="fk_human_reviews_supersedes"),
            nullable=True,
        ))
    op.add_column("human_reviews", sa.Column("previous_event_hash", sa.String(length=64), nullable=True))
    op.add_column("human_reviews", sa.Column(
        "event_hash", sa.String(length=64), nullable=False, server_default="",
    ))

    bind = op.get_bind()
    reviews = sa.Table("human_reviews", sa.MetaData(), autoload_with=bind)
    rows = [dict(row) for row in bind.execute(sa.select(reviews).order_by(
        reviews.c.review_pass_id, reviews.c.chunk_id, reviews.c.created_at, reviews.c.id,
    )).mappings()]
    latest_by_item: dict[tuple[object, object], dict] = {}
    for row in rows:
        old_decision = str(row["decision"] or "").casefold().strip()
        status = {
            "approve": "approved", "approved": "approved", "accept": "approved", "accepted": "approved",
            "reject": "rejected", "rejected": "rejected",
            "needs_revision": "needs_revision", "needs revision": "needs_revision",
            "revise": "needs_revision", "revision": "needs_revision",
            "pending": "pending",
        }.get(old_decision, "needs_revision")
        key = (row["review_pass_id"], row["chunk_id"])
        previous = latest_by_item.get(key)
        row["status"] = status
        row["approval_date"] = row["created_at"] if status == "approved" else None
        row["review_notes"] = row["comment"]
        row["correction"] = row["proposed_text"]
        row["override_reason"] = None
        row["supersedes_review_id"] = previous["id"] if previous else None
        row["previous_event_hash"] = previous["event_hash"] if previous else None
        row["event_hash"] = _digest(row)
        bind.execute(reviews.update().where(reviews.c.id == row["id"]).values(
            status=row["status"], approval_date=row["approval_date"],
            review_notes=row["review_notes"], correction=row["correction"],
            override_reason=row["override_reason"],
            supersedes_review_id=row["supersedes_review_id"],
            previous_event_hash=row["previous_event_hash"], event_hash=row["event_hash"],
        ))
        latest_by_item[key] = row

    op.create_index("ix_human_reviews_pass_chunk_created", "human_reviews",
                    ["review_pass_id", "chunk_id", "created_at"])
    op.create_index("ix_human_reviews_status_created", "human_reviews", ["status", "created_at"])
    op.execute(sa.text(
        "CREATE TRIGGER trg_human_reviews_validate_insert BEFORE INSERT ON human_reviews "
        "WHEN NEW.status NOT IN ('pending', 'approved', 'rejected', 'needs_revision') "
        "OR (NEW.status = 'approved' AND NEW.approval_date IS NULL) "
        "OR (NEW.status != 'approved' AND NEW.approval_date IS NOT NULL) "
        "OR length(NEW.event_hash) != 64 "
        "BEGIN SELECT RAISE(ABORT, 'invalid human review event'); END"
    ))
    op.execute(sa.text(
        "CREATE TRIGGER trg_human_reviews_supersedes_scope BEFORE INSERT ON human_reviews "
        "WHEN NEW.supersedes_review_id IS NOT NULL AND NOT EXISTS ("
        "SELECT 1 FROM human_reviews prior WHERE prior.id = NEW.supersedes_review_id "
        "AND prior.review_pass_id = NEW.review_pass_id "
        "AND prior.chunk_id IS NEW.chunk_id) "
        "BEGIN SELECT RAISE(ABORT, 'superseded review must be for the same pass and chunk'); END"
    ))
    op.execute(sa.text(
        "CREATE TRIGGER trg_human_reviews_no_update BEFORE UPDATE ON human_reviews "
        "BEGIN SELECT RAISE(ABORT, 'review events are immutable; append a new event'); END"
    ))
    op.execute(sa.text(
        "CREATE TRIGGER trg_human_reviews_no_delete BEFORE DELETE ON human_reviews "
        "BEGIN SELECT RAISE(ABORT, 'record is immutable; append a new version or event'); END"
    ))


def downgrade() -> None:
    op.execute(sa.text("DROP TRIGGER IF EXISTS trg_human_reviews_validate_insert"))
    op.execute(sa.text("DROP TRIGGER IF EXISTS trg_human_reviews_supersedes_scope"))
    op.execute(sa.text("DROP TRIGGER IF EXISTS trg_human_reviews_no_update"))
    op.drop_index("ix_human_reviews_status_created", table_name="human_reviews")
    op.drop_index("ix_human_reviews_pass_chunk_created", table_name="human_reviews")
    op.drop_column("human_reviews", "event_hash")
    op.drop_column("human_reviews", "previous_event_hash")
    op.drop_column("human_reviews", "supersedes_review_id")
    op.drop_column("human_reviews", "override_reason")
    op.drop_column("human_reviews", "correction")
    op.drop_column("human_reviews", "review_notes")
    op.drop_column("human_reviews", "approval_date")
    op.drop_column("human_reviews", "status")
    op.execute(sa.text(
        "CREATE TRIGGER trg_human_reviews_no_update BEFORE UPDATE ON human_reviews "
        "BEGIN SELECT RAISE(ABORT, 'record is immutable; append a new version or event'); END"
    ))
    op.execute(sa.text(
        "CREATE TRIGGER trg_human_reviews_no_delete BEFORE DELETE ON human_reviews "
        "BEGIN SELECT RAISE(ABORT, 'record is immutable; append a new version or event'); END"
    ))
