from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db.models import HumanReview


@dataclass(frozen=True, slots=True)
class AuditVerification:
    review_pass_id: UUID
    event_count: int
    valid: bool
    errors: tuple[str, ...]


class AuditTrailService:
    """Append-only, per-review-item SHA-256 event chains."""

    def __init__(self, session: Session):
        self.session = session

    def latest_for_item(self, review_pass_id: UUID, chunk_id: UUID | None) -> HumanReview | None:
        statement = select(HumanReview).where(
            HumanReview.review_pass_id == review_pass_id,
            HumanReview.chunk_id == chunk_id,
        ).order_by(HumanReview.created_at.desc(), HumanReview.id.desc()).limit(1)
        return self.session.scalar(statement)

    def append_event(
        self,
        *,
        review_pass_id: UUID,
        chunk_id: UUID | None,
        reviewer: str,
        status: str,
        review_notes: str | None = None,
        correction: str | None = None,
        override_reason: str | None = None,
        approval_date: datetime | None = None,
        resulting_translation_version_id: UUID | None = None,
        event_id: UUID | None = None,
        created_at: datetime | None = None,
    ) -> HumanReview:
        prior = self.latest_for_item(review_pass_id, chunk_id)
        happened_at = created_at or datetime.now(timezone.utc)
        if prior is not None and prior.created_at is not None:
            prior_time = prior.created_at
            if prior_time.tzinfo is None:
                prior_time = prior_time.replace(tzinfo=timezone.utc)
            current_time = happened_at
            if current_time.tzinfo is None:
                current_time = current_time.replace(tzinfo=timezone.utc)
            if current_time <= prior_time:
                happened_at = prior_time + timedelta(microseconds=1)
        event = HumanReview(
            id=event_id or uuid4(),
            review_pass_id=review_pass_id,
            chunk_id=chunk_id,
            reviewer=reviewer,
            decision=status,
            status=status,
            approval_date=approval_date,
            comment=review_notes,
            review_notes=review_notes,
            proposed_text=correction,
            correction=correction,
            override_reason=override_reason,
            supersedes_review_id=prior.id if prior else None,
            previous_event_hash=prior.event_hash if prior else None,
            resulting_translation_version_id=resulting_translation_version_id,
            created_at=happened_at,
            event_hash="",
        )
        event.event_hash = self.hash_event(event)
        self.session.add(event)
        return event

    def events_for_pass(self, review_pass_id: UUID, *, chunk_id: UUID | None = None,
                        include_all_chunks: bool = True) -> list[HumanReview]:
        statement = select(HumanReview).where(HumanReview.review_pass_id == review_pass_id)
        if not include_all_chunks:
            statement = statement.where(HumanReview.chunk_id == chunk_id)
        return list(self.session.scalars(statement.order_by(
            HumanReview.created_at, HumanReview.id,
        )))

    def verify_pass(self, review_pass_id: UUID) -> AuditVerification:
        events = self.events_for_pass(review_pass_id)
        prior_by_chunk: dict[UUID | None, HumanReview] = {}
        errors: list[str] = []
        for event in events:
            prior = prior_by_chunk.get(event.chunk_id)
            expected_parent = prior.id if prior else None
            expected_hash = prior.event_hash if prior else None
            if event.supersedes_review_id != expected_parent:
                errors.append(f"{event.id}: supersedes link does not match event order")
            if event.previous_event_hash != expected_hash:
                errors.append(f"{event.id}: previous event hash does not match")
            if not event.event_hash or event.event_hash != self.hash_event(event):
                errors.append(f"{event.id}: event hash mismatch")
            prior_by_chunk[event.chunk_id] = event
        return AuditVerification(review_pass_id, len(events), not errors, tuple(errors))

    @classmethod
    def hash_event(cls, event: HumanReview) -> str:
        payload = {
            "id": str(event.id),
            "review_pass_id": str(event.review_pass_id),
            "reviewer": event.reviewer,
            "decision": event.decision,
            "status": event.status,
            "approval_date": cls._timestamp(event.approval_date),
            "chunk_id": str(event.chunk_id) if event.chunk_id else None,
            "comment": event.comment,
            "proposed_text": event.proposed_text,
            "review_notes": event.review_notes,
            "correction": event.correction,
            "override_reason": event.override_reason,
            "resulting_translation_version_id": (
                str(event.resulting_translation_version_id)
                if event.resulting_translation_version_id else None
            ),
            "supersedes_review_id": str(event.supersedes_review_id) if event.supersedes_review_id else None,
            "previous_event_hash": event.previous_event_hash,
            "created_at": cls._timestamp(event.created_at),
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _timestamp(value: datetime | None) -> str | None:
        if value is None:
            return None
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat(timespec="microseconds")
