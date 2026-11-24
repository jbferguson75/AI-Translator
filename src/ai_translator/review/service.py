from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import UUID, uuid4

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..db.models import (
    Chunk, HumanReview, ReviewPass, TranslationSegment, TranslationVersion,
)
from .audit import AuditTrailService

REVIEW_STATUSES = frozenset({"pending", "approved", "rejected", "needs_revision"})


@dataclass(frozen=True, slots=True)
class ReviewItemView:
    review_event_id: UUID
    review_pass_id: UUID
    chunk_id: UUID
    stable_key: str
    source_text: str
    candidate_text: str
    status: str
    reviewer: str
    review_notes: str | None
    correction: str | None
    override_reason: str | None
    resulting_translation_version_id: UUID | None


class HumanReviewService:
    """Create append-only human decisions and separate approved versions."""

    def __init__(self, session: Session):
        self.session = session
        self.audit = AuditTrailService(session)

    def create_pass(self, *, translation_version_id: UUID, name: str,
                    assigned_to: str | None = None, created_by: str = "local-user") -> tuple[ReviewPass, int]:
        if not name.strip():
            raise ValueError("Review pass name is required")
        if assigned_to is not None and not assigned_to.strip():
            raise ValueError("assigned_to cannot be blank")
        if not created_by.strip():
            raise ValueError("created_by cannot be blank")
        version = self.session.get(TranslationVersion, translation_version_id)
        if version is None:
            raise ValueError(f"Translation version {translation_version_id} does not exist")
        segments = self._segments(version.id)
        if not segments:
            raise ValueError("Translation version has no segments to review")
        review_pass = ReviewPass(
            translation_version_id=version.id,
            name=name.strip(),
            status="open",
            assigned_to=assigned_to.strip() if assigned_to else None,
        )
        self.session.add(review_pass)
        self.session.flush()
        reviewer = assigned_to.strip() if assigned_to else created_by.strip()
        for _segment, chunk in segments:
            self.audit.append_event(
                review_pass_id=review_pass.id,
                chunk_id=chunk.id,
                reviewer=reviewer,
                status="pending",
                review_notes="Awaiting human review",
            )
        self.session.flush()
        return review_pass, len(segments)

    def list_passes(self, *, translation_version_id: UUID | None = None,
                    status: str | None = None) -> list[ReviewPass]:
        statement = select(ReviewPass)
        if translation_version_id is not None:
            statement = statement.where(ReviewPass.translation_version_id == translation_version_id)
        if status is not None:
            statement = statement.where(ReviewPass.status == status)
        return list(self.session.scalars(statement.order_by(ReviewPass.created_at.desc(), ReviewPass.id)))

    def items(self, review_pass_id: UUID, *, status: str | None = None) -> list[ReviewItemView]:
        review_pass = self._require_pass(review_pass_id)
        base_version = self.session.get(TranslationVersion, review_pass.translation_version_id)
        segments = self._segments(base_version.id)
        latest_version = self._latest_approved_version(review_pass.id) or base_version
        current_text_by_chunk = {
            segment.chunk_id: segment.target_text
            for segment, _chunk in self._segments(latest_version.id)
        }
        latest_events = self._latest_events_by_chunk(review_pass.id)
        result = []
        for _segment, chunk in segments:
            event = latest_events.get(chunk.id)
            item_status = event.status if event else "pending"
            if status is not None and item_status != status:
                continue
            result.append(ReviewItemView(
                review_event_id=event.id if event else UUID(int=0),
                review_pass_id=review_pass.id,
                chunk_id=chunk.id,
                stable_key=chunk.stable_key,
                source_text=chunk.source_text,
                candidate_text=current_text_by_chunk.get(chunk.id, ""),
                status=item_status,
                reviewer=event.reviewer if event else (review_pass.assigned_to or "unassigned"),
                review_notes=event.review_notes if event else None,
                correction=event.correction if event else None,
                override_reason=event.override_reason if event else None,
                resulting_translation_version_id=(event.resulting_translation_version_id if event else None),
            ))
        return result

    def decide(self, *, review_event_id: UUID, status: str, reviewer: str,
               review_notes: str | None = None, correction: str | None = None,
               override_reason: str | None = None,
               approval_date: datetime | None = None) -> HumanReview:
        if status not in REVIEW_STATUSES:
            raise ValueError("status must be pending, approved, rejected, or needs_revision")
        if not reviewer.strip():
            raise ValueError("reviewer is required")
        if correction is not None and not correction.strip():
            raise ValueError("correction cannot be blank; omit it to keep the candidate text")
        if status != "approved" and approval_date is not None:
            raise ValueError("approval_date is only valid when status is approved")
        prior_event = self.session.get(HumanReview, review_event_id)
        if prior_event is None:
            raise ValueError(f"Review event {review_event_id} does not exist")
        current = self.audit.latest_for_item(prior_event.review_pass_id, prior_event.chunk_id)
        if current is None or current.id != prior_event.id:
            raise ValueError("This review event is stale; list the review pass and decide its current event")
        review_pass = self._require_pass(prior_event.review_pass_id)
        resulting_version_id: UUID | None = None
        event_id = uuid4()
        approved_at = None
        if status == "approved":
            approved_at = approval_date or datetime.now(timezone.utc)
            base_version = self._latest_approved_version(review_pass.id)
            if base_version is None:
                base_version = self.session.get(TranslationVersion, review_pass.translation_version_id)
            current_segment = self.session.scalar(select(TranslationSegment).where(
                TranslationSegment.translation_version_id == base_version.id,
                TranslationSegment.chunk_id == prior_event.chunk_id,
            ))
            if current_segment is None:
                raise ValueError("The review item is not present in the current translation version")
            selected_text = correction if correction is not None else current_segment.target_text
            if selected_text != current_segment.target_text and not (override_reason and override_reason.strip()):
                raise ValueError("Provide --override-reason when an approval changes the candidate text")
            resulting_version = self._create_human_version(
                base_version=base_version,
                review_pass=review_pass,
                chunk_id=prior_event.chunk_id,
                target_text=selected_text,
                reviewer=reviewer.strip(),
                approval_date=approved_at,
                review_event_id=event_id,
                review_notes=review_notes,
                override_reason=override_reason,
            )
            resulting_version_id = resulting_version.id

        event = self.audit.append_event(
            review_pass_id=review_pass.id,
            chunk_id=prior_event.chunk_id,
            reviewer=reviewer.strip(),
            status=status,
            review_notes=review_notes,
            correction=correction if correction else None,
            override_reason=override_reason if override_reason else None,
            approval_date=approved_at,
            resulting_translation_version_id=resulting_version_id,
            event_id=event_id,
        )
        self.session.flush()
        self._refresh_pass_status(review_pass)
        return event

    def history(self, review_event_id: UUID) -> list[HumanReview]:
        event = self.session.get(HumanReview, review_event_id)
        if event is None:
            raise ValueError(f"Review event {review_event_id} does not exist")
        return list(self.session.scalars(select(HumanReview).where(
            HumanReview.review_pass_id == event.review_pass_id,
            HumanReview.chunk_id == event.chunk_id,
        ).order_by(HumanReview.created_at, HumanReview.id)))

    def verify_audit(self, review_pass_id: UUID):
        self._require_pass(review_pass_id)
        return self.audit.verify_pass(review_pass_id)

    def _create_human_version(self, *, base_version: TranslationVersion, review_pass: ReviewPass,
                              chunk_id: UUID, target_text: str, reviewer: str,
                              approval_date: datetime, review_event_id: UUID,
                              review_notes: str | None, override_reason: str | None) -> TranslationVersion:
        segments = self._segments(base_version.id)
        if not any(segment.chunk_id == chunk_id for segment, _chunk in segments):
            raise ValueError("The approved translation version does not contain the reviewed segment")
        payload = []
        transformed = []
        for segment, chunk in segments:
            text = target_text if chunk.id == chunk_id else segment.target_text
            payload.append({"stable_key": chunk.stable_key, "target_text": text})
            transformed.append((chunk, text))
        content_hash = hashlib.sha256(json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")).hexdigest()
        latest = self.session.scalar(select(func.max(TranslationVersion.version_number)).where(
            TranslationVersion.translation_id == base_version.translation_id
        )) or 0
        original_text = next(segment.target_text for segment, chunk in segments if chunk.id == chunk_id)
        changed = target_text != original_text
        version = TranslationVersion(
            translation_id=base_version.translation_id,
            version_number=latest + 1,
            parent_version_id=base_version.id,
            model_run_id=None,
            author_kind="human",
            artifact_kind="HUMAN_EDIT" if changed else "FINAL_APPROVED",
            status="reviewed" if changed else "approved",
            content_hash=content_hash,
            run_manifest={
                "kind": "human_approved_translation",
                "review_pass_id": str(review_pass.id),
                "human_review_event_id": str(review_event_id),
                "source_translation_version_id": str(review_pass.translation_version_id),
                "parent_translation_version_id": str(base_version.id),
                "reviewer": reviewer,
                "approval_date": approval_date.isoformat(),
                "review_notes": review_notes,
                "override_reason": override_reason,
                "approved_chunk_id": str(chunk_id),
            },
        )
        self.session.add(version)
        self.session.flush()
        from ..db.models import TranslationVersionParent
        self.session.add(TranslationVersionParent(
            child_version_id=version.id, parent_version_id=base_version.id,
            relationship_kind="parent", ordinal=0,
        ))
        for chunk, text in transformed:
            self.session.add(TranslationSegment(
                translation_version_id=version.id,
                chunk_id=chunk.id,
                target_text=text,
                target_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            ))
        self.session.flush()
        if not changed:
            return version
        # Keep the human correction and its approval as distinct immutable artifacts.
        final_version = TranslationVersion(
            translation_id=base_version.translation_id,
            version_number=latest + 2,
            parent_version_id=version.id,
            model_run_id=None,
            author_kind="human",
            artifact_kind="FINAL_APPROVED",
            status="approved",
            content_hash=content_hash,
            run_manifest={"kind": "final_approved_translation",
                          "human_edit_translation_version_id": str(version.id),
                          "review_pass_id": str(review_pass.id),
                          "human_review_event_id": str(review_event_id),
                          "reviewer": reviewer, "approval_date": approval_date.isoformat()},
        )
        self.session.add(final_version)
        self.session.flush()
        self.session.add(TranslationVersionParent(
            child_version_id=final_version.id, parent_version_id=version.id,
            relationship_kind="parent", ordinal=0,
        ))
        for chunk, text in transformed:
            self.session.add(TranslationSegment(
                translation_version_id=final_version.id, chunk_id=chunk.id, target_text=text,
                target_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            ))
        self.session.flush()
        return final_version

    def _refresh_pass_status(self, review_pass: ReviewPass) -> None:
        expected_chunks = {chunk.id for _segment, chunk in self._segments(review_pass.translation_version_id)}
        latest = self._latest_events_by_chunk(review_pass.id)
        pending = [chunk_id for chunk_id in expected_chunks
                   if chunk_id not in latest or latest[chunk_id].status == "pending"]
        if pending:
            review_pass.status = "open"
            review_pass.completed_at = None
        else:
            review_pass.status = "completed"
            review_pass.completed_at = datetime.now(timezone.utc)
        self.session.flush()

    def _latest_approved_version(self, review_pass_id: UUID) -> TranslationVersion | None:
        result_id = self.session.scalar(select(HumanReview.resulting_translation_version_id).where(
            HumanReview.review_pass_id == review_pass_id,
            HumanReview.resulting_translation_version_id.is_not(None),
        ).order_by(HumanReview.created_at.desc(), HumanReview.id.desc()).limit(1))
        return self.session.get(TranslationVersion, result_id) if result_id else None

    def _latest_events_by_chunk(self, review_pass_id: UUID) -> dict[UUID, HumanReview]:
        events = self.session.scalars(select(HumanReview).where(
            HumanReview.review_pass_id == review_pass_id,
            HumanReview.chunk_id.is_not(None),
        ).order_by(HumanReview.created_at, HumanReview.id))
        result: dict[UUID, HumanReview] = {}
        for event in events:
            result[event.chunk_id] = event
        return result

    def _segments(self, translation_version_id: UUID):
        return list(self.session.execute(
            select(TranslationSegment, Chunk)
            .join(Chunk, Chunk.id == TranslationSegment.chunk_id)
            .where(TranslationSegment.translation_version_id == translation_version_id)
            .order_by(Chunk.ordinal)
        ).all())

    def _require_pass(self, review_pass_id: UUID) -> ReviewPass:
        review_pass = self.session.get(ReviewPass, review_pass_id)
        if review_pass is None:
            raise ValueError(f"Review pass {review_pass_id} does not exist")
        return review_pass
