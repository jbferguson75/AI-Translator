# Human review and approval

Human review is event-sourced. A review pass identifies the immutable translation snapshot being reviewed; each chunk gets an initial `pending` event. Later decisions append a new event that supersedes the prior event for that item. Existing review events cannot be updated or deleted.

## Workflow

Upgrade the database, then create a review pass:

```sh
ai-translator db upgrade
ai-translator human-review create TRANSLATION_VERSION_UUID \
  --name "First human review" --assigned-to "reviewer@example.org"
ai-translator human-review passes --translation-version TRANSLATION_VERSION_UUID
ai-translator human-review items REVIEW_PASS_UUID --status pending
```

The `items` command prints the current event UUID, source text, and candidate translation. Record that event UUID when deciding:

```sh
ai-translator human-review decide REVIEW_EVENT_UUID \
  --status approved --reviewer "reviewer@example.org" --notes "Checked against source"
```

An approval without a correction still creates a new complete translation snapshot with `author_kind="human"`. It has a parent link to the preceding AI or human version and leaves all prior versions unchanged. With a correction, pass both the revised text and why it overrides the candidate:

```sh
ai-translator human-review decide REVIEW_EVENT_UUID \
  --status approved --reviewer "reviewer@example.org" \
  --correction "Corrected target text" --override-reason "Restores the omitted subject" \
  --notes "Compared with the source verse"
```

Other decisions use `--status rejected` or `--status needs_revision`. A `needs_revision` correction is stored as a proposed correction but does not create an approved translation version. To set a backdated approval, pass `--approval-date` with an ISO-8601 datetime; otherwise the approval timestamp is set to the current UTC time.

Each item decision appends a versioned `HumanReview` event. A stale event UUID is rejected; list items again and act on the latest event. The review pass becomes completed when all items have a non-pending status. A later appended decision can reopen the pass.

## Audit trail

Every review event stores reviewer, status, approval date, review notes, correction, override reason, resulting human translation version, superseded event, previous event hash, and its own SHA-256 hash. SQLite triggers reject edits/deletes and invalid inserted events. Verify a pass with:

```sh
ai-translator human-review history REVIEW_EVENT_UUID
ai-translator human-review audit REVIEW_PASS_UUID
```

Audit chains are independent per review item (pass and chunk). The migration backfills existing review rows into the event schema and preserves their old decision, comment, proposed text, and resulting-version references.

Alembic revision `0006_human_review_workflow` adds the review status and audit fields. Run `ai-translator db upgrade` before using these commands on an existing database.
