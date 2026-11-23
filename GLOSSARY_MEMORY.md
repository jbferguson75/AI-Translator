# Glossaries and translation memory

Apply schema changes before using these commands:

```sh
ai-translator db upgrade
```

## Glossaries

Projects can have multiple glossaries for a source/target locale pair. Add entries with a priority; higher numbers take precedence during prompt injection. Equal-priority entries that prescribe different translations are treated as unresolved conflicts and stop translation before any model call. Resolve them by changing the preferred translation or raising one entry's priority.

```sh
ai-translator glossary create "Scripture terms" --project "Book of Mormon - Simplified" --source-locale en --target-locale es
ai-translator glossary list --project "Book of Mormon - Simplified"
ai-translator glossary add-entry GLOSSARY_UUID "atonement" "expiación" --priority 10 --notes "Approved scripture term"
ai-translator glossary entries GLOSSARY_UUID
ai-translator glossary conflicts --project "Book of Mormon - Simplified" --source-locale en --target-locale es
ai-translator glossary update-entry ENTRY_UUID --priority 20
```

Only terms found in a source chunk are injected. The exact selected entry, priority, glossary, and notes are recorded in the translation run's prompt context. Usage counts and document/chunk references are updated only for chunks successfully translated.

## Translation memory

Entries are project- and locale-scoped. New entries are unapproved by default; only approved entries can be injected. Lookup ranks exact source matches first, then token Jaccard similarity. If same-source entries have equally ranked but different approved translations, translation stops until the conflict is resolved. Approve the intended entry and unapprove or remove competing entries. If multiple conflicting entries remain approved, differing historical usage counts provide a deterministic ranking; tied counts still require an explicit choice.

```sh
ai-translator memory add "in the beginning" "al principio" \
  --project "Book of Mormon - Simplified" --source-locale en --target-locale es \
  --source-reference "Genesis 1:1" --approved
ai-translator memory lookup "in the beginning" --project "Book of Mormon - Simplified" \
  --source-locale en --target-locale es
ai-translator memory conflicts --project "Book of Mormon - Simplified"
ai-translator memory list --project "Book of Mormon - Simplified"
```

Up to three best positive-overlap approved matches are injected per chunk. Successful model output increments each injected entry's usage count and records a unique `DOCUMENT_UUID:stable_chunk_key` reference. `memory approve`, `memory unapprove`, `memory update`, and `memory remove` manage entries; `glossary update-entry` or `glossary remove-entry` resolves glossary conflicts.

The database columns for usage/priority are added by Alembic revision `0004_glossary_memory_usage`.

## Draft, suggest, approve, revise

Create a draft for the desired scripture scope, then analyze that immutable version for repeated terms. Suggestions are filtered to source phrases found in at least two translated verses and retain the verse references, model response, exact prompt, and run metadata.

```sh
ai-translator glossary revise DRAFT_TRANSLATION_VERSION_UUID \
  --config ai-translator.toml --verbose
ai-translator glossary suggest TRANSLATION_VERSION_UUID --config ai-translator.toml
ai-translator glossary suggestions --translation-version TRANSLATION_VERSION_UUID
ai-translator glossary approve-suggestion SUGGESTION_UUID --glossary GLOSSARY_UUID --priority 10
ai-translator translate DOCUMENT_UUID --book "1 Nephi" --chapter 1 \
  --target-locale en --reading-level "4th grade" --task-kind simplification
```

Reject proposals with `glossary reject-suggestion SUGGESTION_UUID`. Approval creates an entry in a project glossary only when its project and locale pair match the draft. `glossary revise` creates a new translation version linked to the draft, preserves its original book/chapter scope, and injects currently approved glossary entries automatically; the draft and its evaluation history remain unchanged. Translation scope, prompt snapshots, and suggestions are retained for audit. The `--book` and `--chapter` options can also be used on `translate` without glossary suggestions. Alembic revision `0005_glossary_suggestion_workflow` stores suggestion runs and reviewed proposals.
