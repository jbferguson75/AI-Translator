# Scripture plugin

This package adds scripture-specific evaluators without changing the platform package or its database schema. It calls the platform's existing `EvaluationService`, so each evaluator/model run is persisted as a normal, versioned evaluation linked to an immutable translation version.

## Install

From the repository root, with the platform virtual environment active:

```sh
pip install -e .
pip install -e plugins/scripture
```

The plugin exposes its own CLI; it does not replace or patch `ai-translator` commands.

## Commands

```sh
scripture evaluators
scripture evaluate TRANSLATION_VERSION_UUID --config ai-translator-evaluation.toml --verbose
scripture evaluate TRANSLATION_VERSION_UUID --config ai-translator-evaluation.toml \
  --report reports/scripture-review.md
scripture review-report TRANSLATION_VERSION_UUID --output reports/scripture-review.md
```

The default evaluation runs these evaluator IDs:

- `scripture.doctrinal_consistency`
- `scripture.terminology_drift`
- `scripture.author_voice`
- `scripture.glossary_compliance`

The existing evaluator configuration supplies Ollama model aliases. `--models` can select one or more configured aliases. The plugin checks the document's existing `domain` value is `scripture`.

## Author and doctrine metadata

Bundled author profiles map `1 Nephi` and `2 Nephi` to Nephi; `Jacob`, `Enos`, `Jarom`, and `Omni` to their namesakes; `Words of Mormon` and `Mormon` to Mormon; and `Moroni` to Moroni. These are editable review cues, not rigid style rules. Supply a replacement file with `--author-metadata` using the same shape as `src/scripture_plugin/data/authors.json` to correct or extend book attribution.

The starter doctrine glossary at `src/scripture_plugin/data/doctrine_glossary.json` gives reviewers context without asserting authoritative definitions. Entries may include `target_locale` and `author` scopes. Use `--doctrine-glossary` to load a project-specific JSON glossary. Approved platform glossary entries for the document's project and locale pair are automatically added to terminology drift and compliance prompts.

The deterministic report drift scan flags a repeated source term when the configured preferred target form is absent in one or more occurrences. Morphology, paraphrase, and context can cause false positives; use it as a review lead, not as an automatic correction. The model-based terminology-drift evaluator separately considers contextual variation.

## Review reports

Reports include the translation version ID and content hash, known author mappings, stored scores/reasoning/issues/recommendations from all Scripture evaluator versions, and deterministic preferred-term drift findings. They are Markdown artifacts written locally and can be regenerated from the stored database results. Automated doctrine assessments are not authoritative; human review remains required.
