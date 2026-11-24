# Feedback-guided revision and synthesis

The initial model translations and their evaluation results remain immutable. The `refine` command creates one new candidate per selected translation model, using the saved feedback for that candidate. The `synthesize` command creates candidate translations from multiple versions of the same translation task and uses the source text as the authority when candidate wording disagrees. Both commands immediately run the configured core evaluators on every successfully generated version.

## Revise one evaluated candidate

Run evaluation first. Then pass that candidate version ID to `refine`:

```sh
ai-translator evaluate TRANSLATION_VERSION_UUID \
  --config ai-translator-evaluation.toml \
  --models qwen,mistral,gemma

ai-translator refine TRANSLATION_VERSION_UUID \
  --config ai-translator.toml \
  --evaluation-config ai-translator-evaluation.toml \
  --models qwen,mistral,gemma \
  --evaluator-models qwen,mistral,gemma \
  --verbose
```

The revision prompt includes source text, the selected candidate, the latest stored results by evaluation category and evaluator model, applicable glossary entries, approved translation-memory matches, and the translation task's reading level. The service rejects missing feedback for any input segment rather than silently revising without evaluation.

## Synthesize several candidates

Evaluate each source candidate, then supply their version IDs together:

```sh
ai-translator synthesize VERSION_ID_1 VERSION_ID_2 VERSION_ID_3 \
  --config ai-translator.toml \
  --evaluation-config ai-translator-evaluation.toml \
  --models qwen,mistral,gemma \
  --evaluator-models qwen,mistral,gemma \
  --verbose
```

Candidates must belong to the same translation task and contain exactly the same source chunks. One new synthesis version is produced per selected synthesis model. Every synthesis result is evaluated immediately.

## Versioning and audit trail

Each new version stores its operation type, input version IDs, evaluator-version IDs, prompt version and hash, model metadata, per-chunk source and input-target hashes, evaluation-result IDs, and glossary/memory snapshots. It is linked to a primary parent version; synthesis also records all input parents in its manifest. Previous translations are not changed. Review or approve the new outputs with the regular human-review workflow.

Evaluation runs use the core plugins and evaluator aliases from the evaluation configuration unless overridden. For scripture documents, run `scripture evaluate NEW_VERSION_ID --config ai-translator-evaluation.toml --models qwen,mistral,gemma` on each generated version to add the Scripture-specific evaluator results. The refinement and synthesis commands consume those saved Scripture results when they are present on their input versions.
