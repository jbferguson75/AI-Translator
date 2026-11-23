# Plugin-based evaluation

## Built-in evaluators

- `fidelity` checks source/translation meaning, additions, omissions, and terminology.
- `readability` checks clarity, sentence complexity, vocabulary, and the translation's target reading level.
- `consistency` checks repeated terms, names, phrasing, register, and formatting within the supplied segments.

Each plugin implements `EvaluationPlugin`: a stable `plugin_id` and version, prompt template, `build_prompt(EvaluationInput)`, and `parse_output(response)`. Register external plugins with `EvaluatorRegistry.register()`. The registry rejects duplicate IDs and invalid interfaces. Plugins receive read-only source/translation segment values and return validated `EvaluationOutput` values.

## Configuration and commands

Copy [ai-translator-evaluation.example.toml](ai-translator-evaluation.example.toml) to `ai-translator-evaluation.toml`; adjust evaluator model tags to the locally installed Ollama models. The file selects one or more evaluator models, plugins, batch size, inference options, and Ollama endpoint.

```sh
ai-translator evaluators list
ai-translator evaluate TRANSLATION_VERSION_UUID --config ai-translator-evaluation.toml --limit 20 --verbose
ai-translator evaluation search "Atonement"
ai-translator evaluation search "adjective omitted" --plugin fidelity
```

`evaluate --verbose` (`-v`) reports the selected translation scope, plugins and models, batch progress, Ollama wait/completion times, errors, and saved evaluation version IDs to stderr. Without it, the command prints only its per-candidate summary.

The evaluator calls Ollama's `/api/generate` endpoint in JSON mode and validates score (0–100), confidence (0–1), reasoning, recommendations, and detected issues. It accepts `issues` as an alias for `detected_issues`. [Ollama structured output reference](https://github.com/ollama/ollama/blob/main/docs/api.md)

## Persistence and search

Each plugin/model pair creates a new `EvaluationVersion` under that plugin's `Evaluation` series and points to the exact immutable translation version. Each model output is preserved in `EvaluationResult`, including aggregate score/confidence/reasoning/recommendations/issues, every batch response, prompt hashes, model digest when available, and run metadata. Failure or interruption after a successful batch produces a `partial` version; prior versions are never overwritten.

`EvaluationResult.searchable_text` contains reasoning, detected issues, and recommendations. `EvaluationRepository.search_results()` performs case-insensitive substring searches; `ai-translator evaluation search` exposes the same query. Additive columns and indexes are introduced by Alembic revision `0002_eval_payload`; run `ai-translator db upgrade` before evaluating an existing database.

Scores are evaluator model judgments, not calibrated probabilities. Confidence is stored as supplied by the evaluator. The plugin version, prompt version/hash, exact translation version, model tag/digest, options, and segment hashes are retained so results can be audited and compared across reruns.
