# Translation engine

## Configure Ollama models

Copy the example and edit it for the model tags installed in your Ollama instance:

```sh
cp ai-translator.example.toml ai-translator.toml
ollama pull qwen3:8b
ollama pull mistral:latest
ollama pull gemma3:4b
ai-translator models --config ai-translator.toml
```

The example declares aliases `qwen`, `mistral`, and `gemma`, with a model tag under each alias. It is configured for English simplification at a 4th-grade level. Change `translation.candidate_models` to select which configured aliases produce candidates. Ollama model tags depend on the local install; adjust them to match `ollama list`.

## Generate candidates

Use a source document UUID from the import receipt. One immutable translation version is written for each selected model:

```sh
ai-translator translate DOCUMENT_UUID \
  --config ai-translator.toml \
  --target-locale en \
  --reading-level "4th grade" \
  --task-kind simplification \
  --verbose
```

`--verbose` (`-v`) writes progress to stderr: selected scope and models, glossary/memory counts, each chunk's Ollama request and completion time, errors, and saved version numbers. Normal output remains a concise per-model summary.

For a small first run, `--limit 5` translates only the first five chunks and records each result as `partial`. Each model candidate is persisted in its own short transaction. A failed model is recorded with status `failed` or `partial`, while successful candidates remain stored. Running again creates new version numbers; earlier translations are never updated.

Scripture can be scoped by book and chapter; `--limit` then applies inside that scope. A complete chapter run is marked completed for its selected scope, while the run manifest also records whether it covers the full document:

```sh
ai-translator translate DOCUMENT_UUID --book "1 Nephi" --chapter 1 --target-locale en
```

The default prompt is selected by task and domain. Simplification asks the model to translate into the target locale when needed and simplify to the requested level while preserving source meaning. The selected reading level is stored on the translation task, so different levels receive separate candidate sets. `translation.prompt_file` can select a custom UTF-8 template. Templates use `$source_locale`, `$target_locale`, `$task_kind`, `$domain`, `$reading_level`, `$glossary_terms`, `$translation_memory`, `$locator`, and `$source_text`. Every template body is saved as an immutable `PromptVersion`; each model run records its template version/hash, effective options, model digest when Ollama reports one, per-chunk input and prompt hashes, injected glossary/memory snapshots, and response metadata.

Approved translation memories are ranked deterministically by exact source hash and then word-set Jaccard similarity. The top three positive matches are injected per chunk. Glossary entries for the project's source and target locales are injected when their source term occurs in the chunk.

The client uses Ollama's `POST /api/generate` endpoint with `stream: false` so each verse response can be stored as one aligned segment. [Ollama API reference](https://github.com/ollama/ollama/blob/main/docs/api.md)
