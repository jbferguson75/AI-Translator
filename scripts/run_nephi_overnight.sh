#!/usr/bin/env bash
# Translate 1 Nephi and 2 Nephi, scan draft terminology, evaluate, then run all
# hierarchical Scripture review levels. State files make reruns resumable.

set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
AI_TRANSLATOR_BIN="${AI_TRANSLATOR_BIN:-$ROOT/.venv/bin/ai-translator}"
SCRIPTURE_BIN="${SCRIPTURE_BIN:-$ROOT/.venv/bin/scripture}"

TRANSLATION_CONFIG="${TRANSLATION_CONFIG:-$ROOT/ai-translator.toml}"
EVALUATION_CONFIG="${EVALUATION_CONFIG:-$ROOT/ai-translator-evaluation.toml}"
DATABASE="${DATABASE:-sqlite:///ai_translator.db}"
PROJECT="${PROJECT:-Book of Mormon - Simplified}"
DOCUMENT_UUID="${DOCUMENT_UUID:-}"

# Defaults follow the requested English, fourth-grade simplification workflow.
TARGET_LOCALE="${TARGET_LOCALE:-en}"
TASK_KIND="${TASK_KIND:-simplification}"
READING_LEVEL="${READING_LEVEL:-4th grade}"

# Run every configured candidate model by default. Override these variables to
# narrow a run; each model's translations and evaluations are independently versioned.
TRANSLATION_MODELS="${TRANSLATION_MODELS:-llama,mistral,gemma}"
EVALUATOR_MODELS="${EVALUATOR_MODELS:-llama,mistral,gemma}"
GLOSSARY_MODEL="${GLOSSARY_MODEL:-${TRANSLATION_MODELS%%,*}}"
REVIEW_LEVELS="${REVIEW_LEVELS:-1,2,3,4}"
RUN_GLOSSARY_SUGGESTIONS="${RUN_GLOSSARY_SUGGESTIONS:-0}"
GLOSSARY_SUGGESTION_LIMIT="${GLOSSARY_SUGGESTION_LIMIT:-50}"

AUTHOR_METADATA="${AUTHOR_METADATA:-$ROOT/plugins/scripture/src/scripture_plugin/data/authors.json}"
DOCTRINE_GLOSSARY="${DOCTRINE_GLOSSARY:-$ROOT/plugins/scripture/src/scripture_plugin/data/doctrine_glossary.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/reports/nephi-overnight-$(date -u +%Y%m%dT%H%M%SZ)}"
STATE_DIR="$OUTPUT_DIR/state"
RUN_DB_UPGRADE="${RUN_DB_UPGRADE:-1}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment not found: $PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -x "$AI_TRANSLATOR_BIN" ]]; then
  echo "ai-translator CLI not found: $AI_TRANSLATOR_BIN (install the project in .venv first)" >&2
  exit 1
fi
if [[ ! -x "$SCRIPTURE_BIN" ]]; then
  echo "scripture CLI not found: $SCRIPTURE_BIN (install plugins/scripture in .venv first)" >&2
  exit 1
fi
for file in "$TRANSLATION_CONFIG" "$EVALUATION_CONFIG" "$AUTHOR_METADATA" "$DOCTRINE_GLOSSARY"; do
  if [[ ! -f "$file" ]]; then
    echo "Required configuration file not found: $file" >&2
    exit 1
  fi
done

mkdir -p "$STATE_DIR"
LOG_FILE="$OUTPUT_DIR/overnight.log"
exec > >(tee -a "$LOG_FILE") 2>&1

# Trace every shell command into the same live console and log, including
# commands whose CLI does not expose its own --verbose option.
set -x
trap 'status=$?; echo "ERROR: command failed with exit $status at line ${LINENO}; see the xtrace immediately above." >&2' ERR

ai() { "$AI_TRANSLATOR_BIN" "$@"; }
scr() { "$SCRIPTURE_BIN" "$@"; }

echo "Starting Nephi books run at $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "Project: $PROJECT"
echo "Database: $DATABASE"
echo "Translation: $TARGET_LOCALE ($TASK_KIND, $READING_LEVEL)"
echo "Translation model(s): $TRANSLATION_MODELS"
echo "Evaluator model(s): $EVALUATOR_MODELS"
echo "Hierarchical review levels: $REVIEW_LEVELS"
echo "Model-generated glossary suggestions: $RUN_GLOSSARY_SUGGESTIONS"
echo "Output and resume state: $OUTPUT_DIR"

if [[ "$RUN_DB_UPGRADE" == "1" ]]; then
  ai db upgrade --database "$DATABASE"
fi

if [[ -z "$DOCUMENT_UUID" ]]; then
  DOCUMENT_UUID="$(set +E; PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" - "$PROJECT" "$DATABASE" <<'PY'
import sys
from sqlalchemy import select
from ai_translator.db.models import Document, Project
from ai_translator.db.session import make_engine, make_session_factory

project_name, database = sys.argv[1], sys.argv[2]
engine = make_engine(database)
session = make_session_factory(engine)()
try:
    project = session.scalar(select(Project).where(Project.name == project_name))
    if project is None:
        raise SystemExit(f"No project named {project_name!r} exists")
    documents = list(session.scalars(select(Document).where(
        Document.project_id == project.id,
        Document.domain == "scripture",
    ).order_by(Document.created_at)))
    if len(documents) != 1:
        project_documents = list(session.scalars(select(Document).where(
            Document.project_id == project.id,
        ).order_by(Document.created_at)))
        other_scripture_documents = list(session.scalars(
            select(Document).where(Document.domain == "scripture").order_by(Document.created_at)
        ))
        choices = "\n".join(
            f"  {item.id}\t{item.title}\tdomain={item.domain}\tproject={item.project.name}"
            for item in (project_documents if project_documents else other_scripture_documents)
        ) or "  (none in this database)"
        if not documents and project_documents:
            detail = f"Project {project_name!r} has no document with domain='scripture'."
            remedy = "Re-import the structured source with 'ai-translator import scripture'."
        elif not documents and other_scripture_documents:
            detail = f"Project {project_name!r} has no scripture document, but other projects do."
            remedy = "Set PROJECT to the project containing the intended scripture document."
        else:
            detail = f"Expected exactly one scripture document in project {project_name!r}; found {len(documents)}."
            remedy = "Set DOCUMENT_UUID to the intended scripture document UUID."
        raise SystemExit(
            f"{detail} {remedy}\nCandidates:\n{choices}"
        )
    print(documents[0].id)
finally:
    session.close()
    engine.dispose()
PY
)"
fi

if [[ ! "$DOCUMENT_UUID" =~ ^[[:xdigit:]-]{36}$ ]]; then
  echo "DOCUMENT_UUID is not a UUID: $DOCUMENT_UUID" >&2
  exit 1
fi
echo "Scripture document: $DOCUMENT_UUID"

settings_file="$STATE_DIR/run-settings"
settings_tmp="$settings_file.tmp"
{
  printf 'project=%s\n' "$PROJECT"
  printf 'document_uuid=%s\n' "$DOCUMENT_UUID"
  printf 'database=%s\n' "$DATABASE"
  printf 'target_locale=%s\n' "$TARGET_LOCALE"
  printf 'task_kind=%s\n' "$TASK_KIND"
  printf 'reading_level=%s\n' "$READING_LEVEL"
  printf 'translation_models=%s\n' "$TRANSLATION_MODELS"
  printf 'evaluator_models=%s\n' "$EVALUATOR_MODELS"
  printf 'review_levels=%s\n' "$REVIEW_LEVELS"
  printf 'run_glossary_suggestions=%s\n' "$RUN_GLOSSARY_SUGGESTIONS"
  printf 'glossary_suggestion_limit=%s\n' "$GLOSSARY_SUGGESTION_LIMIT"
  shasum -a 256 "$TRANSLATION_CONFIG" "$EVALUATION_CONFIG" "$AUTHOR_METADATA" "$DOCTRINE_GLOSSARY"
} > "$settings_tmp"
if [[ -f "$settings_file" ]] && ! cmp -s "$settings_tmp" "$settings_file"; then
  rm -f "$settings_tmp"
  echo "OUTPUT_DIR already contains state for different settings; choose a new OUTPUT_DIR or restore the original settings." >&2
  exit 1
fi
mv "$settings_tmp" "$settings_file"

run_translate() {
  local book="$1"
  local slug="$2"
  local ids_file="$STATE_DIR/$slug.translation-versions"

  if [[ -s "$ids_file" ]]; then
    echo "Reusing saved translation version IDs for $book from $ids_file"
    return
  fi

  local output_file="$STATE_DIR/$slug.translate-output"
  local command_status
  echo "Translating $book..."
  set +e
  ai translate "$DOCUMENT_UUID" \
    --book "$book" \
    --target-locale "$TARGET_LOCALE" \
    --task-kind "$TASK_KIND" \
    --reading-level "$READING_LEVEL" \
    --models "$TRANSLATION_MODELS" \
    --config "$TRANSLATION_CONFIG" \
    --verbose \
    --database "$DATABASE" 2>&1 | tee "$output_file"
  command_status="${PIPESTATUS[0]}"
  set -e
  if [[ "$command_status" != "0" ]]; then
    echo "Translation command failed for $book (exit $command_status)" >&2
    return "$command_status"
  fi

  local ids_tmp="$ids_file.tmp"
  sed -nE '/: completed, version /s/.*\[id ([[:xdigit:]-]{36})\].*/\1/p' "$output_file" > "$ids_tmp"
  if [[ ! -s "$ids_tmp" ]]; then
    echo "No completed translation versions were produced for $book; see $output_file" >&2
    rm -f "$ids_tmp"
    return 1
  fi
  mv "$ids_tmp" "$ids_file"
  echo "Saved completed translation version IDs for $book:"
  sed 's/^/  /' "$ids_file"
}

run_version_workflow() {
  local book="$1"
  local slug="$2"
  local version_id="$3"
  local marker_prefix="$STATE_DIR/$version_id"

  if [[ ! -f "$marker_prefix.glossary-scan.done" ]]; then
    echo "Scanning approved glossary usage and terminology drift for $book version $version_id..."
    scr review-report "$version_id" \
      --output "$OUTPUT_DIR/$slug-$version_id-glossary-scan.md" \
      --author-metadata "$AUTHOR_METADATA" \
      --doctrine-glossary "$DOCTRINE_GLOSSARY" \
      --database "$DATABASE"
    if [[ "$RUN_GLOSSARY_SUGGESTIONS" == "1" ]]; then
      local scan_output="$STATE_DIR/$version_id.glossary-suggestions-output"
      echo "Generating optional term suggestions from the first $GLOSSARY_SUGGESTION_LIMIT segments..."
      set +e
      ai glossary suggest "$version_id" \
        --config "$TRANSLATION_CONFIG" \
        --model "$GLOSSARY_MODEL" \
        --limit "$GLOSSARY_SUGGESTION_LIMIT" \
        --database "$DATABASE" 2>&1 | tee "$scan_output"
      local command_status="${PIPESTATUS[0]}"
      set -e
      if [[ "$command_status" != "0" ]] || grep -q 'Glossary suggestion run .*: failed' "$scan_output"; then
        echo "Optional glossary suggestion scan failed for $version_id; evaluate/review was not started." >&2
        return 1
      fi
      ai glossary suggestions --translation-version "$version_id" --status proposed --database "$DATABASE"
      echo "Suggestions are proposed for human approval; none were auto-approved."
    fi
    echo "completed" > "$marker_prefix.glossary-scan.done"
  else
    echo "Glossary scan already complete for $version_id; skipping."
  fi

  if [[ ! -f "$marker_prefix.evaluate.done" ]]; then
    echo "Running Scripture evaluator models for $book version $version_id..."
    scr evaluate "$version_id" \
      --config "$EVALUATION_CONFIG" \
      --models "$EVALUATOR_MODELS" \
      --author-metadata "$AUTHOR_METADATA" \
      --doctrine-glossary "$DOCTRINE_GLOSSARY" \
      --report "$OUTPUT_DIR/$slug-$version_id-scripture-evaluation.md" \
      --verbose \
      --database "$DATABASE"
    echo "completed" > "$marker_prefix.evaluate.done"
  else
    echo "Scripture evaluation already complete for $version_id; skipping."
  fi

  if [[ ! -f "$marker_prefix.review.done" ]]; then
    echo "Running hierarchical Scripture review for $book version $version_id..."
    scr review all "$version_id" \
      --levels "$REVIEW_LEVELS" \
      --config "$EVALUATION_CONFIG" \
      --models "$EVALUATOR_MODELS" \
      --author-metadata "$AUTHOR_METADATA" \
      --doctrine-glossary "$DOCTRINE_GLOSSARY" \
      --report "$OUTPUT_DIR/$slug-$version_id-hierarchical-review.md" \
      --verbose \
      --database "$DATABASE"
    echo "completed" > "$marker_prefix.review.done"
  else
    echo "Hierarchical review already complete for $version_id; skipping."
  fi
}

run_translate "1 Nephi" "1-nephi"
run_translate "2 Nephi" "2-nephi"

for book in "1 Nephi" "2 Nephi"; do
  if [[ "$book" == "1 Nephi" ]]; then slug="1-nephi"; else slug="2-nephi"; fi
  ids_file="$STATE_DIR/$slug.translation-versions"
  while IFS= read -r version_id; do
    [[ -n "$version_id" ]] || continue
    run_version_workflow "$book" "$slug" "$version_id"
  done < "$ids_file"
done

echo "Nephi books run finished at $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "Logs, resumable state, and reports: $OUTPUT_DIR"
