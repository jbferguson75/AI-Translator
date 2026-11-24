#!/usr/bin/env bash
# Import (if needed), translate, evaluate, refine, synthesize, review, and
# produce a side-by-side report for 1 Nephi chapter 1 using llama and gemma.

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
SOURCE_JSON="${SOURCE_JSON:-}"
DOCUMENT_UUID="${DOCUMENT_UUID:-}"
TARGET_LOCALE="${TARGET_LOCALE:-en}"
READING_LEVEL="${READING_LEVEL:-4th grade}"
TASK_KIND="${TASK_KIND:-simplification}"
MODELS="${MODELS:-llama,gemma}"
REVIEW_LEVELS="${REVIEW_LEVELS:-1,2,3,4}"
AUTHOR_METADATA="${AUTHOR_METADATA:-$ROOT/plugins/scripture/src/scripture_plugin/data/authors.json}"
DOCTRINE_GLOSSARY="${DOCTRINE_GLOSSARY:-$ROOT/plugins/scripture/src/scripture_plugin/data/doctrine_glossary.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/reports/nephi-chapter1-$(date -u +%Y%m%dT%H%M%SZ)}"

for executable in "$PYTHON_BIN" "$AI_TRANSLATOR_BIN" "$SCRIPTURE_BIN"; do
  if [[ ! -x "$executable" ]]; then
    echo "Required executable not found: $executable" >&2
    exit 1
  fi
done
for file in "$TRANSLATION_CONFIG" "$EVALUATION_CONFIG" "$AUTHOR_METADATA" "$DOCTRINE_GLOSSARY"; do
  if [[ ! -f "$file" ]]; then
    echo "Required config file not found: $file" >&2
    exit 1
  fi
done

mkdir -p "$OUTPUT_DIR"
LOG_FILE="$OUTPUT_DIR/workflow.log"
exec > >(tee -a "$LOG_FILE") 2>&1
set -x
trap 'status=$?; echo "ERROR: command failed with exit $status at line ${LINENO}; see the preceding command output in $LOG_FILE" >&2' ERR

ai() { "$AI_TRANSLATOR_BIN" "$@"; }
scr() { "$SCRIPTURE_BIN" "$@"; }

capture() {
  local output_file="$1"
  shift
  set +e
  "$@" 2>&1 | tee "$output_file"
  local status="${PIPESTATUS[0]}"
  set -e
  return "$status"
}

extract_completed_ids() {
  local output_file="$1"
  local id_file="$2"
  sed -nE '/: completed, version /s/.*\[id ([[:xdigit:]-]{36})\].*/\1/p' \
    "$output_file" | awk '!seen[$0]++' > "$id_file"
}

echo "Starting 1 Nephi chapter 1 workflow at $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "Project: $PROJECT | Database: $DATABASE | Models: $MODELS"
echo "Target: $TARGET_LOCALE ($TASK_KIND, $READING_LEVEL) | Review levels: $REVIEW_LEVELS"
echo "Reports and logs: $OUTPUT_DIR"

ai db upgrade --database "$DATABASE"

if [[ -z "$DOCUMENT_UUID" ]]; then
  set +e
  DOCUMENT_UUID="$(PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" - "$PROJECT" "$DATABASE" <<'PY'
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
        raise SystemExit(f"No project named {project_name!r}. Create it with 'ai-translator project create'.")
    documents = list(session.scalars(select(Document).where(
        Document.project_id == project.id, Document.domain == "scripture",
    ).order_by(Document.created_at)))
    if len(documents) != 1:
        raise SystemExit(
            f"Expected one scripture document for {project_name!r}, found {len(documents)}. "
            "Set DOCUMENT_UUID explicitly."
        )
    print(documents[0].id)
finally:
    session.close()
    engine.dispose()
PY
  )"
  lookup_status=$?
  set -e
  if [[ "$lookup_status" != "0" ]]; then
    if [[ -z "$SOURCE_JSON" ]]; then
      echo "Document lookup failed. Set DOCUMENT_UUID, or set SOURCE_JSON to import a scripture JSON file." >&2
      exit "$lookup_status"
    fi
    import_output="$OUTPUT_DIR/import.log"
    capture "$import_output" ai import scripture "$SOURCE_JSON" \
      --project "$PROJECT" --source-locale en --database "$DATABASE"
    DOCUMENT_UUID="$(sed -nE 's/.*\(document ([[:xdigit:]-]{36})\).*/\1/p' "$import_output" | tail -n 1)"
  fi
fi

if [[ ! "$DOCUMENT_UUID" =~ ^[[:xdigit:]-]{36}$ ]]; then
  echo "DOCUMENT_UUID is not a UUID: $DOCUMENT_UUID" >&2
  exit 1
fi
echo "Using scripture document $DOCUMENT_UUID"

translation_output="$OUTPUT_DIR/translate.log"
capture "$translation_output" ai translate "$DOCUMENT_UUID" \
  --book "1 Nephi" --chapter 1 \
  --target-locale "$TARGET_LOCALE" --task-kind "$TASK_KIND" \
  --reading-level "$READING_LEVEL" --models "$MODELS" \
  --config "$TRANSLATION_CONFIG" --verbose --database "$DATABASE"

ORIGINAL_IDS_FILE="$OUTPUT_DIR/original-version-ids.txt"
extract_completed_ids "$translation_output" "$ORIGINAL_IDS_FILE"
if [[ ! -s "$ORIGINAL_IDS_FILE" ]]; then
  echo "No completed original translations were found in $translation_output" >&2
  exit 1
fi

ALL_VERSION_IDS=()
REFINED_VERSION_IDS=()
while IFS= read -r original_id; do
  [[ -n "$original_id" ]] || continue
  ALL_VERSION_IDS+=("$original_id")

  scr review-report "$original_id" \
    --output "$OUTPUT_DIR/$original_id-glossary-scan.md" \
    --author-metadata "$AUTHOR_METADATA" --doctrine-glossary "$DOCTRINE_GLOSSARY" \
    --database "$DATABASE"
  capture "$OUTPUT_DIR/$original_id-scripture-evaluation.log" \
    scr evaluate "$original_id" --config "$EVALUATION_CONFIG" \
    --models "$MODELS" --author-metadata "$AUTHOR_METADATA" \
    --doctrine-glossary "$DOCTRINE_GLOSSARY" --verbose --database "$DATABASE"

  refinement_output="$OUTPUT_DIR/$original_id-refine.log"
  capture "$refinement_output" ai refine "$original_id" \
    --config "$TRANSLATION_CONFIG" --evaluation-config "$EVALUATION_CONFIG" \
    --models "$MODELS" --evaluator-models "$MODELS" --verbose --database "$DATABASE"
  refined_ids_file="$OUTPUT_DIR/$original_id-refined-version-ids.txt"
  extract_completed_ids "$refinement_output" "$refined_ids_file"
  if [[ ! -s "$refined_ids_file" ]]; then
    echo "No completed refinement candidates were found for $original_id" >&2
    exit 1
  fi
  while IFS= read -r refined_id; do
    [[ -n "$refined_id" ]] || continue
    REFINED_VERSION_IDS+=("$refined_id")
    ALL_VERSION_IDS+=("$refined_id")
  done < "$refined_ids_file"
done < "$ORIGINAL_IDS_FILE"

if (( ${#REFINED_VERSION_IDS[@]} < 2 )); then
  echo "Synthesis requires at least two completed refined candidates." >&2
  exit 1
fi

synthesis_output="$OUTPUT_DIR/synthesize.log"
capture "$synthesis_output" ai synthesize "${REFINED_VERSION_IDS[@]}" \
  --config "$TRANSLATION_CONFIG" --evaluation-config "$EVALUATION_CONFIG" \
  --models "$MODELS" --evaluator-models "$MODELS" --verbose --database "$DATABASE"
SYNTHESIS_IDS_FILE="$OUTPUT_DIR/synthesized-version-ids.txt"
extract_completed_ids "$synthesis_output" "$SYNTHESIS_IDS_FILE"
if [[ ! -s "$SYNTHESIS_IDS_FILE" ]]; then
  echo "No completed synthesis candidates were found in $synthesis_output" >&2
  exit 1
fi

while IFS= read -r synthesized_id; do
  [[ -n "$synthesized_id" ]] || continue
  ALL_VERSION_IDS+=("$synthesized_id")
  scr review-report "$synthesized_id" \
    --output "$OUTPUT_DIR/$synthesized_id-glossary-scan.md" \
    --author-metadata "$AUTHOR_METADATA" --doctrine-glossary "$DOCTRINE_GLOSSARY" \
    --database "$DATABASE"
  capture "$OUTPUT_DIR/$synthesized_id-scripture-review.log" \
    scr review all "$synthesized_id" --levels "$REVIEW_LEVELS" \
    --config "$EVALUATION_CONFIG" --models "$MODELS" \
    --author-metadata "$AUTHOR_METADATA" --doctrine-glossary "$DOCTRINE_GLOSSARY" \
    --verbose --database "$DATABASE"
done < "$SYNTHESIS_IDS_FILE"

ai review compare "${ALL_VERSION_IDS[@]}" --database "$DATABASE" \
  > "$OUTPUT_DIR/evaluation-comparison.json"
ai report side-by-side "${ALL_VERSION_IDS[@]}" \
  --output "$OUTPUT_DIR/1-nephi-chapter-1-side-by-side.md" --database "$DATABASE"
while IFS= read -r synthesized_id; do
  [[ -n "$synthesized_id" ]] || continue
  ai report lineage "$synthesized_id" --output "$OUTPUT_DIR/$synthesized_id-lineage.json" \
    --database "$DATABASE"
done < "$SYNTHESIS_IDS_FILE"

printf '%s\n' "${ALL_VERSION_IDS[@]}" > "$OUTPUT_DIR/all-version-ids.txt"
echo "Workflow completed at $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "Side-by-side report: $OUTPUT_DIR/1-nephi-chapter-1-side-by-side.md"
echo "Evaluation comparison: $OUTPUT_DIR/evaluation-comparison.json"
echo "All version IDs: $OUTPUT_DIR/all-version-ids.txt"
