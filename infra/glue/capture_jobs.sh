#!/usr/bin/env bash
# Capture the Glue job definitions, and the workflow that orchestrates them,
# into infra/glue/ as committable documents.
#
# Run from the repo root, after creating or changing anything in the console:
#   bash infra/glue/capture_jobs.sh
#
# For each job:  `aws glue get-job`      -> prepare_job_update.py capture -> check
# For the chain: `aws glue get-workflow`
#                `aws glue get-triggers` -> prepare_workflow.py  capture -> check
#
# The raw dumps land in a temp directory and are deleted, because they carry the
# real account id and real artifact paths. Only the templated output is written
# into the repo.
#
# The account id is derived from STS rather than stored anywhere.
#
# A workflow carries no prefix and no ARN -- a trigger names its jobs directly
# -- so there is nothing in it to template. Its capture is still checked, and
# the check FAILS on any 12-digit number, so a leak cannot ride along silently.
set -euo pipefail

# Job name == this file's basename == the entry script's basename. Keeping all
# three aligned is what lets CI derive the job name and the ScriptLocation from
# one place. Override with e.g. JOBS="a b" bash infra/glue/capture_jobs.sh
read -r -a JOBS <<< "${JOBS:-gp_raw_ingest gp_raw_to_silver gp_silver_to_gold}"
ARTIFACT_PREFIX="${ARTIFACT_PREFIX:-s3://globalpartners-project-cjk/artifacts/manual}"
OUT_DIR="${OUT_DIR:-infra/glue}"
# The orchestration. One level down from the jobs because infra/glue/*.json is
# globbed as one-file-per-job by both the validate and the deploy step, which
# derive the job name from the filename.
WORKFLOW="${WORKFLOW:-gp_daily}"
WORKFLOW_DIR="${WORKFLOW_DIR:-infra/glue/workflows}"

# The venv interpreter differs by platform; fall back to whatever is on PATH.
if   [ -x .venv/Scripts/python.exe ]; then PY=.venv/Scripts/python.exe
elif [ -x .venv/bin/python ];        then PY=.venv/bin/python
else                                      PY=python
fi

for helper in infra/glue/prepare_job_update.py infra/glue/prepare_workflow.py; do
  if [ ! -f "$helper" ]; then
    echo "run this from the repo root -- ${helper} not found" >&2
    exit 1
  fi
done

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
echo "account ${ACCOUNT_ID} (masked in every output file, never committed)"
echo "prefix  ${ARTIFACT_PREFIX}"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

mkdir -p "$OUT_DIR"
captured=()

for job in "${JOBS[@]}"; do
  echo
  echo "=== ${job} ==="

  if ! aws glue get-job --job-name "$job" > "${TMP}/${job}.getjob.json" 2>"${TMP}/err"; then
    echo "  skipped -- $(tr -d '\n' < "${TMP}/err")" >&2
    continue
  fi

  "$PY" infra/glue/prepare_job_update.py capture "${TMP}/${job}.getjob.json" \
    --real-prefix "$ARTIFACT_PREFIX" \
    --account-id "$ACCOUNT_ID" \
    > "${OUT_DIR}/${job}.json"

  captured+=("${OUT_DIR}/${job}.json")
done

# --- the workflow and its triggers ------------------------------------------
#
# get-triggers returns EVERY trigger in the account, including other projects';
# prepare_workflow.py keeps only the ones whose WorkflowName matches and says
# how many it ignored.
#
# --merge-state carries StartOnCreation over from the committed file. AWS does
# not return it -- a trigger's live armed/disarmed state is `State`, which is
# read-only -- so without this a capture would silently rewrite the file's
# intent to whatever the default is.
echo
echo "=== ${WORKFLOW} ==="

mkdir -p "$WORKFLOW_DIR"
WORKFLOW_FILE="${WORKFLOW_DIR}/${WORKFLOW}.json"

if ! aws glue get-workflow --name "$WORKFLOW" > "${TMP}/workflow.json" 2>"${TMP}/err"; then
  echo "  skipped -- $(tr -d '
' < "${TMP}/err")" >&2
elif ! aws glue get-triggers > "${TMP}/triggers.json" 2>"${TMP}/err"; then
  echo "  skipped -- $(tr -d '
' < "${TMP}/err")" >&2
else
  merge=()
  [ -f "$WORKFLOW_FILE" ] && merge=(--merge-state "$WORKFLOW_FILE")

  "$PY" infra/glue/prepare_workflow.py capture "${TMP}/workflow.json"     --triggers "${TMP}/triggers.json" "${merge[@]}"     > "${TMP}/captured_workflow.json"

  mv "${TMP}/captured_workflow.json" "$WORKFLOW_FILE"
  workflow_captured="$WORKFLOW_FILE"
fi

echo
echo "=== validating ==="
status=0
for f in "${captured[@]}"; do
  "$PY" infra/glue/prepare_job_update.py check "$f" || status=1
done

if [ -n "${workflow_captured:-}" ]; then
  "$PY" infra/glue/prepare_workflow.py check "$workflow_captured"     --job-dir "$OUT_DIR" || status=1
fi

echo
if [ "$status" -ne 0 ]; then
  echo "FAILED -- do not commit until the errors above are resolved" >&2
  exit 1
fi

echo "captured ${#captured[@]} job(s):"
for f in "${captured[@]}"; do echo "  $f"; done
if [ -n "${workflow_captured:-}" ]; then
  echo "and the workflow:"
  echo "  $workflow_captured"
fi
echo
echo "Warnings are advisory. Errors mean something leaked or update-job would"
echo "reject the file, and block the script."
