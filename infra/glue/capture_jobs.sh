#!/usr/bin/env bash
# Capture every Glue job definition into infra/glue/ as a committable JobUpdate.
#
# Run from the repo root, after creating or changing a job in the console:
#   bash infra/glue/capture_jobs.sh
#
# For each job: `aws glue get-job` -> prepare_job_update.py capture -> check.
# The raw dumps land in a temp directory and are deleted, because they carry the
# real account id and real artifact paths. Only the templated output is written
# into the repo.
#
# The account id is derived from STS rather than stored anywhere.
set -euo pipefail

# Job name == this file's basename == the entry script's basename. Keeping all
# three aligned is what lets CI derive the job name and the ScriptLocation from
# one place. Override with e.g. JOBS="a b" bash infra/glue/capture_jobs.sh
read -r -a JOBS <<< "${JOBS:-gp_raw_ingest gp_raw_to_silver gp_silver_to_gold}"
ARTIFACT_PREFIX="${ARTIFACT_PREFIX:-s3://globalpartners-project-cjk/artifacts/manual}"
OUT_DIR="${OUT_DIR:-infra/glue}"

# The venv interpreter differs by platform; fall back to whatever is on PATH.
if   [ -x .venv/Scripts/python.exe ]; then PY=.venv/Scripts/python.exe
elif [ -x .venv/bin/python ];        then PY=.venv/bin/python
else                                      PY=python
fi

if [ ! -f infra/glue/prepare_job_update.py ]; then
  echo "run this from the repo root -- infra/glue/prepare_job_update.py not found" >&2
  exit 1
fi

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

echo
echo "=== validating ==="
status=0
for f in "${captured[@]}"; do
  "$PY" infra/glue/prepare_job_update.py check "$f" || status=1
done

echo
if [ "$status" -ne 0 ]; then
  echo "FAILED -- do not commit until the errors above are resolved" >&2
  exit 1
fi

echo "captured ${#captured[@]} job(s):"
for f in "${captured[@]}"; do echo "  $f"; done
echo
echo "Warnings are advisory. Errors mean something leaked or update-job would"
echo "reject the file, and block the script."
