# Deploying to AWS Glue

Three Glue jobs run the pipeline: `gp-raw-ingest`, `gp_raw_to_silver`,
`gp_silver_to_gold`. Each is created **by hand once**, captured to JSON here,
and from then on updated by CI.

Manual-first is deliberate. All the uncertainty is in the job parameters, and
the console gives a 30-second edit-run loop where CI gives a five-minute one.
knowledge lives, and leaving it in the console makes it unreproducible.

## What goes to S3, and what reads it

`python -m build --wheel` produces a wheel containing **only** `python_scripts*`
— `packages.find` in `pyproject.toml` excludes `config/` and `glue_jobs/`, so
those upload as loose files. Three artifacts, three different mechanisms:

```
s3://globalpartners-project-cjk/artifacts/<git-sha>/
├── globalpartners_pipeline-0.1.0-py3-none-any.whl   --additional-python-modules
├── glue_jobs/
│   ├── raw_ingest.py                                 ScriptLocation
│   ├── raw_to_silver.py                              ScriptLocation
│   └── silver_to_gold.py                             ScriptLocation
└── config/
    ├── local.yaml                                    --extra-files
    └── prod.yaml
```

- **The wheel** is pip-installed into the job environment, so
  `from python_scripts... import main` resolves.
- **The entry script** is what `ScriptLocation` points at. Glue wants one `.py`
  file and does not run `python -m`, which is the only reason `glue_jobs/`
  exists. One job per script.
- **The config** arrives through `--extra-files`, which **flattens** to the job's
  working directory — `s3://…/config/local.yaml` lands at `./local.yaml`, not
  `./config/local.yaml`. That is why the jobs pass `--config-dir .`

Without `--config-dir`, `load_config` looks two directories above
`config_loader.py`, which is the repo root locally but `site-packages/` once the
wheel is installed. It would raise `FileNotFoundError`.

## First-time manual setup

### 1. Roles

Create both from the files in `infra/iam/`, replacing `__ACCOUNT_ID__`,
`__GITHUB_OWNER__` and `__GITHUB_REPO__`:

- `glue-job-execution-role.json` — assumed by the jobs
- `github-actions-deploy-role.json` — assumed by CI via OIDC; put the ARN in the
  repo secret `AWS_DEPLOY_ROLE_ARN`

Every S3 statement is prefix-scoped, never bucket-root. That is not only least
privilege: a bucket-wide lifecycle or permission rule would also cover the
Iceberg warehouse, and transitioning those objects to Glacier breaks the tables
outright. Iceberg needs its metadata and all current data files readable on
demand.

### 2. Upload artifacts once by hand

Run the `deploy` workflow with `workflow_dispatch` and copy the prefix from its
summary, or do it locally:

```bash
pip install build
python -m build --wheel

PREFIX=s3://globalpartners-project-cjk/artifacts/manual
aws s3 cp dist/ "$PREFIX/" --recursive --exclude "*" --include "*.whl"
aws s3 cp glue_jobs/ "$PREFIX/glue_jobs/" --recursive --include "*.py"
aws s3 cp config/local.yaml "$PREFIX/config/local.yaml"
```

A `manual/` prefix keeps the hand-made first job out of the commit-pinned space.

### 3. Create one job in the console

Start with `gp-raw-to-silver` restricted to `date_dim`: 365 rows, `replace`
strategy, no watermark, no database. It still exercises everything uncertain —
catalog config, jar loading, config delivery, IAM, S3 layout. A failure costs
about $0.03.

| setting | value |
|---|---|
| Glue version | 6.0 |
| Worker type | G.1X |
| **Number of workers** | **2** |
| **Job timeout** | **10** minutes |
| Max retries | 0 |
| Script location | `s3://…/artifacts/manual/glue_jobs/raw_to_silver.py` |

Job parameters:

| key | value |
|---|---|
| `--datalake-formats` | `iceberg` |
| `--conf` | `spark.sql.extensions=org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions` |
| `--additional-python-modules` | `s3://…/artifacts/manual/globalpartners_pipeline-0.1.0-py3-none-any.whl` |
| `--extra-files` | `s3://…/artifacts/manual/config/local.yaml` |
| `--config-dir` | `.` |
| `--env` | `local` |
| `--period` | `2026-10-16` |
| `--tables` | `date_dim` |
| `--TempDir` | `s3://globalpartners-project-cjk/glue-temp/` |
| `--enable-continuous-cloudwatch-log` | `true` |

Those first two cannot come from Python. Jars can't join a running JVM, and
`spark.sql.extensions` is a static config read once at session creation — the
catalog settings themselves *are* applied from `_catalog_settings` in
`spark_session.py`, because Spark resolves catalogs lazily.

**Do not add `--enable-glue-datacatalog`.** It points `spark_catalog` at a
Glue-backed Hive metastore and has nothing to do with the named `iceberg`
catalog. It only adds a variable.

### 4. Capture the job

```bash
mkdir -p infra/glue
aws glue get-job --job-name gp-raw-to-silver > infra/glue/gp-raw-to-silver.json
```

Then run it through `prepare_job_update.py` in capture mode, which strips the
read-only fields and templates the artifact prefix:

```bash
aws glue get-job --job-name gp_raw_to_silver > get-job.json

python infra/glue/prepare_job_update.py capture get-job.json   --real-prefix s3://globalpartners-project-cjk/artifacts/manual   > infra/glue/gp_raw_to_silver.json
```

Commit the result. Verify three fields before trusting it:

| field | required | why |
|---|---|---|
| `NumberOfWorkers` | `2` | the console defaults to 10 -- 5x the bill for a 15 MB dataset |
| `Timeout` | 10-20 | the default `2880` is **48 hours**; a hung job at 2 DPU bills ~$42 |
| `MaxCapacity` | **absent** | API conflict: Glue 2.0+ takes `WorkerType`+`NumberOfWorkers` **or** `MaxCapacity`, never both |

`capture` removes `MaxCapacity` for you -- `get-job` does return it, set to the
DPU total, even on a WorkerType job.

## Deploying an update

CI substitutes the commit being deployed and applies it:

```bash
python infra/glue/prepare_job_update.py deploy infra/glue/gp_raw_to_silver.json   --prefix "s3://$S3_BUCKET/artifacts/$GITHUB_SHA" > job-update.json

aws glue update-job --job-name gp_raw_to_silver --job-update file://job-update.json

# On Windows/Git Bash use a RELATIVE path like this. `/tmp` is bash's own
# mapping (C:/Users/<you>/AppData/Local/Temp), while the AWS CLI is a native
# Windows exe that reads file:///tmp/... as C:\tmp\... and cannot find it.
```

Python rather than jq so the same tool runs on Windows and on the CI runner.
The thing it exists to handle: **`get-job` output is not valid `update-job`
input.** `JobUpdate` rejects the read-only fields `get-job` returns, and the
error names the field rather than the reason, so it reads as a config problem.

Rollback is re-running the workflow at an older commit; nothing points at a
mutable `latest` prefix.

## The daily schedule

Glue alone runs the chain -- no EventBridge rule, no Step Function. One
`SCHEDULED` trigger starts it and two `CONDITIONAL` triggers carry it forward,
all three inside the `gp_daily` workflow:

```
cron(0 7 * * ? *)  ->  gp_raw_ingest  --SUCCEEDED->  gp_raw_to_silver  --SUCCEEDED->  gp_silver_to_gold
```

The definition is `infra/glue/workflows/gp_daily.json`, and `prepare_workflow.py`
is its `prepare_job_update.py`: `capture`, `check`, `names`, `emit`.

**It lives one directory down from the jobs on purpose.** `infra/glue/*.json`
is one file per job, and both the validate step and the deploy step take the
job name from the filename -- a workflow beside them would be deployed as
`aws glue update-job --job-name gp_daily` and fail.

### Four things about it that are not obvious

**Nothing is templated, and `check` still matters.** A job definition carries
the artifact prefix and a role ARN; a workflow carries neither, because a
trigger names its jobs directly. So there is nothing to substitute -- `check`
only has to prove nothing crept in, and it fails on any 12-digit number or any
`arn:aws:` string.

**CI creates as well as updates, unlike the jobs.** Manual-first exists because
all the uncertainty is in the job parameters. A workflow has no equivalent
uncertainty, so a fresh account comes up from a push alone.

**CI cannot arm the schedule.** The deploy role has no `glue:StartTrigger`,
`StopTrigger`, `DeleteTrigger` or `DeleteWorkflow`. A push can change *what*
the daily run does; only a human changes *whether* it runs, and CI cannot
dismantle the orchestration at all:

```bash
aws glue start-trigger --name gp_daily_start   # arm
aws glue stop-trigger  --name gp_daily_start   # disarm
aws glue get-trigger   --name gp_daily_start --query Trigger.State
```

`StartOnCreation` in the committed file applies **only at creation**. Flipping
it later changes nothing about a trigger that already exists, which is why the
file's note points at `get-trigger` for the live answer.

**`create-trigger` and `update-trigger` take different documents.**
`TriggerUpdate` has no `Type` and no `WorkflowName` -- Glue will not let a
trigger change either -- so sending the create shape to `update-trigger` is
rejected. `prepare_workflow.py emit --op create|update` is that difference.

### Why `--period` is not in DefaultArguments

It was, pinned to `2026-10-16`, and a daily trigger would have reprocessed that
one day forever. Glue triggers have no expression language, so a schedule
cannot compute "yesterday" -- which is exactly why `--period` defaults to
`helpers.yesterday_utc()` in the first place. The fix is to **omit** the
argument, not to set it. `prepare_job_update.py check` warns if it comes back.

Backfill by passing it explicitly on a job run. `--period 2026-10-16` is the
only date that selects the historical load, since every row shares that
`inserted_at`; each job's `Description` says so.

### Capturing a console change

`infra/glue/capture_jobs.sh` now captures the workflow and its triggers too.
Two things it handles that a hand-rolled `aws glue get-triggers > file` would
not: `get-triggers` returns **every** trigger in the account, so the capture
keeps only those whose `WorkflowName` matches and reports how many it ignored;
and `StartOnCreation` is carried over from the committed file, because AWS does
not return it -- a trigger's live armed state is the read-only `State` field,
so a naive capture would silently rewrite your intent.

The same Windows caveat as above applies if you run the sync by hand: the AWS
CLI is a native exe and reads `file:///tmp/...` as `C:\tmp\...`. Use a
relative path.

