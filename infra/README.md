# Deploying to AWS Glue

Three Glue jobs run the pipeline: `gp-raw-ingest`, `gp-raw-to-silver`,
`gp-silver-to-gold`. Each is created **by hand once**, captured to JSON here,
and from then on updated by CI.

Manual-first is deliberate. All the uncertainty is in the job parameters, and
the console gives a 30-second edit-run loop where CI gives a five-minute one.
Manual-only is the mistake — the job configuration is where the Iceberg-on-Glue
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
# Windows exe that reads file:///tmp/... as C:	mp\... and cannot find it.
```

Python rather than jq so the same tool runs on Windows and on the CI runner.
The thing it exists to handle: **`get-job` output is not valid `update-job`
input.** `JobUpdate` rejects the read-only fields `get-job` returns, and the
error names the field rather than the reason, so it reads as a config problem.

Rollback is re-running the workflow at an older commit; nothing points at a
mutable `latest` prefix.

## Open items

- **`GLUE_COMMAND_ARGUMENTS` is unverified.** `create_spark_session` uses it to
  detect Glue. If it is not set in 6.0, the Glue branch never runs, execution
  falls into the local path, and it tries to resolve jars through Ivy. A
  `print(sorted(os.environ))` in the first run settles it;
  `"--JOB_NAME" in sys.argv` is the fallback.
- **Which env prefix to run.** The existing raw data is under `local/raw/`, so
  `--env prod` reads an empty prefix and silently produces an empty silver
  table. Either run `--env local` on Glue or copy the prefix first.
- **No maintenance job.** Nothing runs `expire_snapshots` or
  `remove_orphan_files`, so storage and metadata grow. Pennies at this volume,
  but expiring snapshots is also what would bound time travel — set
  `history.expire.max-snapshot-age-ms` deliberately if the dashboard's snapshot
  picker becomes a real feature.
