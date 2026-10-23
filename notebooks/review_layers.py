"""Read pipeline output into pandas for eyeball review -- no Spark session.

Importable helpers so the notebook stays short and these can be reused from a
REPL. Deliberately read-only: nothing here writes, deletes or moves an object.

Paths come from `build_path` and the loaded config rather than string literals,
so this cannot drift from what the jobs actually wrote. The one translation is
the scheme: the jobs address `s3a://bucket/key` (a Hadoop filesystem), while
pyarrow's S3FileSystem wants `bucket/key` with no scheme at all.

pyarrow rather than s3fs on purpose -- s3fs depends on aiobotocore, which pins
botocore to a narrow range and conflicts with the installed boto3. pyarrow
talks to S3 through its own bundled AWS SDK and resolves credentials from the
usual chain (~/.aws/credentials, environment, instance role).
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pyarrow.fs as pafs

from python_scripts.common.config_loader import load_config
from python_scripts.common.models.config_models import AppConfig
from python_scripts.io_utils.paths import build_path

# The jobs write with the s3a:// Hadoop scheme; pyarrow wants a bare key.
_HADOOP_SCHEMES = ("s3a://", "s3://")


def connect(env: str = "local") -> tuple[AppConfig, pafs.S3FileSystem]:
    """Load the config and open an S3 filesystem in its region."""
    config = load_config(env)
    return config, pafs.S3FileSystem(region=config.region)


def _strip_scheme(path: str) -> str:
    for scheme in _HADOOP_SCHEMES:
        if path.startswith(scheme):
            return path[len(scheme):]
    return path


def layer_path(config: AppConfig, layer: str, dataset: str,
               batch_date: datetime | None = None) -> str:
    """Build a pyarrow-ready path for one dataset in one layer.

    `batch_date` None gives the dataset root, which reads every batch at once --
    useful for "show me everything that was ever quarantined".
    """
    load_type = "full" if batch_date else "static"
    full = build_path("s3a", config.bucket, config.env, config.layer(layer),
                      dataset, batch_date or datetime(1970, 1, 1), load_type)
    return _strip_scheme(full)


def read(fs: pafs.S3FileSystem, path: str) -> pd.DataFrame:
    """Read a parquet prefix into pandas, returning empty rather than raising.

    A missing prefix is the normal answer to "was anything quarantined?", so it
    is not treated as an error.
    """
    try:
        return pd.read_parquet(path, filesystem=fs)
    except (FileNotFoundError, OSError) as exc:
        if "Path does not exist" in str(exc) or isinstance(exc, FileNotFoundError):
            return pd.DataFrame()
        raise


def read_raw(config, fs, dataset: str,
             batch_date: datetime | None = None) -> pd.DataFrame:
    """Rows as landed by raw_ingest -- a faithful copy of the source."""
    return read(fs, layer_path(config, "raw", dataset, batch_date))


def read_quarantine(config, fs, dataset: str, stage: str,
                    batch_date: datetime | None = None) -> pd.DataFrame:
    """Rows held back, with their original values and a reason.

    Args:
        stage: "raw_ingest" or "raw_to_silver" -- each writes to its own prefix
            so one cannot overwrite the other.
    """
    return read(fs, layer_path(config, "quarantine", f"{stage}/{dataset}",
                               batch_date))


def list_prefixes(config, fs, layer: str | None = None) -> pd.DataFrame:
    """Inventory what actually exists: prefix, file count, total size.

    The quickest way to answer "did that run write anything, and where".
    """
    root = f"{config.bucket}/{config.env}"
    if layer:
        root = f"{root}/{config.layer(layer)}"

    selector = pafs.FileSelector(root, recursive=True, allow_not_found=True)
    files = [f for f in fs.get_file_info(selector) if f.path.endswith(".parquet")]

    if not files:
        return pd.DataFrame(columns=["prefix", "files", "bytes"])

    rows = [{"prefix": f.path.rsplit("/", 1)[0], "bytes": f.size} for f in files]
    return (
        pd.DataFrame(rows)
        .groupby("prefix", as_index=False)
        .agg(files=("bytes", "size"), bytes=("bytes", "sum"))
        .sort_values("prefix")
        .reset_index(drop=True)
    )


def reject_summary(config, fs, datasets: list[str] | None = None) -> pd.DataFrame:
    """Every quarantined row across both stages, with why it was held.

    Returns the reason counts rather than the rows, which is the shape you want
    when deciding whether a run is healthy.
    """
    datasets = datasets or list(config.tables)
    frames = []

    for dataset in datasets:
        for stage in ("raw_ingest", "raw_to_silver"):
            held = read_quarantine(config, fs, dataset, stage)
            if not held.empty:
                frames.append(held.assign(_dataset=dataset))

    if not frames:
        return pd.DataFrame(columns=["_dataset", "_quarantine_stage",
                                     "_reject_reason", "rows"])

    combined = pd.concat(frames, ignore_index=True)
    return (
        combined
        .groupby(["_dataset", "_quarantine_stage", "_reject_reason"],
                 as_index=False)
        .size()
        .rename(columns={"size": "rows"})
        .sort_values("rows", ascending=False)
        .reset_index(drop=True)
    )
