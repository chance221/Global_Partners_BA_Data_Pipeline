"""Load and validate a per-environment pipeline config.

Config files live in config/{env}.yaml and are selected by the --env CLI
argument. Validation happens at load time so a bad config fails in seconds,
before a Spark session is built or any S3 object is written.

Two fields are validated strictly because being wrong about them is not merely
an error -- it is destructive or silently corrupting:

  load_type     -- drives the S3 path shape. An unrecognised value would send a
                   full snapshot to the dataset root, overwriting every
                   historical partition.
  business_keys -- drives the SCD1 merge grain. A wrong value collapses
                   distinct rows into one with no error raised.
"""

from __future__ import annotations


from pathlib import Path
from typing import Any

from python_scripts.common.models.config_models import (
    AppConfig,
    GoldTableConfig,
    TableConfig,
)

import yaml


REQUIRED_SECTIONS = ("aws", "database", "layers", "tables")

DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


def load_config(env: str, config_dir: Path | str | None = None) -> AppConfig:
    """Read and validate config/{env}.yaml.

    Args:
        env: environment name, matching the config filename and the {env}
            segment of the S3 path.
        config_dir: directory holding the config files. Defaults to the repo's
            config/ directory, resolved from this file rather than the working
            directory so it works under `python -m` and on Glue.

    Returns:
        A validated AppConfig.

    Raises:
        FileNotFoundError: no config file for this environment.
        ValueError: the file is malformed, or a table entry is invalid. The
            message names the offending section or dataset.
    """
    directory = Path(config_dir) if config_dir else DEFAULT_CONFIG_DIR
    path = directory / f"{env}.yaml"

    if not path.is_file():
        available = sorted(p.stem for p in directory.glob("*.yaml"))
        raise FileNotFoundError(
            f"no config for env {env!r} at {path}; available: {available}"
        )

    with path.open(encoding="utf-8") as handle:
        raw: Any = yaml.safe_load(handle)

    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a mapping at the top level, got {type(raw).__name__}")

    missing = [s for s in REQUIRED_SECTIONS if s not in raw]
    if missing:
        raise ValueError(f"{path}: missing section(s) {missing}")

    if not raw["tables"]:
        raise ValueError(f"{path}: 'tables' section is empty")

    tables: dict[str, TableConfig] = {}
    for name, entry in raw["tables"].items():
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: table {name!r} must be a mapping")
        try:
            tables[name] = TableConfig(**entry)
        except TypeError as exc:
            # Missing or unexpected key -- TypeError names the field but not the
            # table, so re-raise with the dataset included.
            raise ValueError(f"{path}: table {name!r}: {exc}") from exc
        except ValueError as exc:
            raise ValueError(f"{path}: table {name!r}: {exc}") from exc

    gold: dict[str, GoldTableConfig] = {}
    for name, entry in (raw.get("gold") or {}).items():
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: gold table {name!r} must be a mapping")
        try:
            gold[name] = GoldTableConfig(**entry)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path}: gold table {name!r}: {exc}") from exc

    catalog = raw.get("catalog") or {}
    if not isinstance(catalog, dict):
        raise ValueError(f"{path}: 'catalog' section must be a mapping")

    try:
        return AppConfig(
            env=env,
            region=raw["aws"]["region"],
            secret_name=raw["aws"]["secret_name"],
            bucket=raw["aws"]["bucket"],
            database=raw["database"]["name"],
            layers=raw["layers"],
            tables=tables,
            gold=gold,
            catalog_type=catalog.get("type", "hadoop"),
            warehouse=catalog.get("warehouse"),
        )
    except KeyError as exc:
        raise ValueError(f"{path}: missing required key {exc}") from exc
