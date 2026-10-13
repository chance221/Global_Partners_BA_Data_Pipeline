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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

VALID_LOAD_TYPES = frozenset({"static", "full"})
REQUIRED_SECTIONS = ("aws", "database", "layers", "tables")

DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


@dataclass(frozen=True)
class TableConfig:
    """One dataset's ingestion contract.

    Args:
        source_table: schema-qualified JDBC identifier, e.g. "dbo.order_items".
        load_type: "static" (one copy, overwritten) or "full" (one snapshot per
            batch date, written under batch_year=/batch_month=/batch_day=).
        business_keys: columns identifying a row for the SCD1 merge.
        required_columns: column name -> expected source type. A contract
            subset, not the full schema; unlisted columns are allowed.
    """

    source_table: str
    load_type: str
    business_keys: list[str]
    required_columns: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.load_type not in VALID_LOAD_TYPES:
            raise ValueError(
                f"load_type {self.load_type!r} is not one of "
                f"{sorted(VALID_LOAD_TYPES)}"
            )

        if not self.business_keys:
            raise ValueError("business_keys must not be empty")

        duplicates = {k for k in self.business_keys
                      if self.business_keys.count(k) > 1}
        if duplicates:
            raise ValueError(f"business_keys contains duplicates: {sorted(duplicates)}")

        # A business key is by definition a column we depend on, so it must also
        # appear in required_columns. This catches a rename applied in one place
        # and not the other.
        undeclared = set(self.business_keys) - set(self.required_columns)
        if self.required_columns and undeclared:
            raise ValueError(
                f"business_keys not listed in required_columns: {sorted(undeclared)}"
            )


@dataclass(frozen=True)
class AppConfig:
    """Flattened view of one environment's config file."""

    env: str
    region: str
    secret_name: str
    bucket: str
    database: str
    layers: dict[str, str]
    tables: dict[str, TableConfig]

    def get_table_config(self, name: str) -> TableConfig:
        """Look up one dataset, with an error that lists the valid names."""
        try:
            return self.tables[name]
        except KeyError:
            raise KeyError(
                f"unknown dataset {name!r}; config defines {sorted(self.tables)}"
            ) from None

    def layer(self, name: str) -> str:
        """Resolve a layer name ("raw" -> the configured path segment)."""
        try:
            return self.layers[name]
        except KeyError:
            raise KeyError(
                f"unknown layer {name!r}; config defines {sorted(self.layers)}"
            ) from None


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

    try:
        return AppConfig(
            env=env,
            region=raw["aws"]["region"],
            secret_name=raw["aws"]["secret_name"],
            bucket=raw["aws"]["bucket"],
            database=raw["database"]["name"],
            layers=raw["layers"],
            tables=tables,
        )
    except KeyError as exc:
        raise ValueError(f"{path}: missing required key {exc}") from exc
