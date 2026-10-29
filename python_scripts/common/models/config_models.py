from dataclasses import dataclass, field

from python_scripts.io_utils.iceberg_write import VALID_WRITE_STRATEGIES
from python_scripts.spark.spark_session import VALID_CATALOG_TYPES
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

VALID_LOAD_TYPES = frozenset({"static", "full", "incremental"})

# How a batch is applied to the silver Iceberg table. Independent of load_type,
# which only says how much we extract -- only "merge" needs a unique key, which
# is why order_item_options can still be loaded incrementally despite having no
# key of its own.
#
#   merge          MERGE INTO on business_keys. Requires the keys to be unique
#                  within the batch; Iceberg raises if two source rows match one
#                  target row.
#   delete_insert  Delete every target row whose silver_scope_keys appear in the
#                  batch, then insert the batch.
#   replace        Rewrite every row. For small static dimensions.
VALID_SILVER_WRITES = VALID_WRITE_STRATEGIES


@dataclass(frozen=True)
class TableConfig:
    """Dataset's ingestion contract.

    Args:
        source_table: schema-qualified JDBC identifier, e.g. "dbo.order_items".
        load_type: "static" (one copy, overwritten) or "full" (one snapshot per
            batch date, written under batch_year=/batch_month=/batch_day=).
        business_keys: columns identifying a row for the SCD1 merge.
        required_columns: column name -> expected source type. A contract
            subset, not the full schema; unlisted columns are allowed.
        silver_cleaners: column name -> ordered [pattern, replacement] regex
            pairs applied to that column's VALUES. silver_transforms fixes a
            column's type; this fixes its content, for source data that is
            structurally valid but textually wrong. Order matters -- each
            pattern sees the output of the one before it.
    """

    source_table: str
    load_type: str
    business_keys: list[str]
    required_columns: dict[str, str] = field(default_factory=dict)
    silver_transforms: dict [str, str] = field(default_factory=dict)
    silver_cleaners: dict[str, list[list[str]]] = field(default_factory=dict)
    silver_write: str = "merge"
    silver_scope_keys: list[str] = field(default_factory=list)
    silver_partition_by: list[str] = field(default_factory=list)
    watermark_column: str | None = None

    def __post_init__(self) -> None:
        if self.load_type not in VALID_LOAD_TYPES:
            raise ValueError(
                f"load_type {self.load_type!r} is not one of "
                f"{sorted(VALID_LOAD_TYPES)}"
            )

        if self.silver_write not in VALID_SILVER_WRITES:
            raise ValueError(
                f"silver_write {self.silver_write!r} is not one of "
                f"{sorted(VALID_SILVER_WRITES)}"
            )

        # An unbounded delete would clear the table on every run and still
        # report success, so this is validated rather than defaulted.
        if self.silver_write == "delete_insert" and not self.silver_scope_keys:
            raise ValueError(
                "silver_write 'delete_insert' requires silver_scope_keys; "
                "without them the delete would match the entire table"
            )

        if self.silver_write != "delete_insert" and self.silver_scope_keys:
            raise ValueError(
                f"silver_scope_keys is only used by silver_write "
                f"'delete_insert', not {self.silver_write!r}"
            )
        
        if self.load_type == "incremental" and not self.watermark_column:
            raise ValueError(
                "load_type 'incremental' requires watermark_column"
            )

        # A malformed rule would otherwise surface as a Spark analysis error
        # deep in the job, naming a column but not the config that broke it.
        for column, rules in self.silver_cleaners.items():
            for rule in rules:
                if len(rule) != 2:
                    raise ValueError(
                        f"silver_cleaners[{column!r}] rule {rule!r} must be "
                        f"[pattern, replacement], got {len(rule)} element(s)"
                    )

        # Cleaning a column that is not carried into silver is a no-op the
        # author would never see, so it fails here instead.
        if self.required_columns:
            uncarried = {c for c in self.silver_cleaners
                         if c.lower() not in {r.lower()
                                              for r in self.required_columns}}
            if uncarried:
                raise ValueError(
                    f"silver_cleaners names columns absent from "
                    f"required_columns: {sorted(uncarried)}"
                )

        if not self.business_keys:
            raise ValueError("business_keys must not be empty")

        duplicates = {k for k in self.business_keys
                      if self.business_keys.count(k) > 1}
        if duplicates:
            raise ValueError(f"business_keys contains duplicates: {sorted(duplicates)}")

        # A business key is by definition a column we depend on, so it must also
        # appear in required_columns. 
        undeclared = set(self.business_keys) - set(self.required_columns)
        if self.required_columns and undeclared:
            raise ValueError(
                f"business_keys not listed in required_columns: {sorted(undeclared)}"
            )


@dataclass(frozen=True)
class GoldTableConfig:
    """Derived tables in the gold layer.

    Args:
        depends_on: silver tables the transform reads.
        grain_keys: columns that make a gold row unique.
        scope_keys: for write "delete_insert", the columns whose values the
            delete covers. Must be a subset of grain_keys -- scoping by
            something outside the grain would delete rows the batch cannot
            replace.
        event_time_column: silver column used to decide which rows an
            incremental run must recompute. Omit to recompute the whole table
            every run, which is right for a small dimension.
        write: how the batch is applied. See VALID_WRITE_STRATEGIES.
        partition_by: Iceberg partition transforms, applied at creation only.
        output_columns: column name -> Spark type, for the columns consumers
            depend on. Deliberately an *output* contract, the opposite
            direction to TableConfig.required_columns, which is an input
            contract against raw -- hence the different name.

            Deliberately a subset, not the full schema. A gold table is a
            published interface, so renaming a column consumers read is a
            breaking change and should cost a config edit; but listing every
            column would make each additive change a two-place edit and the
            test fail on harmless additions. Declare what a dashboard would
            break without, and leave the rest to the transform.
    """

    depends_on: list[str]
    grain_keys: list[str]
    scope_keys: list[str] = field(default_factory=list)
    event_time_column: str | None = None
    write: str = "delete_insert"
    partition_by: list[str] = field(default_factory=list)
    output_columns: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.depends_on:
            raise ValueError("depends_on must not be empty")

        if not self.grain_keys:
            raise ValueError("grain_keys must not be empty")

        if self.write not in VALID_WRITE_STRATEGIES:
            raise ValueError(
                f"write {self.write!r} is not one of {sorted(VALID_WRITE_STRATEGIES)}"
            )

        if self.write == "delete_insert" and not self.scope_keys:
            raise ValueError(
                "write 'delete_insert' requires scope_keys; without them the "
                "delete would match the entire table"
            )

        outside = set(self.scope_keys) - set(self.grain_keys)
        if outside:
            raise ValueError(
                f"scope_keys not in grain_keys: {sorted(outside)}; the delete "
                f"would remove rows this batch cannot replace"
            )

        # The grain is the one thing every consumer joins on, so it cannot be
        # left out of a declared contract. Mirrors the business_keys check on
        # TableConfig above.
        undeclared = set(self.grain_keys) - set(self.output_columns)
        if self.output_columns and undeclared:
            raise ValueError(
                f"grain_keys not listed in output_columns: {sorted(undeclared)}"
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
    gold: dict[str, GoldTableConfig] = field(default_factory=dict)
    # Which Iceberg catalog backend the jobs configure, and where its warehouse
    # lives. "glue" is readable by PyIceberg as well as Spark, which is what
    # lets the dashboard share these tables; "hadoop" is Spark-only and is what
    # the test suite uses, since it needs no AWS and nothing provisioned.
    catalog_type: str = "hadoop"
    warehouse: str | None = None

    def __post_init__(self) -> None:
        if self.catalog_type not in VALID_CATALOG_TYPES:
            raise ValueError(
                f"catalog type {self.catalog_type!r} is not one of "
                f"{sorted(VALID_CATALOG_TYPES)}"
            )

        # Registering Glue tables whose data sits on a local disk would succeed
        # here and fail later, for anyone but this machine.
        if self.catalog_type == "glue" and not (
                self.warehouse or "").startswith(("s3://", "s3a://")):
            raise ValueError(
                "catalog type 'glue' requires an s3:// warehouse; got "
                f"{self.warehouse!r}"
            )

    def get_table_config(self, name: str) -> TableConfig:
        """Look up one dataset, with an error that lists the valid names."""
        try:
            return self.tables[name]
        except KeyError:
            raise KeyError(
                f"unknown dataset {name!r}; config defines {sorted(self.tables)}"
            ) from None

    def get_gold_config(self, name: str) -> GoldTableConfig:
        """Look up one gold table, with an error that lists the valid names."""
        try:
            return self.gold[name]
        except KeyError:
            raise KeyError(
                f"unknown gold table {name!r}; config defines {sorted(self.gold)}"
            ) from None

    def layer(self, name: str) -> str:
        """Resolve a layer name ("raw" -> the configured path segment)."""
        try:
            return self.layers[name]
        except KeyError:
            raise KeyError(
                f"unknown layer {name!r}; config defines {sorted(self.layers)}"
            ) from None

