"""Apply a batch to an Iceberg table. Layer-agnostic.

Silver and gold need the same three strategies for the same reasons, so they
live here once rather than in each layer. The caller supplies the identifier and
the keys; nothing in this module knows which layer it is serving.

  merge          Upsert on `merge_keys`. Requires them to be unique within the
                 batch -- Iceberg raises if two source rows match one target
                 row. Callers are expected to have removed duplicates already.
  delete_insert  Delete every target row whose `scope_keys` appear in the batch,
                 then insert the batch. For a child table with no key of its
                 own, and for a recomputed aggregate, where the batch holds the
                 complete result for each scope value it covers.
  replace        Rewrite the whole table in one atomic commit, schema included.
                 For small tables where a merge buys nothing, and the only
                 strategy that can absorb a change to the transform's columns.

INVARIANT for delete_insert, which this module cannot check: the batch carries
the *complete* set of rows for every scope value it touches. Otherwise the
delete removes rows the insert does not put back.

No strategy emits WHEN NOT MATCHED BY SOURCE THEN DELETE. That is only correct
for a complete snapshot; under an incremental window it would delete history on
the first run.
"""

from __future__ import annotations

import re

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import partitioning

VALID_WRITE_STRATEGIES = frozenset({"merge", "delete_insert", "replace"})

# Iceberg partition transforms that have a DSv2 equivalent. CREATE TABLE takes
# a transform as SQL text, but writeTo().partitionedBy() needs a Column, so a
# "replace" write has to map the one to the other to keep the partition spec.
# From `partitioning`, not the F aliases, which Spark 4.0 deprecated.
_PARTITION_TRANSFORMS = {
    "years": partitioning.years,
    "months": partitioning.months,
    "days": partitioning.days,
    "hours": partitioning.hours,
}


def ensure_namespace(spark: SparkSession, catalog: str, namespace: str) -> None:
    """
    Create the namespace if this is the first table ever written to it.
    """
    spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {catalog}.{namespace}")


def _create_table(spark: SparkSession, df: DataFrame, fqn: str,
                  partition_by: list[str]) -> None:
    
    columns = ", ".join(f"{name} {dtype}" for name, dtype in df.dtypes)
    clause = f" PARTITIONED BY ({', '.join(partition_by)})" if partition_by else ""

    spark.sql(f"CREATE TABLE {fqn} ({columns}) USING iceberg{clause}")


def _partition_columns(partition_by: list[str]) -> list:
    """Turn the config's partition strings into DSv2 partition Columns.

    `_create_table` passes "months(creation_time_utc)" straight through as SQL
    text; `partitionedBy` cannot take text, so the transform has to be resolved
    to the matching function. Raising here is deliberate: silently returning
    fewer columns would drop the partition spec on the next replace.
    """
    columns = []
    for spec in partition_by:
        text = spec.strip()
        if re.fullmatch(r"[\w.]+", text):
            columns.append(F.col(text))           # a bare column name
            continue

        # Anything else must be a transform this module can rebuild. Matching
        # loosely and falling back to F.col would turn an unsupported spec into
        # a column name that does not exist, which is the failure this guards.
        match = re.fullmatch(r"(\w+)\s*\(\s*([\w.]+)\s*\)", text)
        name = match.group(1).lower() if match else None
        if name not in _PARTITION_TRANSFORMS:
            raise ValueError(
                f"partition transform {spec!r} has no DSv2 equivalent, so a "
                f"'replace' write cannot preserve it; supported are "
                f"{sorted(_PARTITION_TRANSFORMS)} and a bare column name"
            )
        columns.append(_PARTITION_TRANSFORMS[name](match.group(2)))
    return columns


def _replace(df: DataFrame, fqn: str, partition_by: list[str]) -> None:
    """Rewrite the table -- rows, schema and partition spec -- in one commit.

    Deliberately not INSERT OVERWRITE. Overwrite can only write the columns the
    target already has, so a transform that adds, drops or renames one needs
    ALTER TABLE first, and Spark SQL commits each ALTER separately: a crash part
    way through leaves a half-migrated table. createOrReplace changes data and
    schema in a single Iceberg commit instead.

    It also keeps the snapshot history, so a bad run stays revertible by time
    travel -- time travel to a pre-change snapshot even reads columns that have
    since been dropped, because snapshots carry their own schema-id. A DROP and
    recreate would not: dropping the table discards the metadata tree that
    indexes every snapshot, which is the very thing reverting needs.

    Creates the table when it is absent, so `replace` needs no first-run path.
    """
    writer = df.writeTo(fqn)
    if partition_by:
        columns = _partition_columns(partition_by)
        writer = writer.partitionedBy(columns[0], *columns[1:])
    writer.createOrReplace()


def _merge(spark: SparkSession, fqn: str, view: str, keys: list[str]) -> None:
    on_clause = " AND ".join(f"t.{k.lower()} = s.{k.lower()}" for k in keys)

    spark.sql(f"""
        MERGE INTO {fqn} t
        USING {view} s
        ON {on_clause}
        WHEN MATCHED THEN UPDATE SET *
        WHEN NOT MATCHED THEN INSERT *
    """)


def _delete_insert(spark: SparkSession, fqn: str, view: str,
                   scope_keys: list[str]) -> None:
    """Replace every target row covered by the batch's scope keys.

    The two statements are separate Iceberg transactions, so a crash between
    them leaves those scope values empty until the run is repeated. Acceptable
    for a batch job that is safe to re-run, and the reason this job must stay
    idempotent.
    """
    lowered = [k.lower() for k in scope_keys]
    projection = ", ".join(lowered)

    # Single-column IN takes a bare column; multi-column needs the tuple form.
    # Valid in Spark SQL, unlike T-SQL which has no row value constructor.
    target = projection if len(lowered) == 1 else f"({projection})"

    spark.sql(f"""
        DELETE FROM {fqn}
        WHERE {target} IN (SELECT DISTINCT {projection} FROM {view})
    """)
    spark.sql(f"INSERT INTO {fqn} SELECT * FROM {view}")


def write_iceberg(spark: SparkSession, df: DataFrame, fqn: str, strategy: str,
                  merge_keys: list[str] | None = None,
                  scope_keys: list[str] | None = None,
                  partition_by: list[str] | None = None,
                  view_name: str | None = None) -> str:
    """Apply `df` to the Iceberg table `fqn`.

    Creates the table on first run, because MERGE and DELETE both need a target
    and a first run has none. "replace" creates it as part of its own single
    commit and so never takes that path.

    Args:
        fqn: three-part identifier, catalog.namespace.table.
        strategy: one of VALID_WRITE_STRATEGIES.
        merge_keys: required for "merge".
        scope_keys: required for "delete_insert".
        view_name: temp view name; defaults to one derived from the table.

    Returns:
        A short description of what happened, for the run log.
    """
    if strategy not in VALID_WRITE_STRATEGIES:
        raise ValueError(
            f"strategy {strategy!r} is not one of {sorted(VALID_WRITE_STRATEGIES)}"
        )

    if strategy == "replace":
        existed = spark.catalog.tableExists(fqn)
        _replace(df, fqn, partition_by or [])
        return f"{'replaced' if existed else 'created'} {fqn}"

    first_run = not spark.catalog.tableExists(fqn)
    if first_run:
        _create_table(spark, df, fqn, partition_by or [])
        ordered = df
    else:
        # Align to the target's column order so INSERT * and SELECT * line up
        # positionally. MERGE and DELETE write into the schema that is already
        # there, so a transform whose columns have changed raises here rather
        # than writing values into the wrong columns.
        ordered = df.select(*spark.table(fqn).columns)

    view = view_name or f"_batch_{fqn.replace('.', '_')}"
    ordered.createOrReplaceTempView(view)

    try:
        if first_run:
            spark.sql(f"INSERT INTO {fqn} SELECT * FROM {view}")
            return f"created {fqn}"

        if strategy == "merge":
            if not merge_keys:
                raise ValueError(f"{fqn}: strategy 'merge' requires merge_keys")
            _merge(spark, fqn, view, merge_keys)
            return f"merged into {fqn}"

        if not scope_keys:
            raise ValueError(
                f"{fqn}: strategy 'delete_insert' requires scope_keys; without "
                f"them the delete would match the entire table"
            )
        _delete_insert(spark, fqn, view, scope_keys)
        return f"delete-inserted into {fqn}"
    finally:
        spark.catalog.dropTempView(view)
