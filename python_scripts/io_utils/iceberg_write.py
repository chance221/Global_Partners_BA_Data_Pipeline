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
  replace        Rewrite every row. For small tables where a merge buys nothing.

INVARIANT for delete_insert, which this module cannot check: the batch carries
the *complete* set of rows for every scope value it touches. Otherwise the
delete removes rows the insert does not put back.

No strategy emits WHEN NOT MATCHED BY SOURCE THEN DELETE. That is only correct
for a complete snapshot; under an incremental window it would delete history on
the first run.
"""

from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession

VALID_WRITE_STRATEGIES = frozenset({"merge", "delete_insert", "replace"})


def ensure_namespace(spark: SparkSession, catalog: str, namespace: str) -> None:
    """Create the namespace if this is the first table ever written to it."""
    spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {catalog}.{namespace}")


def _create_table(spark: SparkSession, df: DataFrame, fqn: str,
                  partition_by: list[str]) -> None:
    
    columns = ", ".join(f"{name} {dtype}" for name, dtype in df.dtypes)
    clause = f" PARTITIONED BY ({', '.join(partition_by)})" if partition_by else ""

    spark.sql(f"CREATE TABLE {fqn} ({columns}) USING iceberg{clause}")


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
    for a batch job that is safe to re-run, and the reason the job must stay
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
    and a first run has none.

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

    first_run = not spark.catalog.tableExists(fqn)
    if first_run:
        _create_table(spark, df, fqn, partition_by or [])
        ordered = df
    else:
        # Align to the target's column order so INSERT * and SELECT * line up
        # positionally after any schema evolution.
        ordered = df.select(*spark.table(fqn).columns)

    view = view_name or f"_batch_{fqn.replace('.', '_')}"
    ordered.createOrReplaceTempView(view)

    try:
        if first_run:
            spark.sql(f"INSERT INTO {fqn} SELECT * FROM {view}")
            return f"created {fqn}"

        if strategy == "replace":
            # Keeps the table identity and snapshot history, unlike a drop and
            # recreate. For an unpartitioned table this replaces every row.
            spark.sql(f"INSERT OVERWRITE {fqn} SELECT * FROM {view}")
            return f"replaced {fqn}"

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
