"""Query the Iceberg layers -- silver and gold -- through Spark.

Companion to `review_layers.py`, which reads raw and quarantine with pandas.
The split is not stylistic. An Iceberg table is a set of parquet files *plus
metadata deciding which of them are live*, so reading the parquet directly
would show rows that a later snapshot deleted. Both `merge` and
`delete_insert` rewrite files, so for silver and gold that is not academic --
pandas would show stale and duplicated rows and give no hint that it had.

Spark is therefore the authoritative reader. Starting a session costs roughly
half a minute (JVM boot, then Ivy resolving the Iceberg jars), which is the
argument for using this from a notebook: the kernel holds one session across
every cell, so you pay once and then query interactively.

Every helper returns a pandas DataFrame so it renders as a table in a
notebook, and every one builds its identifier from the loaded config rather
than a string literal, so this cannot drift from where the jobs actually write.

Read-only by design: nothing here writes, deletes, or expires a snapshot.
"""

from __future__ import annotations

import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from python_scripts.common.config_loader import load_config
from python_scripts.common.models.config_models import AppConfig
from python_scripts.spark.spark_session import ICEBERG_CATALOG, create_spark_session


def connect(env: str = "local") -> tuple[SparkSession, AppConfig]:
    """Start Spark and load the config.

    Call this once per kernel. Re-running it is harmless -- `getOrCreate`
    returns the existing session -- but it will not pick up a changed
    ICEBERG_WAREHOUSE, because the warehouse path is fixed when the JVM starts.
    """
    config = load_config(env)
    spark, _protocol = create_spark_session(config)
    spark.sparkContext.setLogLevel("ERROR")
    return spark, config


def fqn(config: AppConfig, layer: str, table: str) -> str:
    """Three-part Iceberg identifier for a table in a layer."""
    return f"{ICEBERG_CATALOG}.{config.layer(layer)}.{table}"


# --- what exists ----------------------------------------------------------

def namespaces(spark: SparkSession) -> pd.DataFrame:
    """Every namespace in the catalog -- silver, gold, whatever else exists."""
    return spark.sql(f"SHOW NAMESPACES IN {ICEBERG_CATALOG}").toPandas()


def tables(spark: SparkSession, config: AppConfig, layer: str) -> pd.DataFrame:
    """Table names in one layer. Empty means no job has written there yet."""
    namespace = f"{ICEBERG_CATALOG}.{config.layer(layer)}"
    try:
        return spark.sql(f"SHOW TABLES IN {namespace}").toPandas()
    except Exception:
        # The namespace itself does not exist until the first write.
        return pd.DataFrame(columns=["namespace", "tableName", "isTemporary"])


def inventory(spark: SparkSession, config: AppConfig, layer: str) -> pd.DataFrame:
    """Rows, files, bytes and last-written time for every table in a layer.

    Start here after a run. Read entirely from Iceberg metadata -- the row
    counts come from the manifests, so nothing scans the data.

    A table present with 0 rows means the job created it and wrote an empty
    frame, which is a different failure from the job never running.
    """
    rows = []
    for name in tables(spark, config, layer).get("tableName", []):
        identifier = fqn(config, layer, name)
        record = {"table": name}

        try:
            counts = spark.sql(
                "SELECT count(*) AS files, sum(record_count) AS rows, "
                "sum(file_size_in_bytes) AS bytes "
                f"FROM {identifier}.files"
            ).collect()[0]
            record["rows"] = counts["rows"] or 0
            record["files"] = counts["files"] or 0
            record["bytes"] = counts["bytes"] or 0
        except Exception as exc:
            record["rows"] = record["files"] = record["bytes"] = None
            record["error"] = type(exc).__name__

        try:
            snaps = spark.sql(
                "SELECT count(*) AS snapshots, max(committed_at) AS last_write "
                f"FROM {identifier}.snapshots"
            ).collect()[0]
            record["snapshots"] = snaps["snapshots"]
            record["last_write"] = snaps["last_write"]
        except Exception:
            pass

        rows.append(record)

    return pd.DataFrame(rows)


# --- one table ------------------------------------------------------------

def schema(spark: SparkSession, config: AppConfig, layer: str,
           table: str) -> pd.DataFrame:
    """Column names and types, as Iceberg stores them."""
    return spark.sql(f"DESCRIBE TABLE {fqn(config, layer, table)}").toPandas()


def peek(spark: SparkSession, config: AppConfig, layer: str, table: str,
         limit: int = 20, order_by: str | None = None) -> pd.DataFrame:
    """A few rows. `limit` is a guard -- toPandas pulls into this process."""
    df = spark.table(fqn(config, layer, table))
    if order_by:
        df = df.orderBy(order_by)
    return df.limit(limit).toPandas()


def snapshots(spark: SparkSession, config: AppConfig, layer: str,
              table: str) -> pd.DataFrame:
    """Every write to the table, oldest first.

    The `operation` column is the useful one: `append`, `overwrite` or
    `delete`. It is how you confirm a delete_insert actually deleted, or that
    a `replace` rewrote rather than appended.
    """
    return spark.sql(
        "SELECT committed_at, snapshot_id, parent_id, operation, summary "
        f"FROM {fqn(config, layer, table)}.snapshots ORDER BY committed_at"
    ).toPandas()


def partitions(spark: SparkSession, config: AppConfig, layer: str,
               table: str) -> pd.DataFrame:
    """Rows and files per partition -- the small-file question, answered.

    One row for the whole table if it is unpartitioned.
    """
    return spark.sql(
        f"SELECT * FROM {fqn(config, layer, table)}.partitions").toPandas()


def files(spark: SparkSession, config: AppConfig, layer: str, table: str,
          limit: int = 100) -> pd.DataFrame:
    """Individual live data files, largest first."""
    return spark.sql(
        "SELECT file_path, record_count, file_size_in_bytes, partition "
        f"FROM {fqn(config, layer, table)}.files "
        f"ORDER BY file_size_in_bytes DESC LIMIT {int(limit)}"
    ).toPandas()


def as_of(spark: SparkSession, config: AppConfig, layer: str, table: str,
          snapshot_id: int) -> DataFrame:
    """The table as one snapshot left it -- a lazy DataFrame, not pandas.

    Take the id from `snapshots()`. Useful for diffing gold before and after a
    transform change rather than guessing what moved.
    """
    return (spark.read
            .option("snapshot-id", snapshot_id)
            .table(fqn(config, layer, table)))


def sql(spark: SparkSession, query: str, limit: int = 200) -> pd.DataFrame:
    """Run any query, capped. Identifiers look like `iceberg.gold.<table>`."""
    return spark.sql(query).limit(limit).toPandas()


# --- silver ----------------------------------------------------------------

def silver_overview(spark: SparkSession, config: AppConfig) -> pd.DataFrame:
    """One row per silver table: volume, batch coverage and key uniqueness.

    The query to run after a load. Driven by the table contracts in config
    rather than hardcoded column lists, so adding a dataset to `tables:` adds
    it here with no change.

    `rows` vs `distinct_keys` is the interesting column pair, and what it means
    depends on the write strategy:

      merge          they must be equal. Iceberg refuses a MERGE where two
                     source rows match one target row, so a gap means rows got
                     quarantined rather than merged.
      delete_insert  rows may exceed distinct_keys legitimately -- order_item_options
                     has no unique key of its own, which is why it is scoped by
                     ORDER_ID instead.
      replace        equal, since the whole table is rewritten each run.
    """
    rows = []

    for name, contract in config.tables.items():
        identifier = fqn(config, "silver", name)
        try:
            df = spark.table(identifier)
        except Exception as exc:
            rows.append({"table": name, "error": type(exc).__name__})
            continue

        keys = [k.lower() for k in contract.business_keys]
        aggs = [
            F.count(F.lit(1)).alias("rows"),
            F.countDistinct(*[F.col(k) for k in keys]).alias("distinct_keys"),
            F.countDistinct("batch_date").alias("batches"),
            F.max("batch_date").alias("last_batch"),
        ]

        # Only order_items carries event time; the others have no such column.
        event_col = "creation_time_utc"
        if event_col in df.columns:
            aggs += [F.min(F.to_date(F.col(event_col))).alias("first_event"),
                     F.max(F.to_date(F.col(event_col))).alias("last_event")]

        record = df.agg(*aggs).collect()[0].asDict()
        record.update({"table": name, "write": contract.silver_write,
                       "keys": ",".join(keys)})
        rows.append(record)

    frame = pd.DataFrame(rows)
    front = [c for c in ("table", "write", "rows", "distinct_keys", "batches",
                         "last_batch", "first_event", "last_event", "keys")
             if c in frame.columns]
    return frame[front + [c for c in frame.columns if c not in front]]


def orphan_options(spark: SparkSession, config: AppConfig) -> pd.DataFrame:
    """Option rows whose parent line item is missing from order_items.

    Should be empty. The extract widens to whole parents for delete_insert
    tables, so an option can only arrive with its siblings -- a non-empty
    result means that widening failed or the two tables were loaded for
    different periods.
    """
    return spark.sql(
        "SELECT o.order_id, o.lineitem_id, count(*) AS option_rows "
        f"FROM {fqn(config, 'silver', 'order_item_options')} o "
        f"LEFT ANTI JOIN {fqn(config, 'silver', 'order_items')} i "
        "       ON i.order_id = o.order_id AND i.lineitem_id = o.lineitem_id "
        "GROUP BY o.order_id, o.lineitem_id"
    ).limit(50).toPandas()


# --- the measures the dashboard needs -------------------------------------

def weekly(spark: SparkSession, config: AppConfig,
           table: str = "daily_restaurant_sales") -> pd.DataFrame:
    """Roll the daily gold table up to weeks, correctly.

    `total_orders` is safe to SUM: it is a distinct count within a day, and an
    order has one creation timestamp, so the daily sets are disjoint and the
    sums cannot double-count.

    `avg_order_total` is NOT safe to average -- a mean of means weights a
    2-order Tuesday like a 200-order Saturday. It is recomputed as the ratio.
    """
    return spark.sql(
        "SELECT week_start, restaurant_id, "
        "sum(total_orders) AS total_orders, "
        "sum(total_revenue) AS total_revenue, "
        "CAST(sum(total_revenue) / sum(total_orders) AS decimal(10,2)) "
        "    AS avg_order_total "
        f"FROM {fqn(config, 'gold', table)} "
        "GROUP BY week_start, restaurant_id "
        "ORDER BY week_start, total_revenue DESC"
    ).toPandas()


def leaderboard(spark: SparkSession, config: AppConfig,
                table: str = "restaurant_revenue_rank") -> pd.DataFrame:
    """Restaurants by lifetime revenue, best first."""
    return spark.sql(
        "SELECT revenue_rank, restaurant_id, total_revenue, total_orders, "
        "line_items, units, avg_order_total "
        f"FROM {fqn(config, 'gold', table)} ORDER BY revenue_rank"
    ).toPandas()
