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

    `versionAsOf` and not `snapshot-id`: Spark 4.1 removed the latter and
    raises telling you so. It accepts a snapshot id here despite the name.

    Do NOT "fix" that error by reaching for `snapshot_id` with an underscore.
    It is not a recognised option, so Iceberg ignores it silently and hands
    back the CURRENT table -- verified against this table's own manifest
    counts, where it returned 13,413 rows for all seven snapshots including
    the three that hold zero. An option that raises is safe; one that is
    ignored is how you publish a diff of a table against itself.
    """
    return (spark.read
            .option("versionAsOf", snapshot_id)
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


# --- gold, across every table ---------------------------------------------

def gold_overview(spark: SparkSession, config: AppConfig) -> pd.DataFrame:
    """One row per gold table: volume, grain integrity, write strategy.

    The counterpart to `silver_overview`, and the cell to run after a
    silver_to_gold run. Driven by `gold:` in config, so registering a new
    transform adds it here with no edit to this file.

    `rows` and `distinct_grain` must be EQUAL for every gold table. Unlike
    silver there is no write strategy where a gap is legitimate: grain_keys is
    what makes a gold row unique, so a gap means the transform emits
    duplicates and every join against the table will fan out.
    """
    rows = []

    for name, contract in config.gold.items():
        identifier = fqn(config, "gold", name)
        try:
            df = spark.table(identifier)
        except Exception as exc:
            rows.append({"table": name, "error": type(exc).__name__})
            continue

        keys = [k.lower() for k in contract.grain_keys]
        aggs = [
            F.count(F.lit(1)).alias("rows"),
            F.countDistinct(*[F.col(k) for k in keys]).alias("distinct_grain"),
        ]
        if "batch_date" in df.columns:
            aggs.append(F.max("batch_date").alias("last_batch"))
        if "updated_at" in df.columns:
            aggs.append(F.max("updated_at").alias("last_written"))

        record = df.agg(*aggs).collect()[0].asDict()
        record["grain_ok"] = record["rows"] == record["distinct_grain"]
        record.update({
            "table": name,
            "write": contract.write,
            "partitioned": ",".join(contract.partition_by) or "-",
            "grain": ",".join(keys),
        })
        rows.append(record)

    frame = pd.DataFrame(rows)
    front = [c for c in ("table", "write", "rows", "distinct_grain",
                         "grain_ok", "partitioned", "last_batch",
                         "last_written", "grain")
             if c in frame.columns]
    return frame[front + [c for c in frame.columns if c not in front]]


def output_contract(spark: SparkSession, config: AppConfig,
                    table: str | None = None) -> pd.DataFrame:
    """Declared `output_columns` against the columns each table really has.

    The gold counterpart to checking the silver casts landed. `output_columns`
    is deliberately a SUBSET contract -- a transform may emit more than it
    declares -- so this reports only on declared columns, and an extra column
    in the table is not a finding.

    `status` is the column to read. `MISSING` means a transform stopped
    emitting something a dashboard depends on; `WRONG TYPE` means a consumer
    silently gets a different type than the contract promises.
    """
    names = [table] if table else sorted(config.gold)
    rows = []

    for name in names:
        declared = config.gold[name].output_columns
        if not declared:
            continue
        try:
            actual = dict(spark.table(fqn(config, "gold", name)).dtypes)
        except Exception as exc:
            rows.append({"table": name, "column": "-", "declared": "-",
                         "actual": "-", "status": type(exc).__name__})
            continue

        for column, want in declared.items():
            got = actual.get(column.lower(), actual.get(column))
            status = "MISSING" if got is None else (
                "ok" if got == want else "WRONG TYPE")
            rows.append({"table": name, "column": column, "declared": want,
                         "actual": got, "status": status})

    return pd.DataFrame(rows)


# --- loyalty --------------------------------------------------------------

def loyalty_profile(spark: SparkSession, config: AppConfig) -> pd.DataFrame:
    """The five loyalty rows, ordered so the two comparisons read together.

    THE ROWS OVERLAP AND MUST NOT BE SUMMED. A switcher's flagged order is
    counted in three of them, so summing total_revenue over this frame
    reports roughly 2.4x the business.

    Two different comparisons live here. "Never members" against "All
    members" is a LEVEL difference and is confounded by who chooses to
    enrol. "Observed in both states", Pre-member against Member, is a CHANGE
    within the same 2,783 customers and is not.

    `orders_per_customer` is derived here rather than stored. Read it ACROSS
    populations, not between Pre-member and Member: it is a count, and those
    customers were observed un-enrolled for far longer than enrolled, so the
    Pre-member count is inflated by exposure alone. `loyalty_order_rate()`
    is the exposure-normalised version and is what answers the before/after
    question.
    """
    return spark.sql(
        "SELECT population, membership_state, customers, orders, "
        "CAST(orders / customers AS decimal(10,2)) AS orders_per_customer, "
        "total_revenue, avg_order_value, median_order_value, "
        "avg_line_items_per_order, avg_units_per_order "
        f"FROM {fqn(config, 'gold', 'loyalty_order_profile')} "
        "ORDER BY CASE population "
        "           WHEN 'All customers' THEN 1 WHEN 'Never members' THEN 2 "
        "           WHEN 'All members' THEN 3 ELSE 4 END, "
        "         membership_state DESC"
    ).toPandas()


def loyalty_cohorts(spark: SparkSession, config: AppConfig) -> pd.DataFrame:
    """Is "All members" one population, or two unlike ones averaged?

    Splits identified customers three ways: never flagged, flagged on every
    order, and switchers. The gold table's "All members" row pools the last
    two, and they behave nothing alike -- which is why the within-person rows
    exist and why "All members" is the weakest row in the table to quote.

    Computed from silver, since gold has already aggregated the per-customer
    history this needs.
    """
    items = fqn(config, "silver", "order_items")
    return spark.sql(f"""
        WITH per_order AS (
            SELECT user_id, order_id,
                   MAX(CASE WHEN COALESCE(is_loyalty, false)
                            THEN 1 ELSE 0 END) AS flagged,
                   SUM(item_price) AS order_value
            FROM {items}
            WHERE user_id IS NOT NULL AND TRIM(user_id) <> ''
            GROUP BY user_id, order_id
        ), per_customer AS (
            SELECT user_id, MAX(flagged) AS any_f, MIN(flagged) AS all_f,
                   COUNT(*) AS orders, SUM(order_value) AS revenue
            FROM per_order GROUP BY user_id
        )
        SELECT CASE WHEN any_f = 0 THEN 'never flagged'
                    WHEN all_f = 1 THEN 'flagged on every order'
                    ELSE 'switcher (both states)' END AS cohort,
               COUNT(*) AS customers,
               SUM(orders) AS orders,
               CAST(SUM(orders) / COUNT(*) AS decimal(10,2))
                   AS orders_per_customer,
               CAST(SUM(revenue) AS decimal(18,2)) AS revenue,
               CAST(SUM(revenue) / SUM(orders) AS decimal(10,2))
                   AS avg_order_value
        FROM per_customer GROUP BY 1 ORDER BY customers DESC
    """).toPandas()


def loyalty_order_rate(spark: SparkSession, config: AppConfig) -> pd.DataFrame:
    """Order frequency before and after enrolling, normalised for exposure.

    The companion to `orders_per_customer`, and the reason that column cannot
    answer the before/after question alone. Switchers are observed for far
    longer un-enrolled than enrolled, so the raw count falls after joining
    even when nothing about their behaviour changed. Dividing by observed days
    removes that, and the two rates come out flat.

    One caveat bounds where this is readable: a customer with a single order
    in a state spans one day, so that customer contributes one order per day
    and drags the pooled rate up. Switchers average 7 and 13 orders per state,
    so the distortion is small here; for a population full of one-order
    customers -- never members average 3.5 -- it would not be, which is why
    this is restricted to switchers.
    """
    items = fqn(config, "silver", "order_items")
    return spark.sql(f"""
        WITH per_order AS (
            SELECT user_id, order_id,
                   MAX(CASE WHEN COALESCE(is_loyalty, false)
                            THEN 1 ELSE 0 END) AS flagged,
                   MIN(TO_DATE(creation_time_utc)) AS order_date
            FROM {items}
            WHERE user_id IS NOT NULL AND TRIM(user_id) <> ''
            GROUP BY user_id, order_id
        ), switchers AS (
            SELECT user_id FROM per_order
            GROUP BY user_id HAVING MAX(flagged) = 1 AND MIN(flagged) = 0
        ), spans AS (
            SELECT p.user_id, p.flagged, COUNT(*) AS orders,
                   DATEDIFF(MAX(p.order_date), MIN(p.order_date)) + 1 AS days
            FROM per_order p JOIN switchers s USING (user_id)
            GROUP BY p.user_id, p.flagged
        )
        SELECT CASE WHEN flagged = 1 THEN 'Member' ELSE 'Pre-member' END
                   AS membership_state,
               COUNT(*) AS customers,
               SUM(orders) AS orders,
               SUM(days) AS observed_days,
               CAST(SUM(orders) / COUNT(*) AS decimal(10,2))
                   AS orders_per_customer,
               CAST(AVG(days) AS decimal(10,1)) AS avg_observed_days,
               CAST(SUM(orders) / (SUM(days) / 28.0) AS decimal(10,3))
                   AS orders_per_28_days
        FROM spans GROUP BY flagged ORDER BY membership_state DESC
    """).toPandas()


def loyalty_flag_ordering(spark: SparkSession,
                          config: AppConfig) -> pd.DataFrame:
    """Does "Pre-member" actually mean "before enrolling"?

    `is_loyalty` is a line-item flag, and nothing in the source states an
    enrolment date. So the before/after reading rests on an assumption: that a
    switcher's unflagged orders all precede their flagged ones. If instead the
    flag records whether loyalty was APPLIED to an order, a member who forgets
    their card lands in Pre-member and contaminates the before group.

    This is the check that settles it. `clean` customers have every unflagged
    order before their first flagged one; `INTERLEAVED` ones do not.
    """
    items = fqn(config, "silver", "order_items")
    return spark.sql(f"""
        WITH per_order AS (
            SELECT user_id, order_id,
                   MAX(CASE WHEN COALESCE(is_loyalty, false)
                            THEN 1 ELSE 0 END) AS flagged,
                   MIN(TO_DATE(creation_time_utc)) AS order_date
            FROM {items}
            WHERE user_id IS NOT NULL AND TRIM(user_id) <> ''
            GROUP BY user_id, order_id
        ), switchers AS (
            SELECT p.* FROM per_order p
            JOIN (SELECT user_id FROM per_order GROUP BY user_id
                  HAVING MAX(flagged) = 1 AND MIN(flagged) = 0) s
              USING (user_id)
        ), bounds AS (
            SELECT user_id,
                   MAX(CASE WHEN flagged = 0 THEN order_date END) AS last_pre,
                   MIN(CASE WHEN flagged = 1 THEN order_date END) AS first_flag,
                   SUM(CASE WHEN flagged = 0 THEN 1 ELSE 0 END) AS pre_orders
            FROM switchers GROUP BY user_id
        )
        SELECT CASE WHEN last_pre < first_flag
                    THEN 'clean (every unflagged order precedes the first flagged)'
                    ELSE 'INTERLEAVED (unflagged orders after the first flagged)'
               END AS verdict,
               COUNT(*) AS customers,
               SUM(pre_orders) AS pre_member_orders
        FROM bounds GROUP BY 1 ORDER BY customers DESC
    """).toPandas()


# --- loyalty: is the frequency gap a finding or the definition? -----------

# The cohort split, as SQL, because the four helpers below all need it and a
# second copy would be a second thing to keep in step. Mirrors _per_order and
# _with_membership in silver_to_gold/transforms/loyalty_order_profile.py.
_COHORTS = """
    WITH per_order AS (
        SELECT user_id, order_id,
               MAX(CASE WHEN COALESCE(is_loyalty, false) THEN 1 ELSE 0 END)
                   AS flagged,
               MIN(TO_DATE(creation_time_utc)) AS order_date,
               SUM(item_price) AS order_value
        FROM {items}
        WHERE user_id IS NOT NULL AND TRIM(user_id) <> ''
        GROUP BY user_id, order_id
    ), customers AS (
        SELECT user_id, COUNT(*) AS orders, SUM(order_value) AS revenue,
               MAX(flagged) AS any_f, MIN(flagged) AS all_f,
               CASE WHEN MAX(flagged) = 0 THEN 'never flagged'
                    WHEN MIN(flagged) = 1 THEN 'flagged on every order'
                    ELSE 'switcher' END AS cohort
        FROM per_order GROUP BY user_id
    )
"""

_ORDER_BAND = """
    CASE WHEN orders = 1 THEN '01 order'
         WHEN orders = 2 THEN '02 orders'
         WHEN orders BETWEEN 3 AND 4 THEN '03-04'
         WHEN orders BETWEEN 5 AND 9 THEN '05-09'
         WHEN orders BETWEEN 10 AND 19 THEN '10-19'
         WHEN orders BETWEEN 20 AND 49 THEN '20-49'
         ELSE '50+' END
"""


def loyalty_cohort_spread(spark: SparkSession,
                          config: AppConfig) -> pd.DataFrame:
    """Orders per customer as a DISTRIBUTION, not a mean, per cohort.

    `loyalty_cohorts()` reports switchers at ~20.6 orders per customer against
    ~3.5 for both other cohorts. This is the cell that shows why that gap
    cannot be read as a behavioural finding.

    Look at `one_order_customers`. A switcher needs one flagged AND one
    unflagged order, so the label REQUIRES at least two -- the column is 0 for
    switchers by construction, while over half of each other cohort sits at
    exactly one order. The cohorts are not comparable on frequency because
    frequency is part of what decides the cohort.
    """
    return spark.sql(
        _COHORTS.format(items=fqn(config, "silver", "order_items")) +
        """
        SELECT cohort, COUNT(*) AS customers,
               ROUND(AVG(orders), 2) AS mean_orders,
               PERCENTILE(orders, 0.25) AS p25,
               PERCENTILE(orders, 0.50) AS median,
               PERCENTILE(orders, 0.75) AS p75,
               PERCENTILE(orders, 0.90) AS p90,
               MAX(orders) AS max_orders,
               SUM(CASE WHEN orders = 1 THEN 1 ELSE 0 END)
                   AS one_order_customers
        FROM customers GROUP BY cohort ORDER BY median DESC
        """
    ).toPandas()


def loyalty_frequency_bias(spark: SparkSession,
                           config: AppConfig) -> pd.DataFrame:
    """How much of "switchers order more" is just the definition?

    Cohort share within each order-count band. `pct_switcher` is the column to
    read, and it should be read as a warning rather than a result: it climbs
    from 0% among one-order customers -- where the label is impossible -- to
    about 70% among customers with 50+ orders, where a varying flag is almost
    unavoidable.

    That monotonic climb IS the bias. Order count causes the label, so
    explaining order count with the label is circular. Any claim that
    switchers behave differently has to survive holding order count fixed,
    which is what `loyalty_matched_bands()` does.
    """
    return spark.sql(
        _COHORTS.format(items=fqn(config, "silver", "order_items")) +
        f"""
        SELECT {_ORDER_BAND} AS order_band,
               COUNT(*) AS customers,
               SUM(CASE WHEN cohort = 'switcher' THEN 1 ELSE 0 END)
                   AS switchers,
               CAST(100.0 * SUM(CASE WHEN cohort = 'switcher' THEN 1 ELSE 0 END)
                    / COUNT(*) AS decimal(5,1)) AS pct_switcher,
               CAST(100.0 * SUM(CASE WHEN cohort = 'never flagged'
                                     THEN 1 ELSE 0 END)
                    / COUNT(*) AS decimal(5,1)) AS pct_never_flagged,
               CAST(100.0 * SUM(CASE WHEN cohort = 'flagged on every order'
                                     THEN 1 ELSE 0 END)
                    / COUNT(*) AS decimal(5,1)) AS pct_always_flagged
        FROM customers GROUP BY order_band ORDER BY order_band
        """
    ).toPandas()


def loyalty_matched_bands(spark: SparkSession,
                          config: AppConfig) -> pd.DataFrame:
    """The cohorts compared with order count held fixed.

    Inside one band every cohort has roughly the same `mean_orders`, so the
    mechanical advantage is gone and frequency cannot differ -- which is the
    point. What is left is spend, and that comparison is honest.

    The robust finding here: "flagged on every order" has the lowest
    `avg_order_value` in every band, and "never flagged" the highest in most
    of them. Holding at every frequency level is what makes it a real effect
    rather than a composition artifact.

    Treat the top band with care. It spans 20 to 2,124 orders, so it is too
    wide to be a genuine match and `mean_orders` really does differ inside it.
    """
    return spark.sql(
        _COHORTS.format(items=fqn(config, "silver", "order_items")) +
        """
        SELECT CASE WHEN orders = 2 THEN '02 orders'
                    WHEN orders BETWEEN 3 AND 4 THEN '03-04'
                    WHEN orders BETWEEN 5 AND 9 THEN '05-09'
                    WHEN orders BETWEEN 10 AND 19 THEN '10-19'
                    ELSE '20+' END AS order_band,
               cohort, COUNT(*) AS customers,
               CAST(AVG(orders) AS decimal(10,2)) AS mean_orders,
               CAST(SUM(revenue) / SUM(orders) AS decimal(10,2))
                   AS avg_order_value,
               CAST(SUM(revenue) / COUNT(*) AS decimal(10,2))
                   AS revenue_per_customer
        FROM customers WHERE orders >= 2
        GROUP BY order_band, cohort ORDER BY order_band, cohort
        """
    ).toPandas()


def loyalty_first_order(spark: SparkSession,
                        config: AppConfig) -> pd.DataFrame:
    """Classify customers by their FIRST order only, then count what followed.

    The one split with no order-count dependence at all: the label is decided
    by a single order, so a customer cannot earn it by ordering more. That
    makes it the closest thing here to a fair test of "does starting on
    loyalty lead to ordering more?"

    It answers no -- the loyalty-first group averages fewer orders, which is
    the opposite of what the raw cohort gap suggests. Two groups, one
    determination each, no circularity.

    Deliberately NOT normalised per 28 days. The loyalty-first group has a
    median of one order, so most of them span a single day, and the
    exposure-normalised rate would be dominated by that degeneracy rather than
    by behaviour. See `loyalty_order_rate()` for where that denominator is
    safe and where it is not.
    """
    return spark.sql(
        _COHORTS.format(items=fqn(config, "silver", "order_items")) +
        """
        , ranked AS (
            SELECT user_id, flagged,
                   ROW_NUMBER() OVER (PARTITION BY user_id
                                      ORDER BY order_date, order_id) AS rn
            FROM per_order
        ), first_order AS (
            SELECT user_id, flagged AS first_flagged FROM ranked WHERE rn = 1
        )
        SELECT CASE WHEN f.first_flagged = 1
                    THEN 'first order WAS on loyalty'
                    ELSE 'first order was NOT on loyalty' END AS group_label,
               COUNT(*) AS customers,
               SUM(c.orders) AS orders,
               CAST(SUM(c.orders) / COUNT(*) AS decimal(10,2))
                   AS orders_per_customer,
               PERCENTILE(c.orders, 0.50) AS median_orders,
               PERCENTILE(c.orders, 0.90) AS p90_orders,
               CAST(SUM(c.revenue) / SUM(c.orders) AS decimal(10,2))
                   AS avg_order_value,
               CAST(SUM(c.revenue) / COUNT(*) AS decimal(10,2))
                   AS revenue_per_customer
        FROM first_order f JOIN customers c USING (user_id)
        GROUP BY group_label ORDER BY orders_per_customer DESC
        """
    ).toPandas()


def loyalty_null_model(spark: SparkSession, config: AppConfig) -> pd.DataFrame:
    """How big would the switcher gap be if the flag meant NOTHING?

    The decisive version of the frequency-bias argument, because it answers in
    the same units as the claim. Keep the real distribution of orders per
    customer exactly as it is, and replace the loyalty flag with a biased coin
    whose probability is the observed flagged share. The flag then has zero
    relationship to behaviour by construction.

    For a customer with n orders and per-order flag probability p:

        P(switcher | n) = 1 - p**n - (1-p)**n

    since a non-switcher is either all-flagged or none-flagged. Exact, not
    simulated, and it needs only the real histogram of n.

    The result is the point: a meaningless flag produces a LARGER frequency
    gap than the one actually observed. So the observed gap carries no
    information about the program's effect on frequency -- it is what the
    definition produces on its own.

    That the null over-predicts is itself informative. It expects more
    switchers than exist, because a real enrolment flag is sticky -- once
    someone joins they tend to stay flagged -- whereas a coin flip re-draws
    every order.
    """
    items = fqn(config, "silver", "order_items")

    per_order = f"""
        WITH per_order AS (
            SELECT user_id, order_id,
                   MAX(CASE WHEN COALESCE(is_loyalty, false)
                            THEN 1 ELSE 0 END) AS flagged
            FROM {items}
            WHERE user_id IS NOT NULL AND TRIM(user_id) <> ''
            GROUP BY user_id, order_id
        )
    """

    totals = spark.sql(
        per_order + "SELECT COUNT(*) AS orders, SUM(flagged) AS flagged "
                    "FROM per_order").collect()[0]
    p = totals["flagged"] / totals["orders"]

    histogram = spark.sql(
        per_order + "SELECT COUNT(*) AS orders FROM per_order GROUP BY user_id"
    ).toPandas()["orders"].value_counts().to_dict()

    switch_customers = switch_orders = other_customers = other_orders = 0.0
    for n, many in histogram.items():
        p_switch = 1.0 - p ** n - (1.0 - p) ** n
        switch_customers += many * p_switch
        switch_orders += many * n * p_switch
        other_customers += many * (1.0 - p_switch)
        other_orders += many * n * (1.0 - p_switch)

    observed = spark.sql(
        per_order + """
        , customers AS (
            SELECT user_id, COUNT(*) AS orders,
                   MAX(flagged) AS any_f, MIN(flagged) AS all_f
            FROM per_order GROUP BY user_id
        )
        SELECT CASE WHEN any_f = 1 AND all_f = 0 THEN 'switcher'
                    ELSE 'non-switcher' END AS cohort,
               COUNT(*) AS customers, SUM(orders) AS orders
        FROM customers GROUP BY 1
        """).toPandas().set_index("cohort")

    def rate(customers, orders):
        return round(orders / customers, 2)

    obs_switch = rate(observed.loc["switcher", "customers"],
                      observed.loc["switcher", "orders"])
    obs_other = rate(observed.loc["non-switcher", "customers"],
                     observed.loc["non-switcher", "orders"])

    return pd.DataFrame([
        {"scenario": f"NULL: flag is a coin flip (p={p:.4f})",
         "switcher_customers": round(switch_customers),
         "orders_per_switcher": round(switch_orders / switch_customers, 2),
         "orders_per_other": round(other_orders / other_customers, 2),
         "ratio": round((switch_orders / switch_customers)
                        / (other_orders / other_customers), 2)},
        {"scenario": "OBSERVED",
         "switcher_customers": int(observed.loc["switcher", "customers"]),
         "orders_per_switcher": obs_switch,
         "orders_per_other": obs_other,
         "ratio": round(obs_switch / obs_other, 2)},
    ])


# --- customer tables ------------------------------------------------------

def clv_segments(spark: SparkSession, config: AppConfig) -> pd.DataFrame:
    """customer_lifetime_value rolled up to its segments.

    20k rows, so peeking tells you nothing -- the segment distribution is
    what there is to read. `customers` against `revenue_share_pct` is the
    pair that matters: a segment holding few customers and most of the
    revenue is the finding the table exists to surface.

    `avg_order_value` is recomputed as revenue / orders, not averaged out of
    the per-customer column, for the same reason `weekly()` recomputes it.
    """
    return spark.sql(
        "SELECT clv_segment, COUNT(*) AS customers, "
        "SUM(total_orders) AS orders, "
        "SUM(total_lifetime_revenue) AS revenue, "
        "CAST(100.0 * SUM(total_lifetime_revenue) "
        "     / SUM(SUM(total_lifetime_revenue)) OVER () AS decimal(6,2)) "
        "    AS revenue_share_pct, "
        "CAST(SUM(total_lifetime_revenue) / SUM(total_orders) "
        "     AS decimal(10,2)) AS avg_order_value, "
        "MIN(total_lifetime_revenue) AS min_clv, "
        "MAX(total_lifetime_revenue) AS max_clv "
        f"FROM {fqn(config, 'gold', 'customer_lifetime_value')} "
        "GROUP BY clv_segment ORDER BY revenue DESC"
    ).toPandas()


def rfm_segments(spark: SparkSession, config: AppConfig) -> pd.DataFrame:
    """customer_rfm rolled up to segment within segment_group.

    `anchor_date` is the thing to sanity-check first, via `peek` -- the 90-day
    windows are measured from the last order in the data, not from
    batch_date, and the two are over two years apart in this dataset.
    """
    return spark.sql(
        "SELECT segment_group, segment, COUNT(*) AS customers, "
        "SUM(CASE WHEN is_active THEN 1 ELSE 0 END) AS active, "
        "CAST(AVG(recency_days) AS decimal(10,1)) AS avg_recency_days, "
        "CAST(AVG(tenure_days) AS decimal(10,1)) AS avg_tenure_days, "
        "SUM(orders_last90d) AS orders_last90d, "
        "SUM(spend_last90d) AS spend_last90d, "
        "CAST(SUM(total_revenue) / SUM(total_orders) AS decimal(10,2)) "
        "    AS avg_order_value "
        f"FROM {fqn(config, 'gold', 'customer_rfm')} "
        "GROUP BY segment_group, segment "
        "ORDER BY segment_group, customers DESC"
    ).toPandas()


# --- category tables ------------------------------------------------------

def category_grains(spark: SparkSession, config: AppConfig) -> pd.DataFrame:
    """category_period_sales: what each period_type actually holds.

    The three grains do NOT nest -- weeks straddle month boundaries -- so a
    monthly row cannot be summed from weekly ones. That is the whole reason
    they share a table rather than being derived from each other.

    `is_holiday_week` is NULL above the week grain by design, so
    null_holiday_flag should equal `rows` for Month and Season and be 0 for
    Week. Anything else means the flag leaked upward.
    """
    return spark.sql(
        "SELECT period_type, COUNT(*) AS rows, "
        "COUNT(DISTINCT period_start) AS periods, "
        "COUNT(DISTINCT item_category) AS categories, "
        "MIN(period_start) AS first_period, MAX(period_end) AS last_period, "
        "SUM(total_orders) AS orders, SUM(total_sales) AS sales, "
        "SUM(CASE WHEN is_holiday_week IS NULL THEN 1 ELSE 0 END) "
        "    AS null_holiday_flag "
        f"FROM {fqn(config, 'gold', 'category_period_sales')} "
        "GROUP BY period_type ORDER BY period_type"
    ).toPandas()


def holiday_lift(spark: SparkSession, config: AppConfig,
                 category: str | None = None) -> pd.DataFrame:
    """Holiday trading against the ordinary-day baseline, worst lift first.

    `total_sales` is NOT comparable between holidays. Pooling the Observed
    day means holiday_days varies -- 4 for most, 5 for Christmas, New Year's
    Day and Independence Day -- so a holiday with an extra day looks bigger
    for that reason alone. `avg_revenue_per_day` is the comparable figure and
    `lift_pct` against the ordinary baseline is what the table exists to show.

    TRY_DIVIDE rather than `/` because ANSI mode is on in Spark 4: a zero
    baseline would end the run instead of yielding null.

    Summed across categories by default; pass `category` for a single one.
    """
    where = ""
    if category:
        safe = category.replace("'", "''")   # "Kid's" is a real category
        where = f"WHERE item_category = '{safe}' "

    return spark.sql(
        "SELECT holiday_name, MAX(holiday_days) AS holiday_days, "
        "COUNT(*) AS categories, SUM(total_orders) AS orders, "
        "SUM(total_sales) AS total_sales, "
        "CAST(SUM(avg_revenue_per_day) AS decimal(18,2)) "
        "    AS holiday_rev_per_day, "
        "CAST(SUM(ordinary_avg_revenue_per_day) AS decimal(18,2)) "
        "    AS ordinary_rev_per_day, "
        "CAST(100.0 * TRY_DIVIDE("
        "       SUM(avg_revenue_per_day) - SUM(ordinary_avg_revenue_per_day), "
        "       SUM(ordinary_avg_revenue_per_day)) AS decimal(10,1)) "
        "    AS lift_pct "
        f"FROM {fqn(config, 'gold', 'category_holiday_sales')} {where}"
        "GROUP BY holiday_name ORDER BY lift_pct"
    ).toPandas()


def dirty_categories(spark: SparkSession, config: AppConfig) -> pd.DataFrame:
    """item_category values that arrived corrupted from the source.

    Three values in this dataset have a scraped admin URL concatenated into
    them, and one has it spliced into the MIDDLE of the name
    ("Drip C<url>offee"). That matters beyond being ugly: it SPLITS a real
    category in two, so the clean row's totals are understated and both
    category gold tables carry the fragment as a separate row.

    Empty means the source was cleaned. Until then, it is the caveat that
    belongs beside any category ranking.
    """
    return spark.sql(
        "SELECT item_category, LENGTH(item_category) AS name_length, "
        "COUNT(*) AS line_items, COUNT(DISTINCT order_id) AS orders "
        f"FROM {fqn(config, 'silver', 'order_items')} "
        "WHERE item_category LIKE '%http%' "
        "GROUP BY item_category ORDER BY line_items DESC"
    ).toPandas()
