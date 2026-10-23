"""Read silver tables as input to a gold transform.

Gold is incremental by *recomputing* affected grains rather than accumulating
into them. Accumulating only works for additive measures; a distinct count or
an average over changed rows cannot be merged into an existing aggregate.

So a run does three things:
  1. find which event dates this batch touched, from silver.batch_date
  2. read silver filtered to those dates, across ALL batch_dates, so each
     recomputed grain is complete rather than partial
  3. hand the transform a complete slice, and replace those grains in gold

Step 2 is the subtle one. Filtering on batch_date alone would give only the
rows this run loaded, and an aggregate built from those would overwrite a
complete grain with a partial one.
"""

from __future__ import annotations

from datetime import datetime

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from python_scripts.common.models.config_models import AppConfig, GoldTableConfig
from python_scripts.spark.spark_session import ICEBERG_CATALOG

EVENT_DATE = "_event_date"


def silver_table_name(appConfig: AppConfig, table: str,
                      catalog: str = ICEBERG_CATALOG) -> str:
    """Three-part Iceberg identifier for a silver table."""
    return f"{catalog}.{appConfig.layer('silver')}.{table}"


def read_silver(spark: SparkSession, appConfig: AppConfig, table: str,
                catalog: str = ICEBERG_CATALOG) -> DataFrame:
    """Read a whole silver table."""
    return spark.table(silver_table_name(appConfig, table, catalog))


def silver_table_exists(spark: SparkSession, appConfig: AppConfig, table: str,
                        catalog: str = ICEBERG_CATALOG) -> bool:
    """Whether a silver table has been created yet."""
    return spark.catalog.tableExists(silver_table_name(appConfig, table, catalog))


def changed_event_dates(spark: SparkSession, appConfig: AppConfig,
                        goldConfig: GoldTableConfig, batch_date: datetime,
                        catalog: str = ICEBERG_CATALOG) -> DataFrame:
    """Distinct event dates this batch touched, as a one-column DataFrame.

    Unions across every dependency carrying the event time column, so a change
    in any input brings its dates into scope. Returned as a DataFrame rather
    than a Python list so the filter stays a join instead of a large IN clause.
    """
    column = goldConfig.event_time_column
    frames = []

    for table in goldConfig.depends_on:
        df = read_silver(spark, appConfig, table, catalog)
        if column not in df.columns:
            continue
        frames.append(
            df.filter(F.col("batch_date") == F.lit(batch_date.strftime("%Y-%m-%d")).cast("date"))
              .select(F.to_date(F.col(column)).alias(EVENT_DATE))
              .distinct()
        )

    if not frames:
        raise ValueError(
            f"none of {goldConfig.depends_on} has event_time_column "
            f"{column!r}; it cannot scope an incremental read"
        )

    scoped = frames[0]
    for frame in frames[1:]:
        scoped = scoped.unionByName(frame)

    return scoped.distinct()


def read_silver_inputs(spark: SparkSession, appConfig: AppConfig,
                       goldConfig: GoldTableConfig, batch_date: datetime,
                       full: bool = False,
                       catalog: str = ICEBERG_CATALOG) -> dict[str, DataFrame]:
    """Return {table: DataFrame} for a gold transform.

    With `full`, or for a gold table declaring no event_time_column, every
    dependency is returned whole and the transform rebuilds the table.

    Otherwise dependencies carrying the event time column are narrowed to the
    dates this batch touched. Dependencies without it are returned whole --
    order_item_options has no timestamp of its own, and a join to a filtered
    order_items narrows it anyway.
    """
    if full or not goldConfig.event_time_column:
        return {t: read_silver(spark, appConfig, t, catalog)
                for t in goldConfig.depends_on}

    dates = changed_event_dates(spark, appConfig, goldConfig, batch_date, catalog)
    column = goldConfig.event_time_column
    inputs: dict[str, DataFrame] = {}

    for table in goldConfig.depends_on:
        df = read_silver(spark, appConfig, table, catalog)
        if column in df.columns:
            df = (
                df.withColumn(EVENT_DATE, F.to_date(F.col(column)))
                  .join(dates, on=EVENT_DATE, how="left_semi")
                  .drop(EVENT_DATE)
            )
        inputs[table] = df

    return inputs
