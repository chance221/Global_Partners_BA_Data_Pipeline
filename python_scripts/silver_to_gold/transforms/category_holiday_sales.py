"""Holiday sales per menu category, against an ordinary-day baseline.
"""

from __future__ import annotations

from datetime import datetime

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F

# Imported rather than redeclared, so the two category tables cannot disagree
from python_scripts.silver_to_gold.transforms.category_period_sales import (
    NON_CATEGORIES, category_materiality,
)

OBSERVED_SUFFIX = " Observed$"


def _pooled_holiday_name(name: Column) -> Column:
    """The parent holiday's name, with any "Observed" suffix removed."""
    return F.regexp_replace(name, OBSERVED_SUFFIX, "")


def build(inputs: dict[str, DataFrame], batch_date: datetime) -> DataFrame:
    items = inputs["order_items"]
    calendar = inputs["date_dim"]

    # Coalesced for the same reason as in category_period_sales: `~NULL` is
    # NULL and a filter drops it, so a missing category would lose rows.
    is_non_category = F.coalesce(
        F.col("item_category").isin(*NON_CATEGORIES), F.lit(False))

    daily = (items
             .filter(~is_non_category)
             .withColumn("sales_date",
                         F.to_date(F.col("creation_time_utc"))))

    span = daily.agg(F.min("sales_date"), F.max("sales_date")).collect()[0]
    span_start, span_end = span[0], span[1]
    if span_start is None:
        span_start = span_end = batch_date.date()

    covered = calendar.filter(
        (F.col("date_key") >= F.lit(span_start))
        & (F.col("date_key") <= F.lit(span_end))
    ).withColumn("holiday_name", _pooled_holiday_name(F.col("holiday_name")))

    classified = daily.join(
        covered.select(F.col("date_key").alias("sales_date"),
                       "is_holiday", "holiday_name"),
        on="sales_date",
        how="inner",
    )

    per_holiday = (
        classified.filter(F.col("is_holiday"))
        .groupBy("holiday_name", "item_category")
        .agg(
            # Non-additive across categories, as in category_period_sales.
            F.countDistinct("order_id").alias("total_orders"),
            F.count(F.lit(1)).alias("total_order_items"),
            F.sum("item_price").cast("decimal(18,2)").alias("total_sales"),
        )
    )

    holiday_days = (
        covered.filter(F.col("is_holiday"))
        .groupBy("holiday_name")
        .agg(F.countDistinct("date_key").cast("int").alias("holiday_days"))
    )

    ordinary_days = (
        covered.filter(~F.col("is_holiday"))
        .select(F.countDistinct("date_key"))
        .collect()[0][0]
    )

    ordinary_revenue = F.sum("item_price").cast("decimal(18,2)")
    baseline = (
        classified.filter(~F.col("is_holiday"))
        .groupBy("item_category")
        .agg(ordinary_revenue.alias("_ordinary_sales"),
             F.countDistinct("order_id").alias("_ordinary_orders"))
    )
    if ordinary_days:
        baseline = baseline.withColumn(
            "ordinary_avg_revenue_per_day",
            (F.col("_ordinary_sales") / F.lit(ordinary_days))
            .cast("decimal(18,2)"))
    else:
        baseline = baseline.withColumn(
            "ordinary_avg_revenue_per_day",
            F.lit(None).cast("decimal(18,2)"))

    return (
        per_holiday
        .join(holiday_days, on="holiday_name", how="left")
        .join(baseline.drop("_ordinary_orders"), on="item_category", how="left")
        # The same all-time share category_period_sales publishes, so the two
        # tables cannot disagree about what is material. It is a property of
        # the category, not of the holiday, so it repeats down each category's
        # rows -- and is computed from `daily`, every day, not just holidays.
        .join(category_materiality(daily), on="item_category", how="left")
        .withColumn("ordinary_days", F.lit(ordinary_days).cast("int"))
        .withColumn("avg_sales_per_order",
                    (F.col("total_sales") / F.col("total_orders"))
                    .cast("decimal(18,2)"))
        .withColumn("avg_order_items",
                    (F.col("total_order_items") / F.col("total_orders"))
                    .cast("decimal(10,2)"))
        .withColumn("avg_revenue_per_day",
                    (F.col("total_sales") / F.col("holiday_days"))
                    .cast("decimal(18,2)"))
        .drop("_ordinary_sales")
        .withColumn("batch_date",
                    F.lit(batch_date.strftime("%Y-%m-%d")).cast("date"))
        .withColumn("updated_at", F.current_timestamp())
    )
