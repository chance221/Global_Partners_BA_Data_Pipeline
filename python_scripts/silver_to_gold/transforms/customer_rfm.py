"""Customer segments from the trailing 90 days, split into three 30-day periods.

One row per user_id over all history, so no event_time_column: an incremental
run would segment customers on a fraction of their orders. Full rebuild each run.

The anchor is the dataset's own last order date, not batch_date. The data ends
2024-02-20 while batch_date is 2026-10-16 -- 969 days apart -- so anchoring to
the batch would make every customer Dormant and the segmentation meaningless.

The VIP spend threshold is the 80th percentile of 90-day spend among ACTIVE
customers.

Segment order matters: the cases below are evaluated top to bottom and the first
match wins. 
"""

from __future__ import annotations

from datetime import datetime, timedelta

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

ACTIVE_WINDOW_DAYS = 90  #This is shared with loyalty era profile so if you change this you change the active lookback window for that gold table as well.
PERIOD_DAYS = 30 # This is shared with loyalty era profile so if you change this you change the active lookback window for that gold table as well.

# Measured thresholds. 
NEW_MAX_TENURE_DAYS = 30        # first order this recent == no history yet
REENGAGE_DORMANCY_DAYS = 180    # gap before a fresh order counts as a win-back
LOYAL_MIN_ORDERS_90D = 4        # 80th percentile of 90-day order count is 3, so 4+ is top fifth
SLIPPING_MIN_ORDERS_90D = 3     # had a rhythm worth losing
VIP_SPEND_PERCENTILE = 0.8


def build(inputs: dict[str, DataFrame], batch_date: datetime) -> DataFrame:
    orders = inputs["order_items"]

    identified = orders.filter(
        F.col("user_id").isNotNull() & (F.trim(F.col("user_id")) != "")
    )

    per_order = (
        identified
        .groupBy("user_id", "order_id")
        .agg(
            F.min(F.to_date(F.col("creation_time_utc"))).alias("order_date"),
            F.sum("item_price").cast("decimal(18,2)").alias("order_revenue"),
        )
    )


    anchor = per_order.agg(F.max("order_date")).collect()[0][0]
    if anchor is None:

        anchor = batch_date.date()

    p1_start = anchor - timedelta(days=PERIOD_DAYS)              # last 30d
    p2_start = anchor - timedelta(days=2 * PERIOD_DAYS)          # 31-60d
    window_start = anchor - timedelta(days=ACTIVE_WINDOW_DAYS)   # 61-90d

    in_p1 = F.col("order_date") > F.lit(p1_start)
    in_p2 = (F.col("order_date") <= F.lit(p1_start)) & (F.col("order_date") > F.lit(p2_start))
    in_p3 = (F.col("order_date") <= F.lit(p2_start)) & (F.col("order_date") > F.lit(window_start))

    per_customer = (
        per_order
        .groupBy("user_id")
        .agg(
            F.max("order_date").alias("last_order_date"),
            F.min("order_date").alias("first_order_date"),
            F.count(F.lit(1)).alias("total_orders"),
            F.sum("order_revenue").cast("decimal(18,2)").alias("total_revenue"),

            F.sum(F.when(in_p1, 1).otherwise(0)).alias("orders_last_30d"),
            F.sum(F.when(in_p2, 1).otherwise(0)).alias("orders_31_60d"),
            F.sum(F.when(in_p3, 1).otherwise(0)).alias("orders_61_90d"),

            F.sum(F.when(in_p1, F.col("order_revenue")).otherwise(0))
             .cast("decimal(18,2)").alias("spend_last_30d"),
            F.sum(F.when(in_p2, F.col("order_revenue")).otherwise(0))
             .cast("decimal(18,2)").alias("spend_31_60d"),
            F.sum(F.when(in_p3, F.col("order_revenue")).otherwise(0))
             .cast("decimal(18,2)").alias("spend_61_90d"),

            
            F.min(F.when(in_p1, F.col("order_date"))).alias("_first_in_last_30d"),
            F.max(F.when(F.col("order_date") <= F.lit(window_start),
                         F.col("order_date"))).alias("_last_before_window"),
        )
        .withColumn("orders_last90d", F.col("orders_last_30d")
                    + F.col("orders_31_60d") + F.col("orders_61_90d"))
        .withColumn("spend_last90d", (F.col("spend_last_30d")
                    + F.col("spend_31_60d") + F.col("spend_61_90d"))
                    .cast("decimal(18,2)"))
        .withColumn("recency_days",
                    F.datediff(F.lit(anchor), F.col("last_order_date")))
        .withColumn("tenure_days",
                    F.datediff(F.lit(anchor), F.col("first_order_date")))
        .withColumn("is_active", F.col("orders_last90d") > 0)
        .withColumn("dormancy_days",
                    F.datediff(F.col("_first_in_last_30d"),
                               F.col("_last_before_window")))
        .withColumn("avg_gap_days",
                    F.when(F.col("total_orders") > 1,
                           F.datediff(F.col("last_order_date"),
                                      F.col("first_order_date"))
                           / (F.col("total_orders") - 1))
                     .cast("decimal(10,1)"))
    )

    # Partitioned by is_active so the threshold is the 80th percentile among
    # active customers only.
    active_spend = Window.partitionBy("is_active")

    return (
        per_customer
        .withColumn("vip_spend_cut",
                    F.percentile_approx(F.col("spend_last90d"), VIP_SPEND_PERCENTILE)
                     .over(active_spend).cast("decimal(18,2)"))
        # First match wins 
        .withColumn("segment",
                    F.when(~F.col("is_active"), "Dormant")
                     .when((F.col("tenure_days") <= F.lit(NEW_MAX_TENURE_DAYS))
                           & (F.col("orders_last_30d") >= 1), "New")
                     .when((F.col("orders_last_30d") >= 1)
                           & (F.col("orders_31_60d") == 0)
                           & (F.col("orders_61_90d") == 0)
                           & (F.col("dormancy_days") >= F.lit(REENGAGE_DORMANCY_DAYS)),
                           "Re-engaging")
                     .when((F.col("orders_last_30d") >= 2)
                           & ((F.col("orders_31_60d") + F.col("orders_61_90d")) >= 1)
                           & (F.col("spend_last90d") >= F.col("vip_spend_cut")), "VIP")
                     .when((F.col("orders_last_30d") >= 1)
                           & (F.col("orders_last90d") >= F.lit(LOYAL_MIN_ORDERS_90D)),
                           "Loyal")
                     .when(F.col("orders_last_30d") >= 1, "Promising")
                     .when(F.col("orders_last90d") >= F.lit(SLIPPING_MIN_ORDERS_90D),
                           "Slipping")
                     .otherwise("Lapsing"))
        .withColumn("segment_group",
                    F.when(F.col("segment").isin("VIP", "Loyal"), "Core")
                     .when(F.col("segment") == "New", "New Growth")
                     .when(F.col("segment").isin("Promising", "Re-engaging"),
                           "Developing")
                     .when(F.col("segment").isin("Slipping", "Lapsing"), "Churn Risk")
                     .otherwise("Dormant"))
        .withColumn("avg_order_value",
                    (F.col("total_revenue") / F.col("total_orders"))
                    .cast("decimal(10,2)"))
        .withColumn("anchor_date", F.lit(anchor))
        .withColumn("batch_date",
                    F.lit(batch_date.strftime("%Y-%m-%d")).cast("date"))
        .withColumn("updated_at", F.current_timestamp())
        .drop("_first_in_last_30d", "_last_before_window")
    )
