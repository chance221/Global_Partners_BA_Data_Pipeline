"""
Lifetime value per customer, with a High/Medium/Low segment.
"""

from __future__ import annotations

from datetime import datetime

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F


def build(inputs: dict[str, DataFrame], batch_date: datetime) -> DataFrame:
    orders = inputs["order_items"]

    filtered_orders = orders.filter(
        F.col("user_id").isNotNull() & (F.trim(F.col("user_id")) != "")
    )

    per_customer = (
        filtered_orders
        .groupBy("user_id")
        .agg(
            F.sum("item_price").cast("decimal(18,2)").alias("total_lifetime_revenue"),
        
            F.countDistinct("order_id").alias("total_orders"),
            F.count(F.lit(1)).alias("line_items"),
            F.sum("item_quantity").alias("units"),
            F.min(F.to_date(F.col("creation_time_utc"))).alias("first_order_date"),
            F.max(F.to_date(F.col("creation_time_utc"))).alias("last_order_date"),
        )
    )

  
    by_value = Window.orderBy(F.col("total_lifetime_revenue").desc())

    return (
        per_customer
        .withColumn("value_percentile", F.percent_rank().over(by_value))
        
        .withColumn("clv_segment",
                    F.when(F.col("value_percentile") <= 0.2, "High")
                     .when(F.col("value_percentile") >= 0.8, "Low")
                     .otherwise("Medium"))
        .withColumn("avg_order_value",
                    (F.col("total_lifetime_revenue") / F.col("total_orders"))
                    .cast("decimal(10,2)"))
        .withColumn("batch_date",
                    F.lit(batch_date.strftime("%Y-%m-%d")).cast("date"))
        .withColumn("updated_at", F.current_timestamp())
    )
