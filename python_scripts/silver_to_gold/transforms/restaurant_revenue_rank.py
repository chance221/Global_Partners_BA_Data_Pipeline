"""Restaurants ranked by revenue across all history.

Separate from `daily_restaurant_sales` because the grain is different -- one
row per restaurant, not one per restaurant per day -- and because an all-time
rank cannot be computed incrementally.

Why it has to be its own table
------------------------------
The incremental read scopes silver to the event dates a batch touched, so a
transform running incrementally sees those dates and nothing else. A rank over
all history computed from that slice would rank restaurants on a few days of
revenue and then overwrite the real answer. Equally, a rank is not additive:
you cannot merge last run's ranks with this run's.

So this table declares no `event_time_column`, which makes every run a full
rebuild -- `read_silver_inputs` hands back the dependency whole and the
aggregation sees all of silver. With `write: replace` the table is rewritten
end to end, so a restaurant that stops trading cannot linger with a stale
rank the way `delete_insert` would leave it.

That is affordable because the output is tiny (one row per restaurant) and the
input is a single pass over order_items.
"""

from __future__ import annotations

from datetime import datetime

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F


def build(inputs: dict[str, DataFrame], batch_date: datetime) -> DataFrame:
    """One row per restaurant: lifetime revenue, orders, and rank.

    item_price is the fully loaded line total -- options and quantity are
    already in it -- so revenue is a plain sum. See daily_restaurant_sales for
    the evidence behind that.
    """
    items = inputs["order_items"]

    totals = (
        items
        .groupBy("restaurant_id")
        .agg(
            # countDistinct, not count: one row per line item, so counting rows
            # would count line items and call them orders.
            F.countDistinct("order_id").alias("total_orders"),
            F.count(F.lit(1)).alias("line_items"),
            F.sum("item_quantity").alias("units"),
            F.sum("item_price").cast("decimal(18,2)").alias("total_revenue"),
        )
    )

    # No partitionBy, deliberately: a global rank is one ordering over every
    # restaurant, so Spark will warn that it is moving all data to a single
    # partition. That is correct here and must not be "fixed" by adding a
    # partition key -- doing so would rank within each key instead. It is also
    # cheap, because the window runs over the aggregated rows (one per
    # restaurant), not over order_items.
    #
    # rank(), not row_number(): two restaurants on identical revenue should
    # share a position rather than be ordered arbitrarily.
    overall = Window.orderBy(F.col("total_revenue").desc())

    return (
        totals
        # A ratio of two aggregates, so it comes after the agg -- an aggregate
        # function cannot reference another aggregate's alias.
        .withColumn("avg_order_total",
                    (F.col("total_revenue") / F.col("total_orders"))
                    .cast("decimal(10,2)"))
        .withColumn("revenue_rank", F.rank().over(overall))
        .withColumn("batch_date",
                    F.lit(batch_date.strftime("%Y-%m-%d")).cast("date"))
        .withColumn("updated_at", F.current_timestamp())
    )
