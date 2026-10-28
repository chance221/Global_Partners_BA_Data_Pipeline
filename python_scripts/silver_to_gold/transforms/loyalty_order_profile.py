"""Order profile by loyalty membership state, with a control.

Five rows, told apart by (population, membership_state):

  All members             | Loyalty Member      every flagged order
  Observed in both states | Pre-member  switchers' unflagged orders
  Observed in both states | Member      switchers' flagged orders
  Never members           | Non-member  customers with no flagged order ever
  All customers           | All orders  every identified order

"""

from __future__ import annotations

from datetime import datetime
from functools import reduce

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

ALL_MEMBERS = "All members"
BOTH_STATES = "Observed in both states"
NEVER_MEMBERS = "Never members"
ALL_CUSTOMERS = "All customers"

MEMBER = "Member"
PRE_MEMBER = "Pre-member"
NON_MEMBER = "Non-member"
ALL_ORDERS = "All orders"

MEDIAN = 0.5


def _per_order(items: DataFrame) -> DataFrame:
    """One row per (user_id, order_id) with its basket and loyalty flag.
    """
    identified = items.filter(
        F.col("user_id").isNotNull() & (F.trim(F.col("user_id")) != "")
    )
    return (
        identified
        .groupBy("user_id", "order_id")
        .agg(
            F.sum("item_price").cast("decimal(18,2)").alias("order_value"),
            F.count(F.lit(1)).alias("line_items"),
            F.sum("item_quantity").alias("units"),
            F.sum(F.coalesce(F.col("is_loyalty"), F.lit(False)).cast("int"))
             .alias("_flagged_lines"),
        )
        .withColumn("is_loyalty_order", F.col("_flagged_lines") > 0)
        .drop("_flagged_lines")
    )


def _with_membership(per_order: DataFrame) -> DataFrame:
    """Tag every order with the shape of its customer's whole history.
    """
    history = (
        per_order
        .groupBy("user_id")
        .agg(
            F.max(F.col("is_loyalty_order").cast("int")).alias("_any"),
            F.min(F.col("is_loyalty_order").cast("int")).alias("_all"),
        )
        .withColumn("is_member", F.col("_any") == 1)
        # At least one order in each state, which is the only population where
        # before and after describe the same customers.
        .withColumn("is_switcher",
                    (F.col("_any") == 1) & (F.col("_all") == 0))
        .select("user_id", "is_member", "is_switcher")
    )
    return per_order.join(history, on="user_id", how="inner")


def _populations(orders: DataFrame) -> DataFrame:
    """Repeat each order once per population it belongs to
    """
    member = F.col("is_loyalty_order")
    
    rows = [
        (ALL_MEMBERS, MEMBER, member),
        (BOTH_STATES, PRE_MEMBER, F.col("is_switcher") & ~member),
        (BOTH_STATES, MEMBER, F.col("is_switcher") & member),
        (NEVER_MEMBERS, NON_MEMBER, ~F.col("is_member")),
        (ALL_CUSTOMERS, ALL_ORDERS, F.lit(True)),
    ]
    frames = [
        orders.filter(condition)
              .withColumn("population", F.lit(population))
              .withColumn("membership_state", F.lit(state))
        for population, state, condition in rows
    ]
    return reduce(DataFrame.unionByName, frames)


def build(inputs: dict[str, DataFrame], batch_date: datetime) -> DataFrame:
    orders = _with_membership(_per_order(inputs["order_items"]))

    return (
        _populations(orders)
        .groupBy("population", "membership_state")
        .agg(
            F.countDistinct("user_id").alias("customers"),
            F.count(F.lit(1)).alias("orders"),
            F.sum("order_value").cast("decimal(18,2)").alias("total_revenue"),
            F.percentile_approx("order_value", MEDIAN).cast("decimal(10,2)")
             .alias("median_order_value"),
            F.sum("line_items").alias("_line_items"),
            F.sum("units").alias("_units"),
        )
        
        .withColumn("avg_order_value",
                    (F.col("total_revenue") / F.col("orders"))
                    .cast("decimal(10,2)"))
        .withColumn("avg_line_items_per_order",
                    (F.col("_line_items") / F.col("orders"))
                    .cast("decimal(10,2)"))
        .withColumn("avg_units_per_order",
                    (F.col("_units") / F.col("orders"))
                    .cast("decimal(10,2)"))
        .drop("_line_items", "_units")
        .withColumn("batch_date",
                    F.lit(batch_date.strftime("%Y-%m-%d")).cast("date"))
        .withColumn("updated_at", F.current_timestamp())
    )
