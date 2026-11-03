"""Lifetime spend profiled across the three spend tiers: Top, Mid, Bottom.

Grain is (clv_tier, outliers): three tiers x two populations, six rows. The
tiers are a clean partition of the customer base WITHIN a population, so
`SUM(customers)` is only meaningful once `outliers` is pinned to one value --
the same reason there is no "All customers" rollup tier. A reader who groups
by tier alone and forgets `outliers` double-counts, which is the one trap this
grain introduces and the reason `pct_of_customers` is published per row.

TWO POPULATIONS, ONE TIER DEFINITION
------------------------------------
  * `outliers = 'included'` -- every identified customer. The number of record.
  * `outliers = 'excluded'` -- the same tiers with the three volume accounts
    left out of the aggregates.

`clv_history` holds the measurements and the reasoning; the short version is
that three accounts place 1.4-2.4 orders a day for over a year and read as
corporate or aggregator traffic rather than people. Excluding them moves
lifetime revenue -3.66% and the Top tier's spend share 76.59% -> 75.70%, and
leaves every tier boundary exactly where it was. The only median that moves is
the Top tier's own, by one cent ($158.24 -> $158.23), which is
`percentile_approx` working over three fewer rows rather than a real shift.

The tier LABEL is identical in both populations, computed once over the whole
base before the split, so `Top` means the same thing in both and agrees with
`customer_lifetime_value.clv_segment`. Only the aggregates differ.

GRAIN MATTERS IN TWO OF THESE COLUMNS
-------------------------------------
"Average spend per order" has two defensible readings that differ, and the
difference is large at the top of the book:

  * order grain -- tier revenue divided by tier orders, which is the average
    size of an order placed by someone in this tier. That is what
    `avg_spend_per_order` and `median_spend_per_order` are.
  * customer grain -- the mean of each customer's own average order value.
    That weights a one-order customer the same as a 200-order customer, so it
    describes the typical CUSTOMER rather than the typical ORDER.

Published as the order-grain reading because the question "what does an order
from this tier look like" is the one a basket decision needs, and because
`customer_lifetime_value.avg_order_value` already holds the per-customer
number for anyone who wants the other one.

`median_spend_per_order` is the reason this table exists rather than being
computed from `customer_lifetime_value`. A median order value cannot be
recovered from one row per customer at all -- only a median of per-customer
averages can, which is a different number.

A NOTE WORTH CARRYING INTO ANY READING OF THIS TABLE
----------------------------------------------------
`avg_spend_per_order` is $13.84 in BOTH the Top and the Mid tier. The tiers do
not differ in what an order looks like; they differ in how many orders there
are (median 11 against 1). Lifetime value here is frequency, not basket size,
and a "grow the basket" reading of this table is not supported by it. The
Bottom tier is the exception and differs in kind rather than degree: $6.01 an
order and 96.21% one-order customers, which is a failed first purchase rather
than a low-value customer.

WHAT THE TIERS ARE NOT
----------------------
A partition of lifetime spend, which accumulates with observation time. A
customer who joined last month cannot be in the Top tier however good they
are, so `median_tenure_days` is published beside the spend: the tiers are
part cohort age, not pure value. Read it with `clv_tier_segments`, which
splits each tier by its 90-day RFM segment and shows how much of the Top tier
is currently dormant.
"""

from __future__ import annotations

from datetime import datetime

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from python_scripts.silver_to_gold.transforms import clv_history, segment_history

# The grain, so the group-bys and the joins below cannot drift apart.
_GRAIN = ["clv_tier", "outliers"]


def _pct(numerator, denominator):
    """A percentage that returns NULL rather than raising on a zero total.
    """
    return (100 * F.try_divide(numerator, denominator)).cast("decimal(5,2)")


def build(inputs: dict[str, DataFrame], batch_date: datetime) -> DataFrame:
    items = inputs["order_items"]

    orders = segment_history.per_order(items)

    # Tier first, then split into populations: one definition of "top fifth",
    # two sets of aggregates over it. Reversing these two calls would re-rank
    # the excluded population and give `Top` two meanings.
    per_customer = clv_history.with_outlier_variants(
        clv_history.with_tier(clv_history.lifetime(items)))

    # The dataset's own last order date, never batch_date -- the data ends
    # 2024-02-20 and batch_date is pinned well past it, so a recency measured
    # from the batch would make every customer look silent for years.
    anchor = orders.agg(F.max("order_date")).collect()[0][0]
    if anchor is None:
        anchor = batch_date.date()

    customer_grain = (
        per_customer
        .withColumn("recency_days",
                    F.datediff(F.lit(anchor), F.col("last_order_date")))
        .withColumn("tenure_days",
                    F.datediff(F.lit(anchor), F.col("first_order_date")))
        .groupBy(*_GRAIN)
        .agg(
            F.count(F.lit(1)).alias("customers"),
            F.sum("total_lifetime_revenue").cast("decimal(18,2)")
             .alias("total_spend"),
            F.percentile_approx("total_lifetime_revenue", 0.5)
             .cast("decimal(18,2)").alias("median_total_spend"),
            F.avg("total_lifetime_revenue").cast("decimal(18,2)")
             .alias("avg_total_spend"),
            F.min("total_lifetime_revenue").cast("decimal(18,2)")
             .alias("spend_floor"),
            F.max("total_lifetime_revenue").cast("decimal(18,2)")
             .alias("spend_ceiling"),
            F.sum("total_orders").alias("total_orders"),
            F.avg("total_orders").cast("decimal(10,2)").alias("avg_orders"),
            F.percentile_approx("total_orders", 0.5).alias("median_orders"),
            # The single clearest characteristic of the Bottom tier, so it is a
            # column rather than something the reader infers from avg_orders.
            F.avg(F.when(F.col("total_orders") == 1, 1.0).otherwise(0.0))
             .alias("_one_order_frac"),
            F.percentile_approx("tenure_days", 0.5).alias("median_tenure_days"),
            F.percentile_approx("recency_days", 0.5)
             .alias("median_recency_days"),
            # How many of this row's customers the flag caught. Zero in every
            # 'excluded' row by construction; publishing it is what makes the
            # exclusion auditable from the table itself rather than only from
            # the difference between two rows.
            F.sum(F.col("is_volume_outlier").cast("int"))
             .alias("volume_outliers"),
        )
        .withColumn("one_order_pct",
                    (100 * F.col("_one_order_frac")).cast("decimal(5,2)"))
        .drop("_one_order_frac")
    )

    # Order grain. `avg(order_value)` over the orders of a tier is identically
    # tier revenue / tier orders, so the average and the median come from the
    # same frame and cannot disagree about which orders belong to the tier.
    #
    # The join is against the two-variant frame, which is what keeps the order
    # population in step with the customer population for free: a flagged
    # customer has no 'excluded' row, so their orders match only 'included'.
    # No second, order-level exclusion list exists to drift.
    order_grain = (
        orders
        .join(per_customer.select("user_id", *_GRAIN),
              on="user_id", how="inner")
        .groupBy(*_GRAIN)
        .agg(
            F.avg("order_value").cast("decimal(10,2)")
             .alias("avg_spend_per_order"),
            F.percentile_approx("order_value", 0.5).cast("decimal(10,2)")
             .alias("median_spend_per_order"),
            F.avg("units").cast("decimal(10,2)").alias("avg_units_per_order"),
            F.percentile_approx("units", 0.5).alias("median_units_per_order"),
        )
    )

    # Totals PER POPULATION, so a share is of its own population and the three
    # percentages still sum to 100 down each variant. Grouped rather than
    # cross-joined for that reason -- a single-row total would divide the
    # excluded rows by the included base and quietly understate every one.
    totals = customer_grain.groupBy("outliers").agg(
        F.sum("customers").alias("_all_customers"),
        F.sum("total_spend").alias("_all_spend"),
        F.sum("total_orders").alias("_all_orders"),
    )

    return (
        customer_grain
        .join(order_grain, on=_GRAIN, how="inner")
        .join(totals, on="outliers", how="inner")
        # Published because the tiers are NOT 20/60/20: ties stay together, so
        # 475 customers at exactly $9.89 sit just inside Mid and the Bottom
        # tier realises 18.04%. See clv_history.
        .withColumn("pct_of_customers",
                    _pct(F.col("customers"), F.col("_all_customers")))
        .withColumn("pct_of_spend",
                    _pct(F.col("total_spend"), F.col("_all_spend")))
        .withColumn("pct_of_orders",
                    _pct(F.col("total_orders"), F.col("_all_orders")))
        .withColumn("anchor_date", F.lit(anchor))
        .withColumn("batch_date",
                    F.lit(batch_date.strftime("%Y-%m-%d")).cast("date"))
        .withColumn("updated_at", F.current_timestamp())
        .drop("_all_customers", "_all_spend", "_all_orders")
    )
