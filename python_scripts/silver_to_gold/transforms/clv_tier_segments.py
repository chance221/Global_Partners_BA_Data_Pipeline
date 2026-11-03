"""What each lifetime-spend tier is actually made of, by 90-day RFM segment.

Grain is (clv_tier, segment, outliers). This is the table that answers "what
puts a customer in this tier", by crossing a LIFETIME measure against a RECENT
one: `clv_tier` is all of a customer's history, `segment` is the trailing 90
days. The two disagreeing is the interesting part -- a Top-tier customer
sitting in Dormant is past value with no present activity, and no
single-window table can show that.

It is also where the headline finding of the CLV work lives: measured at the
anchor, 2,367 Top-tier customers are currently Dormant and hold $606,478.70,
which is 49.1% of all Top-tier lifetime spend. Nearly half the value of the
best customers belongs to people who have not ordered in 90 days, and that is
invisible in any active-only view because those views drop Dormant by
construction.

TWO POPULATIONS
---------------
`outliers = 'included'` is every identified customer; `outliers = 'excluded'`
leaves out the three volume accounts described in `clv_history`. All three are
Top tier, two are Slipping at the anchor and one has been Dormant for 974
days, so the exclusion touches exactly three cells of this table and nothing
else. The tier labels and the VIP cut are identical in both populations.

A percentage is of its OWN population: `pct_of_tier` sums to 100 within an
(clv_tier, outliers) pair, not within a tier across both. Pin `outliers` to
one value before reading any of them.

The VIP threshold here is the LIVE cut, `vip_cut` at the anchor, not
`frozen_vip_cut`. The frozen bar exists so tables that compare two points in
time use one definition of VIP; this table compares nothing across time, so it
should agree with `customer_rfm` instead, and `customer_rfm` recomputes its cut
every run. `tests/test_segment_history.py` is what keeps that agreement honest.
The cut is computed over all orders, both populations, for the same reason the
tier is: one definition of VIP, whichever way the aggregates are being read.

TWO PERCENTAGES, BECAUSE THEY ANSWER DIFFERENT QUESTIONS
--------------------------------------------------------
  * `pct_of_tier` reads across a tier: what the Top tier is composed of.
    Sums to 100 within each clv_tier.
  * `pct_of_segment` reads across a segment: what share of VIPs have reached
    the Top tier. Sums to 100 within each segment.

Publishing only one of them invites the other to be read off it, which is the
classic row-percent / column-percent confusion. Both are columns so neither
has to be inferred.

Segments are the full nine including Dormant -- unlike every 90-day table,
which drops Dormant because its window figures are zero. Here Dormant is the
most informative cell in the table, because lifetime spend does not go to zero
when activity does.
"""

from __future__ import annotations

from datetime import datetime

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from python_scripts.silver_to_gold.transforms import clv_history, segment_history

# The grain, so the group-bys and the joins below cannot drift apart.
_GRAIN = ["clv_tier", "segment", "outliers"]


def build(inputs: dict[str, DataFrame], batch_date: datetime) -> DataFrame:
    items = inputs["order_items"]

    orders = segment_history.per_order(items)

    # Tier first, then split into populations -- see clv_tier_profile. The
    # excluded variant must not re-rank, or `Top` means two different things.
    per_customer = clv_history.with_outlier_variants(
        clv_history.with_tier(clv_history.lifetime(items)))

    anchor = orders.agg(F.max("order_date")).collect()[0][0]
    if anchor is None:
        anchor = batch_date.date()

    cut = segment_history.vip_cut(orders, anchor)
    labelled = (
        per_customer
        .join(segment_history.segments(orders, anchor, cut)
              .select("user_id", "segment", "segment_group"),
              on="user_id", how="inner")
    )

    cells = (
        labelled
        .groupBy(*_GRAIN, "segment_group")
        .agg(
            F.count(F.lit(1)).alias("customers"),
            F.sum("total_lifetime_revenue").cast("decimal(18,2)")
             .alias("total_spend"),
            F.percentile_approx("total_lifetime_revenue", 0.5)
             .cast("decimal(18,2)").alias("median_total_spend"),
            F.avg("total_orders").cast("decimal(10,2)").alias("avg_orders"),
            F.percentile_approx("total_orders", 0.5).alias("median_orders"),
            F.avg("avg_order_value").cast("decimal(10,2)")
             .alias("avg_customer_order_value"),
            # Which cells the flag actually touched: three of them, so this
            # column is how a reader finds them without diffing two variants.
            F.sum(F.col("is_volume_outlier").cast("int"))
             .alias("volume_outliers"),
        )
    )

    # Denominators from the same frame the cells came from, so a row percentage
    # cannot disagree with the cells it divides -- the same reason
    # customer_segment_transitions takes its denominator from the paired frame.
    #
    # Both are grouped by `outliers` as well, so each share is of its own
    # population. Dropping it from either group-by would divide the excluded
    # cells by the included totals and understate every percentage in the
    # table by a little, which is the kind of error that looks like rounding.
    by_tier = (cells.groupBy("clv_tier", "outliers")
               .agg(F.sum("customers").alias("_tier_customers"),
                    F.sum("total_spend").alias("_tier_spend")))
    by_segment = (cells.groupBy("segment", "outliers")
                  .agg(F.sum("customers").alias("_segment_customers")))

    return (
        cells
        .join(by_tier, on=["clv_tier", "outliers"], how="inner")
        .join(by_segment, on=["segment", "outliers"], how="inner")
        .withColumn("pct_of_tier",
                    (100 * F.try_divide(F.col("customers"),
                                        F.col("_tier_customers")))
                    .cast("decimal(5,2)"))
        .withColumn("pct_of_tier_spend",
                    (100 * F.try_divide(F.col("total_spend"),
                                        F.col("_tier_spend")))
                    .cast("decimal(5,2)"))
        .withColumn("pct_of_segment",
                    (100 * F.try_divide(F.col("customers"),
                                        F.col("_segment_customers")))
                    .cast("decimal(5,2)"))
        .withColumn("vip_spend_cut", F.lit(cut).cast("decimal(18,2)"))
        .withColumn("anchor_date", F.lit(anchor))
        .withColumn("batch_date",
                    F.lit(batch_date.strftime("%Y-%m-%d")).cast("date"))
        .withColumn("updated_at", F.current_timestamp())
        .drop("_tier_customers", "_tier_spend", "_segment_customers")
    )
