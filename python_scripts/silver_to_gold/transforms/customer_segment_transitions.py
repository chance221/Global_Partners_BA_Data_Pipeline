"""Where each segment's customers ended up 180 days later.

One row per (era, seg_t0, grp_t1): of the customers who held segment `seg_t0`
at T0, how many held segment_group `grp_t1` at T1, and what share of that row
they are. `row_pct` sums to 100 across each (era, seg_t0).

HOW A ROW IS COMPOSED
---------------------
Both ends are computed by segmenting the SAME customers twice, from the same
order history, at two anchors 180 days apart:

  seg_t0   segment_history.segments(orders, t0) -- orders on or before T0 only
  grp_t1   segment_history.segments(orders, t1) -- orders on or before T1

The join is inner on user_id and is total in practice: anyone with an order by
T0 still has it by T1, so every T0 customer has a T1 destination. That is what
makes `row_pct` a distribution rather than a sample, and why a row that does
not reach 100 is a bug rather than attrition.

Destination is the GROUP, not the segment, because nine destinations per row is
a matrix nobody reads, and because the groups are the level marketing acts on.
Origin is the SEGMENT, because the whole point is telling Slipping from Lapsing.

THREE ERAS, AND WHY THE ANCHORS DIFFER IN KIND
----------------------------------------------
  Recent      t0 = last order date - 180, t1 = last order date
  Prior       t0 = last order date - 360, t1 = last order date - 180
  Pre-launch  t0 = loyalty launch - 180,  t1 = loyalty launch

Recent and Prior ride the data: their anchors move every time a batch extends
it, which is correct, because "the last 180 days" is a rolling question.

Pre-launch does not move. It is anchored to the program's first flagged order,
a fixed historical event, so it keeps meaning the same thing as batches arrive
and stays comparable with the Recent era measured years later. The distinction
matters enough that T0 and T1 are published per row: a reader can see which
anchors produced a number instead of trusting that they were the intended ones.

Pre-launch against Recent is the before-and-after the program. It is still
descriptive, not causal: a calendar split cannot separate the program from the
January-2023 reprice or from the chain's own growth, both of which sit inside
the comparison. `loyalty_cohort_outcomes` is the within-window comparison that
holds calendar time fixed instead.

TWO STRUCTURAL ZEROS, WHICH ARE NOT FINDINGS
--------------------------------------------
`New Growth` and `Never Returned` read 0.0% in every row, by construction, and
a reader who takes them for a measured result will conclude something false:

  New Growth      requires tenure_days <= 30, unsatisfiable for anyone who
                  already had an order at T0, 180 days earlier.
  Never Returned  requires exactly one lifetime order, which puts recency past
                  180 days for a T0 customer, so the Dormant branch wins first.

They are kept as columns rather than dropped so the destinations stay the full
group list and the percentages visibly sum to 100.

THE VIP BAR IS FROZEN
---------------------
At the day before launch -- `segment_history.frozen_vip_cut`, the same dollar
value `loyalty_cohort_outcomes` reads, so both tables classify VIP
identically. `customer_rfm` floats it, correctly, for a
snapshot; floating it here would pin VIP near a fixed share of each era's base
and make the comparison measure nothing. Published as `vip_spend_cut` for audit.

Full rebuild every run, and no event_time_column: a slice of one batch cannot
say what a customer looked like 180 days before it. Small -- at most nine
origins by seven destinations by three eras -- so unpartitioned.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from decimal import Decimal
from functools import reduce

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from python_scripts.silver_to_gold.transforms.segment_history import (
    frozen_vip_cut,
    launch_date,
    per_order,
    segments,
)

# The gap between the two anchors. 180 days rather than 90 because a 90-day
# transition compares a segment against a window it partly shares: a customer
# Slipping at T0 still has those orders inside T1's 90-day window, so the
# matrix would mostly restate T0. At 180 the windows are disjoint.
TRANSITION_DAYS = 180

RECENT = "Recent"
PRIOR = "Prior"
PRE_LAUNCH = "Pre-launch"


def _matrix(orders: DataFrame, era: str, t0: date, t1: date,
            cut: Decimal) -> DataFrame:
    """One era's matrix: every T0 segment against every T1 group."""
    at_t0 = (segments(orders, t0, cut)
             .select("user_id", F.col("segment").alias("seg_t0")))
    at_t1 = (segments(orders, t1, cut)
             .select("user_id", F.col("segment_group").alias("grp_t1")))

    paired = at_t0.join(at_t1, on="user_id", how="inner")

    # The row denominator, taken from the paired frame rather than from at_t0
    # so it cannot disagree with the cells it divides.
    cohort = paired.groupBy("seg_t0").agg(
        F.count(F.lit(1)).alias("customers_at_t0"))

    return (
        paired
        .groupBy("seg_t0", "grp_t1")
        .agg(F.count(F.lit(1)).alias("customers"))
        .join(cohort, on="seg_t0", how="inner")
        .withColumn("era", F.lit(era))
        .withColumn("t0", F.lit(t0))
        .withColumn("t1", F.lit(t1))
    )


def build(inputs: dict[str, DataFrame], batch_date: datetime) -> DataFrame:
    orders = per_order(inputs["order_items"])

    last_order = orders.agg(F.max("order_date")).collect()[0][0]
    if last_order is None:
        last_order = batch_date.date()
    launch = launch_date(orders, last_order + timedelta(days=1))

    span = timedelta(days=TRANSITION_DAYS)

    # One dollar bar classifies all three eras, and it is the same bar
    # loyalty_cohort_outcomes reads. See the module docstring for why it is
    # frozen at all.
    cut = frozen_vip_cut(orders, launch)

    eras = [
        (RECENT, last_order - span, last_order),
        (PRIOR, last_order - 2 * span, last_order - span),
        (PRE_LAUNCH, launch - span, launch),
    ]

    matrices = reduce(DataFrame.unionByName, [
        _matrix(orders, era, t0, t1, cut) for era, t0, t1 in eras
    ])

    return (
        matrices
        # TRY_DIVIDE and not `/`: ANSI mode is on in Spark 4, so a zero
        # denominator raises instead of returning null.
        .withColumn("row_pct",
                    (F.try_divide(F.col("customers"),
                                  F.col("customers_at_t0")) * 100)
                    .cast("decimal(5,2)"))
        .withColumn("transition_days", F.lit(TRANSITION_DAYS))
        .withColumn("vip_spend_cut", F.lit(cut).cast("decimal(18,2)"))
        .withColumn("batch_date",
                    F.lit(batch_date.strftime("%Y-%m-%d")).cast("date"))
        .withColumn("updated_at", F.current_timestamp())
    )
