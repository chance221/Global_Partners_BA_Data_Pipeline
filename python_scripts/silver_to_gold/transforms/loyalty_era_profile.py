"""The business before and after the loyalty program, on equal windows.

Eighteen rows, told apart by (era, segment): one `All customers` row per era
plus one row per RFM segment per era.

The program's first flagged order is 2023-03-27, which the transform finds
rather than hardcodes. Everything else follows from it:

  post-loyalty : launch .. the dataset's last order date
  pre-loyalty  : the SAME number of days ending the day before launch

Equal day counts are the whole point of the table. `window_days` is published
so a reader can check that rather than take it on trust.

WHAT THIS TABLE IS NOT
----------------------
It is descriptive, not causal. A calendar split cannot separate the program
from the January-2023 menu reprice, from the chain's own growth (quarterly
orders ran 905 to 13,943 over the window), or from 2020 being COVID. For
evidence that the program changed behaviour, use `loyalty_order_profile`:
its "Observed in both states" rows compare the same customers before and
after their own enrolment, which holds calendar time and customer mix fixed.

TWO WINDOWS, BOTH PUBLISHED
---------------------------
`window_days` (331) is the era, and every spend and order measure is computed
over it. `segment_window_days` (90) is the trailing window the RFM segment is
assigned from, because a segment is a statement about recent activity and
stops meaning anything stretched over eleven months. So a segment row reads:
customers who ENDED the era in this segment, and everything they did DURING
the era. Both eras use that identical definition, which is what makes the
pre/post difference comparable even though the level carries a recency flavour.

The 331-day era is also what keeps seasonality out of the comparison. Each
window contains exactly one Memorial Day, July 4th, Labor Day, Columbus Day,
Thanksgiving, Christmas and New Year. A 90-day era would have put Thanksgiving
and Christmas in the post window and neither in the pre window -- and those
two days run -61.3% and -73.5% against an ordinary day, so the program would
have been charged for the calendar.

WHY THE VIP THRESHOLD IS FROZEN
-------------------------------
`customer_rfm` recomputes `vip_spend_cut` as the 80th percentile of 90-day
spend among active customers on every run. That is right for a snapshot and
useless for measuring growth: a floating percentile pins VIP near a fixed
share of the base, so "VIP grew" could never be more than noise.

So the cut is computed ONCE, at the pre-loyalty anchor, and the same dollar
value classifies both eras. VIP growth then means more customers clearing the
bar as it stood before the program, which is a claim worth making. The value
is published for audit.

It is also price-comparable, which is luck worth recording: the pre anchor's
90-day window is 2022-12-27 to 2023-03-26, so all but five days of it sits
after the January reprice, and the post window is entirely after it. A
threshold in dollars means close to the same thing on both sides.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from decimal import Decimal
from functools import reduce

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

# Segmenting at a past anchor lives in `segment_history`, shared with
# customer_segment_transitions and loyalty_cohort_outcomes. These were three
# private helpers in this module until a third consumer needed them; the
# thresholds they read still live in customer_rfm, so there is one definition
# of what a VIP is. ACTIVE_WINDOW_DAYS is imported here because this table
# publishes it as segment_window_days.
from python_scripts.silver_to_gold.transforms.customer_rfm import (
    ACTIVE_WINDOW_DAYS,
)
from python_scripts.silver_to_gold.transforms.segment_history import (
    frozen_vip_cut,
    launch_date,
    per_order as _build_per_order,
    segments as _segments_at,
)

PRE_LOYALTY = "Pre-loyalty"
POST_LOYALTY = "Post-loyalty"

ALL_CUSTOMERS = "All customers"
ALL_GROUP = "All"

MEDIAN = 0.5


def _era_orders(per_order: DataFrame, era: str, start: date, end: date,
                days: int, vip_cut: Decimal) -> DataFrame:
    """The orders placed inside one era, each carrying its customer's segment.

    An inner join: a customer with orders in the era necessarily has a segment
    at the era's anchor, because the anchor is the era's own end date.
    """
    orders = per_order.filter(
        (F.col("order_date") >= F.lit(start)) & (F.col("order_date") <= F.lit(end))
    )
    return (
        orders
        .join(_segments_at(per_order, end, vip_cut)
               .select("user_id", "segment", "segment_group"),
              on="user_id", how="inner")
        .withColumn("era", F.lit(era))
        .withColumn("window_start", F.lit(start))
        .withColumn("window_end", F.lit(end))
        .withColumn("window_days", F.lit(days))
    )


def _with_rollup(era_orders: DataFrame) -> DataFrame:
    """Repeat every order once more under the `All customers` label.

    The same trick `_populations` uses in loyalty_order_profile: relabel and
    union, so one groupBy produces both the per-segment rows and the era
    total. Rows therefore DOUBLE -- summing revenue over this table reports
    twice the business, and the `All customers` rows are the ones to trust.
    """
    rollup = (era_orders
              .withColumn("segment", F.lit(ALL_CUSTOMERS))
              .withColumn("segment_group", F.lit(ALL_GROUP)))
    return era_orders.unionByName(rollup)


def build(inputs: dict[str, DataFrame], batch_date: datetime) -> DataFrame:
    per_order = _build_per_order(inputs["order_items"])

    # The two ends of the data, found in two actions rather than one since the
    # launch lookup moved to segment_history. The launch date is derived, not
    # configured, so a later batch that extends the data moves both windows
    # together and they stay equal.
    last_order = per_order.agg(F.max("order_date")).collect()[0][0]

    # Degenerate batches, kept to the same shape rather than special-cased.
    # No orders at all: fall back to the batch date, as customer_rfm does. No
    # flagged order: put the launch one day past the end, which empties the
    # post era and leaves every order in the pre era -- an honest answer for a
    # batch that contains no program.
    if last_order is None:
        last_order = batch_date.date()
    launch = launch_date(per_order, last_order + timedelta(days=1))

    era_days = max((last_order - launch).days + 1, 1)
    pre_end = launch - timedelta(days=1)
    pre_start = pre_end - timedelta(days=era_days - 1)

    # Frozen once, at the PRE anchor, then used for both eras. See the module
    # docstring: a floating percentile would make growth unmeasurable. Shared
    # with customer_segment_transitions and loyalty_cohort_outcomes, which read
    # the same helper, so the three tables cannot disagree about VIP.
    vip_cut = frozen_vip_cut(per_order, launch)

    eras = [
        (PRE_LOYALTY, pre_start, pre_end),
        (POST_LOYALTY, launch, last_order),
    ]
    labelled = reduce(DataFrame.unionByName, [
        _with_rollup(_era_orders(per_order, era, start, end, era_days, vip_cut))
        for era, start, end in eras
    ])

    # Measures over the ORDERS in the era.
    by_order = (
        labelled
        .groupBy("era", "segment", "segment_group", "window_start",
                 "window_end", "window_days")
        .agg(
            F.countDistinct("user_id").alias("customers"),
            F.count(F.lit(1)).alias("orders"),
            F.sum("order_value").cast("decimal(18,2)").alias("total_revenue"),
            F.percentile_approx("order_value", MEDIAN).cast("decimal(10,2)")
             .alias("median_order_value"),
            F.sum("units").alias("_units"),
        )
    )

    # Measures over the CUSTOMERS in the era, which need a per-customer frame
    # first: a median of per-customer totals cannot be read off order rows.
    per_customer = (
        labelled
        .groupBy("era", "segment", "user_id")
        .agg(
            F.count(F.lit(1)).alias("_orders"),
            F.sum("order_value").cast("decimal(18,2)").alias("_spend"),
        )
    )
    by_customer = (
        per_customer
        .groupBy("era", "segment")
        .agg(
            F.avg("_orders").cast("decimal(10,2)")
             .alias("avg_orders_per_customer"),
            F.percentile_approx("_orders", MEDIAN).cast("decimal(10,2)")
             .alias("median_orders_per_customer"),
            F.avg("_spend").cast("decimal(10,2)")
             .alias("avg_spend_per_customer"),
            F.percentile_approx("_spend", MEDIAN).cast("decimal(10,2)")
             .alias("median_spend_per_customer"),
        )
    )

    joined = by_order.join(by_customer, on=["era", "segment"], how="inner")

    # Shares are taken against the era's `All customers` row rather than a
    # window SUM over the table, because the rollup union means a plain SUM
    # would double every total and halve every share.
    era_totals = (
        joined
        .filter(F.col("segment") == F.lit(ALL_CUSTOMERS))
        .select(F.col("era"),
                F.col("total_revenue").alias("_era_revenue"),
                F.col("customers").alias("_era_customers"))
    )

    return (
        joined
        .join(era_totals, on="era", how="left")
        .withColumn("avg_order_value",
                    F.try_divide(F.col("total_revenue"), F.col("orders"))
                     .cast("decimal(10,2)"))
        # TRY_DIVIDE and not `/`: ANSI mode is on in Spark 4, so a zero
        # denominator raises instead of returning null. Units can legitimately
        # be zero on a malformed batch.
        .withColumn("units_per_order",
                    F.try_divide(F.col("_units"), F.col("orders"))
                     .cast("decimal(10,2)"))
        .withColumn("revenue_per_unit",
                    F.try_divide(F.col("total_revenue"), F.col("_units"))
                     .cast("decimal(10,2)"))
        .withColumn("revenue_share_pct",
                    (F.try_divide(F.col("total_revenue"),
                                  F.col("_era_revenue")) * 100)
                    .cast("decimal(5,2)"))
        .withColumn("customer_share_pct",
                    (F.try_divide(F.col("customers"),
                                  F.col("_era_customers")) * 100)
                    .cast("decimal(5,2)"))
        .withColumn("segment_window_days", F.lit(ACTIVE_WINDOW_DAYS))
        .withColumn("vip_spend_cut", F.lit(vip_cut).cast("decimal(18,2)"))
        .withColumn("batch_date",
                    F.lit(batch_date.strftime("%Y-%m-%d")).cast("date"))
        .withColumn("updated_at", F.current_timestamp())
        .drop("_units", "_era_revenue", "_era_customers")
    )
