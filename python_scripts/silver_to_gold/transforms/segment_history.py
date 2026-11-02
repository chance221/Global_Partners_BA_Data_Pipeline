"""Segmenting customers as they stood at an arbitrary past anchor.

`customer_rfm` segments at one anchor -- the dataset's last order date -- and
needs nothing else. Three tables need the same rules evaluated at a DIFFERENT
anchor, or at several:

  loyalty_era_profile          two anchors, one per era
  customer_segment_transitions six anchors, a T0 and a T1 per era
  loyalty_cohort_outcomes      two anchors, T0 and T1

This module is that shared implementation. It was `_rfm_features`, `_vip_cut`
and `_segments` inside loyalty_era_profile; lifted here when the third and
fourth consumers arrived, because a fourth copy of the segment CASE would have
made drift inevitable.

The thresholds still live in `customer_rfm` and are imported, not copied, so
there is one definition of what a VIP is. The CASE expression here remains a
second implementation of the one in `customer_rfm`, for the reason given there:
that one anchors to the last order date and reads a windowed percentile, so it
cannot take an anchor argument without changing the published table. The two
are kept in step by `tests/test_segment_history.py`, which segments the same
rows both ways at that anchor and asserts they agree.

THE ANCHOR ARGUMENT IS LOAD-BEARING
-----------------------------------
`order_date <= anchor` is what makes these functions usable in the past.
Without it a customer whose first order came AFTER the anchor is counted into
the cohort as Dormant, which silently inflates every denominator. `customer_rfm`
has no equivalent filter because its anchor is the maximum order date, where
there is nothing later to exclude.
"""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from python_scripts.silver_to_gold.transforms.customer_rfm import (
    ACTIVE_WINDOW_DAYS,
    LOYAL_MIN_ORDERS_90D,
    NEW_MAX_TENURE_DAYS,
    PERIOD_DAYS,
    REENGAGE_DORMANCY_DAYS,
    SLIPPING_MIN_ORDERS_90D,
    VIP_SPEND_PERCENTILE,
)

# The five segment_group labels that are not Dormant, plus Dormant. Published
# as a constant because two tables build a destination column from it and a
# typo in one would read as a real zero rather than an error.
CORE = "Core"
DORMANT = "Dormant"


def per_order(items: DataFrame) -> DataFrame:
    """One row per (user_id, order_id) with its date, value, units and flag.

    Anonymous orders are dropped: a per-customer measure cannot be built
    without a customer. An order counts as a loyalty order if ANY of its lines
    is flagged, which is what makes a line-level flag usable at order grain.
    """
    identified = items.filter(
        F.col("user_id").isNotNull() & (F.trim(F.col("user_id")) != "")
    )
    return (
        identified
        .groupBy("user_id", "order_id")
        .agg(
            F.min(F.to_date(F.col("creation_time_utc"))).alias("order_date"),
            F.sum("item_price").cast("decimal(18,2)").alias("order_value"),
            F.sum("item_quantity").alias("units"),
            F.sum(F.coalesce(F.col("is_loyalty"), F.lit(False)).cast("int"))
             .alias("_flagged_lines"),
        )
        .withColumn("is_loyalty_order", F.col("_flagged_lines") > 0)
        .drop("_flagged_lines")
    )


def rfm_features(orders: DataFrame, anchor: date) -> DataFrame:
    """Per-customer RFM features as of `anchor`, ignoring everything after it.

    Bucket boundaries match customer_rfm: the window is half-open,
    (anchor - 90, anchor], so a customer cannot land in two periods.
    """
    history = orders.filter(F.col("order_date") <= F.lit(anchor))

    p1_start = anchor - timedelta(days=PERIOD_DAYS)
    p2_start = anchor - timedelta(days=2 * PERIOD_DAYS)
    window_start = anchor - timedelta(days=ACTIVE_WINDOW_DAYS)

    in_p1 = F.col("order_date") > F.lit(p1_start)
    in_p2 = ((F.col("order_date") <= F.lit(p1_start))
             & (F.col("order_date") > F.lit(p2_start)))
    in_p3 = ((F.col("order_date") <= F.lit(p2_start))
             & (F.col("order_date") > F.lit(window_start)))
    in_window = in_p1 | in_p2 | in_p3

    return (
        history
        .groupBy("user_id")
        .agg(
            F.max("order_date").alias("last_order_date"),
            F.min("order_date").alias("first_order_date"),
            # Lifetime orders AS OF the anchor, which is what the Never
            # Returned branch tests. `history` is already filtered.
            F.count(F.lit(1)).alias("total_orders"),
            F.sum(F.when(in_p1, 1).otherwise(0)).alias("orders_last_30d"),
            F.sum(F.when(in_p2, 1).otherwise(0)).alias("orders_31_60d"),
            F.sum(F.when(in_p3, 1).otherwise(0)).alias("orders_61_90d"),
            F.sum(F.when(in_window, F.col("order_value")).otherwise(0))
             .cast("decimal(18,2)").alias("spend_last90d"),
            F.min(F.when(in_p1, F.col("order_date")))
             .alias("_first_in_last_30d"),
            F.max(F.when(F.col("order_date") <= F.lit(window_start),
                         F.col("order_date"))).alias("_last_before_window"),
        )
        .withColumn("orders_last90d", F.col("orders_last_30d")
                    + F.col("orders_31_60d") + F.col("orders_61_90d"))
        .withColumn("recency_days",
                    F.datediff(F.lit(anchor), F.col("last_order_date")))
        .withColumn("tenure_days",
                    F.datediff(F.lit(anchor), F.col("first_order_date")))
        .withColumn("is_active", F.col("orders_last90d") > 0)
        .withColumn("dormancy_days",
                    F.datediff(F.col("_first_in_last_30d"),
                               F.col("_last_before_window")))
        .drop("_first_in_last_30d", "_last_before_window")
    )


def vip_cut(orders: DataFrame, anchor: date) -> Decimal:
    """The 80th percentile of 90-day spend among ACTIVE customers at `anchor`.

    Collected to a Python scalar rather than left as a window function, which
    is the difference from customer_rfm: a value that has to classify several
    anchors identically cannot be recomputed at each one. A floating
    percentile pins VIP near a fixed share of the base, which would make any
    across-anchor comparison measure nothing.

    Falls back to zero when there is no active customer to measure -- only
    reachable on a degenerate batch, and harmless there because the VIP branch
    also demands two orders in the last 30 days and one before that.
    """
    active = rfm_features(orders, anchor).filter(F.col("is_active"))
    cut = active.agg(
        F.percentile_approx(F.col("spend_last90d"), VIP_SPEND_PERCENTILE)
    ).collect()[0][0]
    return Decimal(0) if cut is None else cut


def frozen_vip_cut(orders: DataFrame, launch: date) -> Decimal:
    """The project-wide frozen VIP bar: the cut as it stood the day before launch.

    Every table that compares across time reads THIS value, so someone moving
    between loyalty_era_profile, customer_segment_transitions and
    loyalty_cohort_outcomes compares one definition of VIP rather than three.
    Each of them publishes it as `vip_spend_cut` so the agreement is auditable
    rather than asserted here.

    The day before launch, specifically, because that is the last anchor
    untouched by the program -- a bar measured after launch would already carry
    whatever the program did to spending, and "VIP grew" would be partly a
    restatement of the threshold moving.

    It is also price-comparable by luck worth recording: that anchor's 90-day
    window is 2022-12-27 to 2023-03-26, so all but five days of it sits after
    the January 2023 reprice, and every later window is entirely after it.
    """
    return vip_cut(orders, launch - timedelta(days=1))


def segments(orders: DataFrame, anchor: date, cut: Decimal) -> DataFrame:
    """Every customer's segment as it stood at `anchor`, with their features.

    First match wins, in the same order as customer_rfm, because the branches
    overlap by design -- a VIP also satisfies Loyal, and Loyal also satisfies
    Promising. Callers that want only the labels select them; the features are
    returned because two consumers need orders_last90d and spend_last90d at
    the same anchor and re-deriving them would double the work.
    """
    return (
        rfm_features(orders, anchor)
        .withColumn("segment",
                    F.when(~F.col("is_active"), "Dormant")
                     .when((F.col("tenure_days") <= F.lit(NEW_MAX_TENURE_DAYS))
                           & (F.col("orders_last_30d") >= 1), "New")
                     .when((F.col("orders_last_30d") >= 1)
                           & (F.col("orders_31_60d") == 0)
                           & (F.col("orders_61_90d") == 0)
                           & (F.col("dormancy_days")
                              >= F.lit(REENGAGE_DORMANCY_DAYS)),
                           "Re-engaging")
                     .when((F.col("orders_last_30d") >= 2)
                           & ((F.col("orders_31_60d")
                               + F.col("orders_61_90d")) >= 1)
                           & (F.col("spend_last90d") >= F.lit(cut)), "VIP")
                     .when((F.col("orders_last_30d") >= 1)
                           & (F.col("orders_last90d")
                              >= F.lit(LOYAL_MIN_ORDERS_90D)), "Loyal")
                     .when(F.col("orders_last_30d") >= 1, "Promising")
                     .when(F.col("orders_last90d")
                           >= F.lit(SLIPPING_MIN_ORDERS_90D), "Slipping")
                     # One lifetime order is not infrequency, it is a customer
                     # who never came back. The two behave nothing alike.
                     .when(F.col("total_orders") == 1, "Never Returned")
                     .otherwise("Lapsing"))
        .withColumn("segment_group",
                    F.when(F.col("segment").isin("VIP", "Loyal"), CORE)
                     .when(F.col("segment") == "New", "New Growth")
                     .when(F.col("segment") == "Promising", "Developing")
                     .when(F.col("segment") == "Slipping", "Churn Risk")
                     .when(F.col("segment").isin("Lapsing", "Re-engaging"),
                           "Infrequent")
                     .when(F.col("segment") == "Never Returned",
                           "Never Returned")
                     .otherwise(DORMANT))
    )


def members_as_of(orders: DataFrame, anchor: date) -> DataFrame:
    """Who held loyalty membership at `anchor`, one row per customer.

    Membership is taken from orders placed ON OR BEFORE the anchor. Reading it
    from all history instead would credit a customer who enrolled late with a
    period they spent outside the program, which is the single easiest way to
    make a loyalty comparison report an effect that is not there.
    """
    return (
        orders
        .filter(F.col("order_date") <= F.lit(anchor))
        .groupBy("user_id")
        .agg(F.max(F.col("is_loyalty_order").cast("int")).alias("_member"))
        .withColumn("is_member", F.col("_member") > 0)
        .drop("_member")
    )


def launch_date(orders: DataFrame, fallback: date) -> date:
    """The program's first flagged order date -- derived, never configured.

    A fixed historical event, unlike the rolling 90-day anchor: once the first
    flagged order is in the data it does not move, so every table anchored to
    it stays comparable across batches. Derived rather than hardcoded so it can
    be audited, and published by every table that uses it.

    `fallback` lands one day past the data when no order carries the flag,
    which empties any post-launch window rather than raising on a batch that
    contains no program.
    """
    found = orders.agg(
        F.min(F.when(F.col("is_loyalty_order"), F.col("order_date")))
    ).collect()[0][0]
    return fallback if found is None else found
