"""Materialise the segmentation aggregates the dashboard reads.

Why pre-materialise instead of querying Iceberg from Streamlit: a Spark session
costs a JVM start and an Ivy resolve on every script run, and Streamlit reruns
the script on every widget interaction. Every aggregate here is at most a few
thousand rows, so they are written once and the app reads them in milliseconds.

Why this calls `customer_rfm.build()` rather than reading the gold table: the
published table still carries the previous segment_group mapping until the Glue
job runs again. Importing the transform means the dashboard and the pipeline
cannot disagree about what a segment is -- there is one definition, and this
uses it.

Decimals are cast to float on the way out. pyspark returns decimal.Decimal for
money columns; Vega cannot serialise it, and it raises TypeError the moment it
meets a float, which is what any share or index calculation does.

Read-only against prod. Run:
    ./.venv/Scripts/python.exe dashboard/build_data.py
"""

from __future__ import annotations

import sys
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pandas as pd
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "notebooks"))

import review_gold as rg                                              # noqa: E402
from python_scripts.silver_to_gold.transforms import customer_rfm      # noqa: E402

OUT = Path(__file__).resolve().parent / "data"

# The first loyalty-flagged order in the data, and the VIP threshold frozen at
# the pre-loyalty anchor. Frozen because a floating percentile pins VIP near a
# fixed share of actives, which makes a pre/post comparison measure nothing.
LAUNCH = date(2023, 3, 27)
FROZEN_VIP_CUT = Decimal("49.95")
TRANSITION_DAYS = 180

# batch_date as the Glue jobs pass it. Only reaches a published column here;
# the anchor the segmentation actually uses is the last order date in silver.
BATCH_DATE = datetime(2026, 10, 16)

# Segments in the order their own definitions force on recency, not by value,
# so every chart in the app stacks them identically and the eye can track one
# segment down the page.
SEGMENT_ORDER = ["VIP", "Loyal", "Promising", "Re-engaging", "New",
                 "Slipping", "Lapsing", "Never Returned"]
GROUP_ORDER = ["Core", "Developing", "New Growth", "Churn Risk",
               "Infrequent", "Never Returned"]
PERIODS = ["P1 last 30d", "P2 31-60d", "P3 61-90d"]


def _write(name: str, frame: pd.DataFrame) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / f"{name}.csv", index=False)
    print(f"  {name}.csv  {len(frame):>5} rows")


def _floats(frame: pd.DataFrame) -> pd.DataFrame:
    """Decimal -> float, in place of pandas keeping it as an object column."""
    for col in frame.columns:
        if frame[col].dtype == object and frame[col].map(
                lambda v: isinstance(v, Decimal)).any():
            frame[col] = frame[col].astype(float)
    return frame


def _per_order(items: DataFrame) -> DataFrame:
    """One row per order, under the same user filter customer_rfm applies.

    `is_loyalty` is a line-level flag, so max() over the order makes it an
    order-level one: an order counts as a loyalty order if any line was.
    """
    ident = items.filter(
        F.col("user_id").isNotNull() & (F.trim(F.col("user_id")) != ""))
    return (ident
            .groupBy("user_id", "order_id")
            .agg(F.min(F.to_date(F.col("creation_time_utc"))).alias("order_date"),
                 F.sum("item_price").cast("decimal(18,2)").alias("order_revenue"),
                 F.sum("item_quantity").alias("order_units"),
                 F.max(F.col("is_loyalty").cast("int")).alias("loyalty_order")))


# --- the 90-day window, whole and split into its three periods ------------

def _order_stats(orders: DataFrame, key: str) -> DataFrame:
    """Order-level measures, which customer_rfm cannot supply.

    It holds one row per customer, so a median taken from it would be a median
    of per-customer averages -- a different and much less useful number than
    the median order.
    """
    total = orders.agg(F.sum("order_revenue")).collect()[0][0] or Decimal(1)
    return (orders
            .groupBy(key)
            .agg(F.count(F.lit(1)).alias("orders"),
                 F.sum("order_revenue").cast("decimal(18,2)").alias("revenue"),
                 F.round(F.avg("order_revenue"), 2).alias("avg_order"),
                 F.expr("percentile_approx(order_revenue, 0.5)").alias("med_order"),
                 F.sum("order_units").alias("units"),
                 F.round(F.avg("order_units"), 2).alias("avg_units"),
                 F.expr("percentile_approx(order_units, 0.5)").alias("med_units"))
            .withColumn("rev_share_pct",
                        F.round(F.lit(100) * F.col("revenue") / F.lit(total), 2)))


def build_window(seg: DataFrame, per_order: DataFrame) -> date:
    """Revenue, share and basket size per segment, whole window and per period."""
    anchor = seg.agg(F.max("anchor_date")).collect()[0][0]
    p1, p2, w0 = (anchor - timedelta(days=d) for d in (30, 60, 90))

    # Dormant drops out via the inner join on an active-only frame. It has no
    # orders inside the window by definition, so it would be a row of zeros --
    # which is the whole reason is_active exists.
    active = seg.filter("is_active")
    orders = (per_order
              .filter(F.col("order_date") > F.lit(w0))
              .join(active.select("user_id", "segment", "segment_group"),
                    on="user_id", how="inner")
              .withColumn("period",
                          F.when(F.col("order_date") > F.lit(p1), PERIODS[0])
                           .when(F.col("order_date") > F.lit(p2), PERIODS[1])
                           .otherwise(PERIODS[2])))

    for key in ("segment", "segment_group"):
        counts = active.groupBy(key).agg(F.count(F.lit(1)).alias("customers"))
        whole = (_order_stats(orders, key)
                 .join(counts, on=key)
                 .withColumn("orders_per_cust",
                             F.round(F.col("orders") / F.col("customers"), 2))
                 .withColumn("rev_per_cust",
                             F.round(F.col("revenue") / F.col("customers"), 2)))
        _write(f"window_{key}", _floats(whole.toPandas()))

        parts = [_floats(_order_stats(orders.filter(F.col("period") == p), key)
                         .withColumn("period", F.lit(p)).toPandas())
                 for p in PERIODS]
        _write(f"period_{key}", pd.concat(parts, ignore_index=True))

    return anchor


# --- cadence --------------------------------------------------------------

def build_cadence(seg: DataFrame) -> None:
    """Recency against each customer's OWN gap, divided per customer.

    The ratio of two medians is not the median of the ratio, and here they
    differ by enough to matter -- Slipping reads 3.16x one way and 3.01x the
    other -- so the division happens per customer before aggregating.

    avg_gap_days is NULL for a one-order customer: there is no gap to be late
    against. That is why `no_cadence` is published beside the medians. Never
    Returned is entirely no_cadence, so it has to be labelled on the chart
    rather than dropped, or 607 customers vanish without a word.
    """
    active = seg.filter("is_active").withColumn(
        "overdue", F.when(F.col("avg_gap_days") > 0,
                          F.col("recency_days") / F.col("avg_gap_days")))
    for key in ("segment", "segment_group"):
        frame = (active
                 .groupBy(key)
                 .agg(F.count(F.lit(1)).alias("customers"),
                      F.expr("percentile_approx(recency_days, 0.5)")
                       .alias("med_recency"),
                      F.round(F.expr("percentile_approx(avg_gap_days, 0.5)"), 1)
                       .alias("med_gap"),
                      F.round(F.expr("percentile_approx(overdue, 0.5)"), 2)
                       .alias("med_overdue"),
                      F.round(F.avg((F.col("overdue") >= 2).cast("int")) * 100, 1)
                       .alias("pct_2x_overdue"),
                      F.sum(F.col("overdue").isNull().cast("int"))
                       .alias("no_cadence"),
                      F.sum("total_revenue").cast("decimal(18,2)")
                       .alias("lifetime_revenue")))
        _write(f"cadence_{key}", _floats(frame.toPandas()))


# --- time to the second order --------------------------------------------

def build_reorder(per_order: DataFrame, anchor: date) -> None:
    """How long a customer took to order again, and what they became.

    Restricted to customers whose FIRST order is at least 90 days before the
    anchor. Without that cut the recent joiners are right-censored -- someone
    who first ordered three days ago cannot have reordered within seven, and
    counting them inflates the never-reordered share.
    """
    ranked = per_order.withColumn(
        "rn", F.row_number().over(
            Window.partitionBy("user_id").orderBy("order_date")))
    cust = (ranked
            .groupBy("user_id")
            .agg(F.min("order_date").alias("first_date"),
                 F.count(F.lit(1)).alias("life_orders"),
                 F.sum("order_revenue").cast("decimal(18,2)").alias("life_revenue"),
                 F.min(F.when(F.col("rn") == 2, F.col("order_date")))
                  .alias("second_date"))
            .withColumn("days_to_second",
                        F.datediff(F.col("second_date"), F.col("first_date")))
            .withColumn("observed_days",
                        F.datediff(F.lit(anchor), F.col("first_date")))
            .filter(F.col("observed_days") >= 90))

    total = cust.count()
    bucket = (F.when(F.col("days_to_second").isNull(), "never")
               .when(F.col("days_to_second") == 0, "same day")
               .when(F.col("days_to_second") <= 3, "1-3 days")
               .when(F.col("days_to_second") <= 7, "4-7 days")
               .when(F.col("days_to_second") <= 14, "8-14 days")
               .when(F.col("days_to_second") <= 30, "15-30 days")
               .otherwise("31+ days"))
    order = {"same day": 0, "1-3 days": 1, "4-7 days": 2, "8-14 days": 3,
             "15-30 days": 4, "31+ days": 5, "never": 6}

    frame = (cust
             .withColumn("bucket", bucket)
             .groupBy("bucket")
             .agg(F.count(F.lit(1)).alias("customers"),
                  F.round(F.avg("life_orders"), 2).alias("avg_life_orders"),
                  F.round(F.avg("life_revenue"), 2).alias("avg_life_revenue"),
                  F.sum("life_revenue").cast("decimal(18,2)")
                   .alias("total_life_revenue"))
             .toPandas())
    frame = _floats(frame)
    frame["bucket_order"] = frame["bucket"].map(order)
    frame["pct_of_customers"] = (100 * frame["customers"] / total).round(1)
    frame["pct_of_revenue"] = (
        100 * frame["total_life_revenue"] / frame["total_life_revenue"].sum()
    ).round(1)
    _write("reorder_cliff", frame.sort_values("bucket_order"))


# --- segmentation at an arbitrary anchor, for the transition matrices -----

def segment_at(per_order: DataFrame, anchor: date, vip_cut: Decimal) -> DataFrame:
    """The segment each customer held at `anchor`, on a frozen VIP threshold.

    The `order_date <= anchor` filter is the load-bearing line: without it,
    orders placed after the anchor leak into its windows and every customer
    looks more active than they were. Borrowed from loyalty_era_profile, which
    needs the same trick to segment a past era.

    The branch order is customer_rfm's, because the branches overlap by design
    and first match wins.
    """
    history = per_order.filter(F.col("order_date") <= F.lit(anchor))
    p1, p2, w0 = (anchor - timedelta(days=d) for d in (30, 60, 90))
    in_p1 = F.col("order_date") > F.lit(p1)
    in_p2 = (F.col("order_date") <= F.lit(p1)) & (F.col("order_date") > F.lit(p2))
    in_p3 = (F.col("order_date") <= F.lit(p2)) & (F.col("order_date") > F.lit(w0))

    pc = (history
          .groupBy("user_id")
          .agg(F.min("order_date").alias("first_order_date"),
               F.count(F.lit(1)).alias("total_orders"),
               F.sum(F.when(in_p1, 1).otherwise(0)).alias("o1"),
               F.sum(F.when(in_p2, 1).otherwise(0)).alias("o2"),
               F.sum(F.when(in_p3, 1).otherwise(0)).alias("o3"),
               F.sum(F.when(in_p1 | in_p2 | in_p3, F.col("order_revenue"))
                      .otherwise(0)).cast("decimal(18,2)").alias("spend90"),
               F.min(F.when(in_p1, F.col("order_date"))).alias("_first_p1"),
               F.max(F.when(F.col("order_date") <= F.lit(w0), F.col("order_date")))
                .alias("_last_pre"))
          .withColumn("o90", F.col("o1") + F.col("o2") + F.col("o3"))
          .withColumn("tenure",
                      F.datediff(F.lit(anchor), F.col("first_order_date")))
          .withColumn("is_active", F.col("o90") > 0)
          .withColumn("dormancy",
                      F.datediff(F.col("_first_p1"), F.col("_last_pre"))))

    return (pc
            .withColumn("segment",
                        F.when(~F.col("is_active"), "Dormant")
                         .when((F.col("tenure") <= 30) & (F.col("o1") >= 1), "New")
                         .when((F.col("o1") >= 1) & (F.col("o2") == 0)
                               & (F.col("o3") == 0) & (F.col("dormancy") >= 180),
                               "Re-engaging")
                         .when((F.col("o1") >= 2)
                               & ((F.col("o2") + F.col("o3")) >= 1)
                               & (F.col("spend90") >= F.lit(vip_cut)), "VIP")
                         .when((F.col("o1") >= 1) & (F.col("o90") >= 4), "Loyal")
                         .when(F.col("o1") >= 1, "Promising")
                         .when(F.col("o90") >= 3, "Slipping")
                         .when(F.col("total_orders") == 1, "Never Returned")
                         .otherwise("Lapsing"))
            .select("user_id", "segment", "o90", "spend90"))


def _group_of():
    """customer_rfm's group mapping, applied to the t1 segment column.

    A function rather than a module constant because F.col() needs a live
    SparkContext, which does not exist at import time.
    """
    return (F.when(F.col("seg_t1").isin("VIP", "Loyal"), "Core")
             .when(F.col("seg_t1") == "New", "New Growth")
             .when(F.col("seg_t1") == "Promising", "Developing")
             .when(F.col("seg_t1") == "Slipping", "Churn Risk")
             .when(F.col("seg_t1").isin("Lapsing", "Re-engaging"), "Infrequent")
             .when(F.col("seg_t1") == "Never Returned", "Never Returned")
             .otherwise("Dormant"))


def _transition(per_order: DataFrame, t0: date, t1: date, era: str) -> pd.DataFrame:
    s0 = (segment_at(per_order, t0, FROZEN_VIP_CUT)
          .select("user_id", F.col("segment").alias("seg_t0")))
    s1 = (segment_at(per_order, t1, FROZEN_VIP_CUT)
          .select("user_id", F.col("segment").alias("seg_t1")))
    joined = s0.join(s1, on="user_id", how="inner").withColumn("grp_t1", _group_of())
    frame = (joined
             .groupBy("seg_t0", "grp_t1")
             .agg(F.count(F.lit(1)).alias("customers"))
             .toPandas())
    frame["era"] = era
    frame["t0"], frame["t1"] = str(t0), str(t1)
    totals = frame.groupby("seg_t0")["customers"].transform("sum")
    frame["row_pct"] = (100 * frame["customers"] / totals).round(1)
    frame["n_at_t0"] = totals
    return frame


def build_transitions(per_order: DataFrame, anchor: date) -> None:
    """Where each segment's customers were 180 days later, before and after launch.

    Two 180-day spans, one wholly before the program and one wholly after, so
    the comparison is not contaminated by the launch falling inside a window.

    Both use the frozen VIP cut. Two destination groups are structurally
    unreachable and come back as zero in every row: New demands tenure <= 30,
    which nobody present at t0 can satisfy at t1, and Never Returned demands a
    single lifetime order, which would put the customer past 180 days of
    recency and so into Dormant first. The app drops both columns.
    """
    span = timedelta(days=TRANSITION_DAYS)
    frames = [_transition(per_order, LAUNCH - span, LAUNCH, "Pre-loyalty"),
              _transition(per_order, anchor - span, anchor, "Post-loyalty")]
    _write("transitions", pd.concat(frames, ignore_index=True))


# --- members against non-members, inside one window ----------------------

def build_loyalty_cohorts(per_order: DataFrame, anchor: date) -> None:
    """The apples-to-apples comparison: same window, membership split.

    The calendar pre/post comparison attributes ten months of everything --
    seasonality, the Jan 2023 reprices -- to the program. This holds the window
    fixed and splits on membership instead.

    Membership is taken AS OF t0. A customer who joined later would otherwise
    be counted as a member for a period they spent outside the program.

    It trades one bias for another and does not remove it: joining is
    self-selected, so members were already more engaged. Comparing WITHIN each
    t0 segment controls for engagement level at t0, which is why that is the
    cut the app shows; the pooled number is published beside it only to make
    the composition effect visible.
    """
    t0 = anchor - timedelta(days=TRANSITION_DAYS)
    member = (per_order
              .filter(F.col("order_date") <= F.lit(t0))
              .groupBy("user_id")
              .agg(F.max("loyalty_order").alias("flag"))
              .withColumn("cohort", F.when(F.col("flag") == 1, "member")
                          .otherwise("non-member"))
              .select("user_id", "cohort"))

    s0 = (segment_at(per_order, t0, FROZEN_VIP_CUT)
          .select("user_id", F.col("segment").alias("seg_t0")))
    s1 = (segment_at(per_order, anchor, FROZEN_VIP_CUT)
          .select("user_id", F.col("segment").alias("seg_t1"),
                  F.col("o90").alias("orders_90d_t1"),
                  F.col("spend90").alias("spend_90d_t1")))
    joined = (s0.join(s1, on="user_id", how="inner")
                .join(member, on="user_id", how="inner"))

    is_core = F.col("seg_t1").isin("VIP", "Loyal")
    measures = [
        F.count(F.lit(1)).alias("customers"),
        F.round(F.avg((F.col("seg_t1") != "Dormant").cast("int")) * 100, 1)
         .alias("still_active_pct"),
        F.round(F.avg(is_core.cast("int")) * 100, 1).alias("to_core_pct"),
        F.round(F.avg((F.col("seg_t1") == "Dormant").cast("int")) * 100, 1)
         .alias("to_dormant_pct"),
        F.round(F.avg("orders_90d_t1"), 2).alias("avg_orders_90d_t1"),
        F.round(F.avg("spend_90d_t1"), 2).alias("avg_spend_90d_t1"),
    ]

    by_segment = joined.groupBy("seg_t0", "cohort").agg(*measures)
    _write("loyalty_by_segment", _floats(by_segment.toPandas()))

    pooled = (joined.groupBy("cohort").agg(*measures)
              .withColumn("seg_t0", F.lit("ALL (pooled)")))
    _write("loyalty_pooled", _floats(pooled.toPandas()))


def main() -> None:
    spark, cfg = rg.connect("prod")
    items = spark.table(rg.fqn(cfg, "silver", "order_items"))

    built = customer_rfm.build({"order_items": items}, BATCH_DATE)
    seg = built.select(
        "user_id", "segment", "segment_group", "is_active", "recency_days",
        "avg_gap_days", "total_orders", "total_revenue", "orders_last90d",
        "spend_last90d", "anchor_date").cache()
    per_order = _per_order(items).cache()

    print("building:")
    anchor = build_window(seg, per_order)
    build_cadence(seg)
    build_reorder(per_order, anchor)
    build_transitions(per_order, anchor)
    build_loyalty_cohorts(per_order, anchor)

    totals = seg.filter("is_active").agg(
        F.count(F.lit(1)).alias("active_customers"),
        F.sum("spend_last90d").cast("decimal(18,2)").alias("spend_90d"),
        F.sum("orders_last90d").alias("orders_90d")).collect()[0]
    allc = seg.agg(F.count(F.lit(1)).alias("all_customers"),
                   F.sum("total_revenue").cast("decimal(18,2)")
                    .alias("lifetime_revenue")).collect()[0]
    _write("meta", pd.DataFrame([{
        "anchor_date": str(anchor),
        "loyalty_launch": str(LAUNCH),
        "frozen_vip_cut": float(FROZEN_VIP_CUT),
        "transition_days": TRANSITION_DAYS,
        "active_customers": totals["active_customers"],
        "all_customers": allc["all_customers"],
        "spend_90d": float(totals["spend_90d"]),
        "orders_90d": totals["orders_90d"],
        "lifetime_revenue": float(allc["lifetime_revenue"]),
    }]))
    print(f"\ndone -> {OUT}")


if __name__ == "__main__":
    main()
