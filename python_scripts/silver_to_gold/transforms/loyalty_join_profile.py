"""Order rate in 30-day periods around each customer's own enrollment date.

The question this answers: once a customer joins the loyalty program, do they
order MORE OFTEN, or just spend more per order? `loyalty_order_profile` cannot
answer it. That table splits a switcher's orders by FLAG, and a switcher's
unflagged orders span up to three years while the flagged ones span at most
eleven months, so orders-per-customer falls after enrollment for reasons that
have nothing to do with the program. On this data the naive reading is 13.24
orders before against 7.42 after -- the program looking actively harmful, and
the number is pure window length.

WHAT MAKES THIS COMPARABLE
--------------------------
Every row is a RATE over a fixed 30-day period, and the periods are symmetric
around each customer's OWN join date rather than around a calendar date:

    period -2  = days -60..-31      period +1 = days   1..30
    period -1  = days -30..-1       period +2 = days  31..60
    period  0  = the join day alone

Two consequences, each of them the point:

  * Exposure is equal by construction, so the measure cannot be inflated by
    observing members for longer. That is the trap `loyalty_era_profile`
    documents at the era level and it applies with more force here.
  * It is WITHIN-PERSON. The same customers appear on both sides, so none of
    this is the difference between people who join and people who do not.

WHAT THE ANCHORING DOES **NOT** BUY, MEASURED
---------------------------------------------
Anchoring per customer is not calendar-independence, and it is tempting to
assume it is. Joining is heavily concentrated at launch: **36.6% of switchers
joined within 30 days of 2023-03-27 and 51.6% within 60**, with the four
busiest single join dates all falling in 2023-03-28..2023-04-01. So for
roughly half the cohort these periods are very nearly a FIXED calendar
window. Period -2's lower edge has a median of 2023-03-23 and an
interquartile range of 2023-02-10..2023-07-05, which puts the modal joiner's
period -2 squarely on the ~29 menu reprices of January 2023.

Read a per-period movement as a calendar effect until something rules that
out. The anchoring removes the window-length artifact; it does not remove the
calendar.

THE JOIN DAY IS EXCLUDED FROM BOTH SIDES, AND SITS IN PERIOD 0 ALONE
--------------------------------------------------------------------
By definition the join date is a day the customer ordered -- it is the day of
their first flagged order. Folding it into period +1 would guarantee that
side an order the pre side has no equivalent of, which on its own produces a
"lift". It is published as its own period, one day wide, so the mechanical
spike is visible rather than smuggled into the first post period.
`window_days` says 1 on that row and 30 on every other, so any rate computed
as `orders / customers / window_days` stays honest across all of them.

THE VOLUME ACCOUNTS ARE PUBLISHED AS A PORTION, NOT DROPPED
-----------------------------------------------------------
They are not a rounding error here. Two flagged accounts place bulk orders --
94, 93, 88, 76, 64 orders on single days in January and February 2023 -- and
in period -2 the twenty heaviest customer-days alone are **30.2% of that
period's orders**. Unnetted, they are the single most striking feature of the
curve and they are a procurement pattern, not a member responding to a
loyalty program.

So every additive measure ships with its `outlier_*` portion, exactly as
`sales_trend_grid` does, and the excluded reading is `measure -
outlier_measure` at any rollup. That keeps the dashboard's volume-outlier
toggle working on this table instead of making it an exception, and it keeps
the judgement reversible: who is flagged comes from `clv_history`, so the
floor lives in one place for every table that cuts on it.

`active_days` and `customers` subtract cleanly too, because a customer-day
belongs to exactly one customer. `median_order_value` does NOT -- a published
median cannot be un-aggregated -- and neither can `avg_order_value`, which a
consumer must recompute as
`(total_revenue - outlier_revenue) / (orders - outlier_orders)`.

WHY TWO COVERAGE POPULATIONS, AND WHY THEY OVERLAP
--------------------------------------------------
A customer can only be in a +/-N-day comparison if the data actually covers N
days either side of their join date -- otherwise a missing period reads as
zero orders when it is really no observation. `coverage_days` is that
requirement, and the table publishes both 90 and 180.

They are not alternatives and they are NOT additive: the 180-day cohort is a
SUBSET of the 90-day cohort, so summing measures across `coverage_days`
double-counts. Both ship because the choice is a real trade-off and neither
side of it should be hidden:

  * 180 reaches far enough out to show whether a lift persists, but it can
    only include customers who joined at least 180 days before the data ends,
    which means EARLY joiners only -- and those are exactly the launch cohort
    the calendar note above is about.
  * 90 keeps many more customers and is closer to representative, but it
    stops while the effect is still visibly elevated.

`customers` is published on every row and is constant within a
`coverage_days` population, which is what makes the periods comparable to
each other and what shows the cost of the wider window.

EVERY MEASURE EXCEPT THE MEDIAN IS ADDITIVE ACROSS PERIODS
----------------------------------------------------------
Summing `orders`, `total_revenue`, `line_items`, `units`, `active_days` and
each `outlier_*` portion over any set of periods within one `coverage_days`
gives the true total for that span, so a consumer can build a +/-90 reading
out of periods -3..-1 and 1..3 without a second table -- and, because the
cohort is then held constant, compare it against the +/-180 reading on the
same customers. `median_order_value` is not additive and is null on a period
with no orders.

WHAT THIS TABLE DOES NOT ESTABLISH
----------------------------------
Being within-person removes self-selection on WHO joins. It does not remove
selection on WHEN they join: people enroll during an active phase, so a
window centred on a self-selected peak shows a lift even if the program did
nothing. Reading the periods in order is what exposes that -- a level shift
and a transient look identical if you only compare two buckets. Settling it
needs a difference-in-differences against matched non-joiners on a
pseudo-join date, which is not something a single aggregate can carry.
"""

from __future__ import annotations

from datetime import datetime
from functools import reduce

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F

from python_scripts.silver_to_gold.transforms import clv_history
# Imported rather than redeclared, so this table and loyalty_order_profile
# cannot disagree about what an order is or about which orders are flagged.
# `_per_order` does not carry a date -- it groups to (user_id, order_id) and
# keeps basket measures -- so the dates are joined on below rather than by
# widening a helper another shipped table depends on.
from python_scripts.silver_to_gold.transforms.loyalty_order_profile import (
    _per_order,
)

PERIOD_DAYS = 30
# The half-widths published, as `coverage_days`. See the docstring: subset
# populations, deliberately overlapping, never summed across.
COVERAGE = (90, 180)
MEDIAN = 0.5


def _period_index(days: Column) -> Column:
    """Signed 30-day period number, with the join day alone in period 0.

    Day 1..30 is period 1 and day -1..-30 is period -1, so neither side
    borrows a day from the other. Expressed with floor division rather than
    `ceil(days / 30)` because Spark's ceil on a negative quotient rounds
    toward zero, which would put day -1 and day +1 in the same period.
    """
    forward = F.floor((days - 1) / PERIOD_DAYS) + 1
    backward = -(F.floor((-days - 1) / PERIOD_DAYS) + 1)
    return (F.when(days == 0, F.lit(0))
            .when(days > 0, forward)
            .otherwise(backward)
            .cast("int"))


def _orders_with_dates(items: DataFrame) -> DataFrame:
    """Per-order basket measures plus the order's own date."""
    dates = (items
             .groupBy("user_id", "order_id")
             .agg(F.min(F.to_date(F.col("creation_time_utc")))
                  .alias("order_date")))
    return _per_order(items).join(dates, on=["user_id", "order_id"],
                                  how="inner")


def _with_outlier_flag(orders: DataFrame, items: DataFrame) -> DataFrame:
    """Tag each order with whether its customer is a flagged volume account.

    Recomputed from silver through the shared rule rather than read from
    `customer_lifetime_value`: a gold transform cannot read another gold
    table, and the flag has to mean the same thing here as it does there.
    """
    flagged = (clv_history.lifetime(items)
               .filter(F.col("is_volume_outlier"))
               .select("user_id")
               .withColumn("_flagged", F.lit(True)))
    # LEFT then coalesced: a customer with no match has no flag, and
    # `NULL & anything` would make their contribution NULL rather than zero.
    return (orders
            .join(flagged, on="user_id", how="left")
            .withColumn("_outlier", F.coalesce(F.col("_flagged"), F.lit(False)))
            .drop("_flagged"))


def _switchers(orders: DataFrame) -> DataFrame:
    """One row per customer observed in both states, with their join date.

    The same `_any and not _all` test loyalty_order_profile uses for its
    "Observed in both states" population, so the two tables describe the same
    2,780 customers. The join date is the first FLAGGED order, which is the
    only enrollment signal silver carries -- there is no membership table.
    """
    return (orders
            .groupBy("user_id")
            .agg(F.max(F.col("is_loyalty_order").cast("int")).alias("_any"),
                 F.min(F.col("is_loyalty_order").cast("int")).alias("_all"),
                 F.min(F.when(F.col("is_loyalty_order"), F.col("order_date")))
                  .alias("join_date"),
                 # Per-customer already, so max is just the value.
                 F.max(F.col("_outlier").cast("int")).alias("_outlier_int"))
            .filter((F.col("_any") == 1) & (F.col("_all") == 0))
            .withColumn("_outlier", F.col("_outlier_int") == 1)
            .select("user_id", "join_date", "_outlier"))


def _portion(flag: Column, measure: Column | None = None) -> Column:
    """The flagged accounts' share of a measure, zero rather than null."""
    value = F.lit(1) if measure is None else measure
    return F.coalesce(F.sum(F.when(flag, value)), F.lit(0))


def _population(orders: DataFrame, switchers: DataFrame, coverage: int,
                data_start, data_end) -> DataFrame:
    """One coverage population: every period, with zeros where there are none.

    The period spine is built by exploding a literal array against a single
    row of the cohort, so a period in which nobody ordered is published as a
    zero rather than going missing -- a gap in a decay curve reads as "no
    data" when it means "no orders". An EMPTY cohort yields no rows at all,
    which is the honest answer and not thirteen rows of zeros implying a
    cohort that never existed.
    """
    cohort = switchers.filter(
        (F.datediff(F.col("join_date"), F.lit(data_start)) >= coverage)
        & (F.datediff(F.lit(data_end), F.col("join_date")) >= coverage))

    tagged = (orders
              .join(cohort.select("user_id", "join_date"), on="user_id",
                    how="inner")
              .withColumn("days_from_join",
                          F.datediff(F.col("order_date"),
                                     F.col("join_date")))
              .filter(F.col("days_from_join").between(-coverage, coverage))
              .withColumn("period_index",
                          _period_index(F.col("days_from_join"))))

    flag = F.col("_outlier")
    # A customer-day, as a comparable value so countDistinct can count the
    # pairs rather than either column on its own.
    customer_day = F.struct("user_id", "order_date")
    measures = (tagged
                .groupBy("period_index")
                .agg(F.count(F.lit(1)).cast("bigint").alias("orders"),
                     F.sum("order_value").cast("decimal(18,2)")
                      .alias("total_revenue"),
                     F.sum("line_items").cast("bigint").alias("line_items"),
                     F.sum("units").cast("bigint").alias("units"),
                     # Distinct customer-days: a customer who ordered three
                     # times on one day is one active day, which is what
                     # separates "comes back more often" from "orders more
                     # per visit".
                     F.countDistinct(customer_day).cast("bigint")
                      .alias("active_days"),
                     F.percentile_approx("order_value", MEDIAN)
                      .cast("decimal(10,2)").alias("median_order_value"),
                     _portion(flag).cast("bigint").alias("outlier_orders"),
                     _portion(flag, F.col("order_value"))
                      .cast("decimal(18,2)").alias("outlier_revenue"),
                     _portion(flag, F.col("line_items")).cast("bigint")
                      .alias("outlier_line_items"),
                     _portion(flag, F.col("units")).cast("bigint")
                      .alias("outlier_units"),
                     F.countDistinct(F.when(flag, customer_day))
                      .cast("bigint").alias("outlier_active_days")))

    periods = sorted([0] + [sign * k
                            for k in range(1, coverage // PERIOD_DAYS + 1)
                            for sign in (-1, 1)])
    spine = (cohort
             .select(F.lit(1).alias("_seed")).limit(1)
             .select(F.explode(F.array(*[F.lit(p) for p in periods]))
                     .alias("period_index")))

    counted = cohort.agg(
        F.countDistinct("user_id").cast("bigint").alias("customers"),
        F.countDistinct(F.when(flag, F.col("user_id"))).cast("bigint")
         .alias("outlier_customers"))

    zeros = {"orders": 0, "total_revenue": 0, "line_items": 0, "units": 0,
             "active_days": 0, "outlier_orders": 0, "outlier_revenue": 0,
             "outlier_line_items": 0, "outlier_units": 0,
             "outlier_active_days": 0}
    return (spine
            .join(measures, on="period_index", how="left")
            .fillna(zeros)
            .crossJoin(counted)
            .withColumn("coverage_days", F.lit(coverage).cast("int"))
            # Period 0 is the join day and is one day wide; everything else is
            # a full 30. Published so a consumer never has to assume it.
            .withColumn("window_days",
                        F.when(F.col("period_index") == 0, F.lit(1))
                         .otherwise(F.lit(PERIOD_DAYS)).cast("int"))
            .withColumn(
                "period_start_day",
                F.when(F.col("period_index") == 0, F.lit(0))
                 .when(F.col("period_index") > 0,
                       (F.col("period_index") - 1) * PERIOD_DAYS + 1)
                 .otherwise(F.col("period_index") * PERIOD_DAYS).cast("int"))
            .withColumn(
                "period_end_day",
                F.when(F.col("period_index") == 0, F.lit(0))
                 .when(F.col("period_index") > 0,
                       F.col("period_index") * PERIOD_DAYS)
                 .otherwise((F.col("period_index") + 1) * PERIOD_DAYS - 1)
                 .cast("int")))


def build(inputs: dict[str, DataFrame], batch_date: datetime) -> DataFrame:
    items = inputs["order_items"]
    orders = _with_outlier_flag(_orders_with_dates(items), items)
    switchers = _switchers(orders)

    # The coverage test asks whether the DATA spans N days either side of a
    # join date, not whether the customer was active in that span: a customer
    # who simply did not order is a real zero. Collected once, like the span
    # in category_holiday_sales, because the bound is a scalar and pushing it
    # through as a column would cross-join every row against it.
    span = orders.agg(F.min("order_date"), F.max("order_date")).collect()[0]
    data_start, data_end = span[0], span[1]
    if data_start is None:
        # No identified orders at all. Keeps the frame buildable; the cohort
        # filter below then matches nothing and the result is empty.
        data_start = data_end = batch_date.date()

    frames = [_population(orders, switchers, coverage, data_start, data_end)
              for coverage in COVERAGE]

    return (reduce(DataFrame.unionByName, frames)
            .withColumn("orders_per_customer",
                        (F.col("orders") / F.col("customers"))
                        .cast("decimal(10,4)"))
            # The one measure that is comparable across EVERY row including
            # the one-day join period.
            .withColumn("orders_per_customer_per_day",
                        (F.col("orders")
                         / F.col("customers") / F.col("window_days"))
                        .cast("decimal(12,6)"))
            .withColumn("active_days_per_customer",
                        (F.col("active_days") / F.col("customers"))
                        .cast("decimal(10,4)"))
            # Null rather than zero when a period has no orders: there were no
            # visits to average over, which is not the same as one order per
            # visit.
            .withColumn("orders_per_active_day",
                        F.when(F.col("active_days") > 0,
                               F.col("orders") / F.col("active_days"))
                        .cast("decimal(10,4)"))
            .withColumn("avg_order_value",
                        F.when(F.col("orders") > 0,
                               F.col("total_revenue") / F.col("orders"))
                        .cast("decimal(10,2)"))
            .withColumn("batch_date",
                        F.lit(batch_date.strftime("%Y-%m-%d")).cast("date"))
            .withColumn("updated_at", F.current_timestamp()))
