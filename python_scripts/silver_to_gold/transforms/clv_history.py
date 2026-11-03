"""The lifetime-value tiering, defined once for every table that cuts on it.

Three gold tables now need "which fifth of lifetime spend is this customer in":
`customer_lifetime_value` publishes it per customer, `clv_tier_profile` and
`clv_tier_segments` publish aggregates of it. A gold transform cannot read
another gold table -- `depends_on` resolves against silver in
`silver_read.read_silver_inputs` -- so all three have to recompute the cut, and
this module is the one place it is written down.

Same reasoning as `segment_history`: the alternative is three copies of a
threshold, where editing one and not the others publishes two different
definitions of "top fifth" and nothing fails.

TIES STAY TOGETHER, AND THAT MOVES THE SHARES OFF 20/60/20
----------------------------------------------------------
`percent_rank` gives every tied row the same value, so a tie that straddles a
boundary lands wholly on one side of it. Measured on the real data:

    Top     4,035 customers  20.00%  $75.94 and up    76.59% of lifetime spend
    Mid    12,499 customers  61.96%  $9.89 - $75.93   21.99%
    Bottom  3,640 customers  18.04%  $0.00 - $9.88     1.41%

The Bottom tier is 18.04% and not 20% because 475 customers have lifetime spend
of exactly $9.89 and all of them share `percent_rank` 0.796064, just inside the
Mid tier. That is the correct behaviour -- two customers who spent the same
amount must not land in different tiers -- but it means the realised shares
have to be published rather than assumed, which is why `pct_of_customers` is a
column on `clv_tier_profile`.

`ntile(5)` would force exact fifths by splitting those 475 customers at an
arbitrary point. That is the same trap the RFM quintile scoring already fell
into on order counts, and it is not worth exact bucket sizes.

THE THREE VOLUME ACCOUNTS ARE FLAGGED, NEVER DROPPED
----------------------------------------------------
Three accounts order at a rate no individual consumer plausibly sustains:

    orders   spend       avg order   span      orders/day
     2,124   $23,631.93     $11.13   1,311 d        1.62
     1,803   $27,258.73     $15.12   1,278 d        1.41
     1,000    $8,175.05      $8.18     418 d        2.39

1.4 to 2.4 orders every single day for a year or more reads as a corporate,
franchise or aggregator account rather than a person, so the tables publish
aggregates both ways and let the reader choose. The rows themselves are never
removed from `customer_lifetime_value`: a deleted row cannot be counted, and
"these three accounts hold 3.66% of lifetime revenue" is itself a finding.

`OUTLIER_ORDER_FLOOR` is a count rather than a percentile deliberately. A p99
rule on lifetime spend flags 202 customers holding 26.73% of revenue, 64 of
them current VIPs -- real best customers excluded for being good. Worse, a
1.5x IQR rule flags 2,464 customers (12.21%) holding 67.00% of revenue: spend
here is severely right-skewed, so IQR treats the entire healthy right tail as
anomalous. Neither is outlier detection on this distribution.

WHAT EXCLUDING THEM ACTUALLY CHANGES -- measured, so the toggle is honest:
  * lifetime revenue -3.66%, lifetime orders -4.15%, mean spend $79.93 ->
    $77.01, and the Top tier's share of spend 76.59% -> 75.70%.
  * the tier cuts do not move: the Top tier still starts at $75.94 and the
    Bottom tier still ends at $9.88 either way, and median lifetime spend
    across the whole base is $23.78 either way. The one median that does move
    is the Top tier's own, by a single cent ($158.24 -> $158.23) -- that is
    `percentile_approx` re-approximating over three fewer rows, not a real
    shift in the middle of the tier.
  * the trailing-90-day figures barely notice: 90-day revenue -0.044%, and
    only the Slipping segment changes at all (-2 customers, -0.44% revenue).
    Two of the three are Slipping at the anchor and the third went Dormant
    974 days ago, so it is already outside every active view.

So the concentration finding survives the exclusion, which is the point of
being able to test it.
"""

from __future__ import annotations

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

# The two cuts, as fractions of the ranked population. 0.0 is the highest
# spender, because the ranking below is descending.
TOP_PERCENTILE = 0.2
LOW_PERCENTILE = 0.8

# The project's one vocabulary for the tiers. `customer_lifetime_value` used
# to publish High/Medium/Low for the identical cut, so the warehouse briefly
# carried two sets of words for one concept -- worse than either. Everything
# now reads from here.
#
# Changing this tuple changes the VALUES in the published column
# `gold.customer_lifetime_value.clv_segment`, not just a label on a chart, so
# anything downstream matching on the old words has to change with it.
TIERS = ("Top", "Mid", "Bottom")

# Lifetime order count at or above which an account is treated as non-consumer.
# Set at 900 rather than a rounder 1,000 because the gap it sits in is wide:
# the ranked lifetime order counts run
#
#     2124, 1803, 1000, 799, 743, 725, 664, 630, 601, 548, 525, 523, ...
#
# so ANY floor from 800 to 1000 isolates exactly the same three accounts, and
# 900 is the middle of that gap -- the most stable point against a refresh
# nudging a count by one or two. A floor of exactly 1000 would sit on top of
# the third account's own value and drop it from the flag on any recount.
#
# This is a judgement about what counts as a customer, not a statistic. It is
# a named constant so the judgement is reviewable in one place.
OUTLIER_ORDER_FLOOR = 900

# The two readings every aggregate table publishes side by side. "included" is
# first so a consumer that takes the first group gets the unfiltered truth.
OUTLIER_VARIANTS = ("included", "excluded")


def lifetime(items: DataFrame) -> DataFrame:
    """One row per customer over all history.

    Aggregated from line items rather than from `segment_history.per_order`,
    deliberately: `line_items` has no meaning at order grain, and
    `last_order_date` would shift for any order whose lines straddle midnight
    UTC, since per_order dates an order by its FIRST line.

    Anonymous rows are dropped: a per-customer measure needs a customer.

    `orders_per_active_day` is published beside the flag so the judgement is
    auditable rather than asserted. Read it with care: a customer whose only
    order is a single day gets `active_days` 1 and therefore a rate of exactly
    1.0/day, so the column is meaningless at low order counts and the median
    over the whole base (1.0) is an artifact, not a baseline. It is
    interpretable only for customers with enough orders to have a rhythm --
    which is why the flag itself keys off the order COUNT and uses this column
    as evidence rather than as its rule.
    """
    identified = items.filter(
        F.col("user_id").isNotNull() & (F.trim(F.col("user_id")) != "")
    )
    return (
        identified
        .groupBy("user_id")
        .agg(
            F.sum("item_price").cast("decimal(18,2)")
             .alias("total_lifetime_revenue"),
            F.countDistinct("order_id").alias("total_orders"),
            F.count(F.lit(1)).alias("line_items"),
            F.sum("item_quantity").alias("units"),
            F.min(F.to_date(F.col("creation_time_utc")))
             .alias("first_order_date"),
            F.max(F.to_date(F.col("creation_time_utc")))
             .alias("last_order_date"),
        )
        .withColumn("avg_order_value",
                    (F.col("total_lifetime_revenue") / F.col("total_orders"))
                    .cast("decimal(10,2)"))
        # Inclusive of both endpoints, so a same-day customer spans 1 day and
        # the rate below never divides by zero.
        .withColumn("active_days",
                    F.datediff(F.col("last_order_date"),
                               F.col("first_order_date")) + F.lit(1))
        .withColumn("orders_per_active_day",
                    F.try_divide(F.col("total_orders"), F.col("active_days"))
                    .cast("decimal(10,4)"))
        .withColumn("is_volume_outlier",
                    F.col("total_orders") >= F.lit(OUTLIER_ORDER_FLOOR))
    )


def with_tier(per_customer: DataFrame,
              labels: tuple[str, str, str] = TIERS,
              column: str = "clv_tier") -> DataFrame:
    """Add `value_percentile` and the spend tier it falls in.

    Ranked descending, so `value_percentile` 0.0 is the biggest spender and the
    top fifth is the LOW end of the percentile range. That reads backwards on
    first encounter, which is exactly why the tier label is published beside it
    rather than left for each consumer to re-derive from the number.

    The window is deliberately unpartitioned -- a percentile of the whole
    customer base is the definition -- so Spark warns that it is moving all
    rows to one partition. At 20k customers that is the cheapest part of the
    job; at 20M it would need an approximate rank instead.

    THE VOLUME OUTLIERS ARE RANKED WITH EVERYONE ELSE, on purpose. Call this
    before `with_outlier_variants`, never after. The tier cut is one
    definition of "top fifth" over the whole customer base, and if the
    excluded variant re-ranked without those three accounts then
    `customer_lifetime_value.clv_segment` would disagree with
    `clv_tier_profile` for whichever variant a reader happened to pick. It
    costs nothing to keep them in: measured, the cuts are identical either way
    ($75.93 and $9.89), because three rows out of 20,174 cannot move a
    percentile.
    """
    by_value = Window.orderBy(F.col("total_lifetime_revenue").desc())
    top, mid, low = labels
    return (
        per_customer
        .withColumn("value_percentile", F.percent_rank().over(by_value))
        .withColumn(column,
                    F.when(F.col("value_percentile")
                           <= F.lit(TOP_PERCENTILE), top)
                     .when(F.col("value_percentile")
                           >= F.lit(LOW_PERCENTILE), low)
                     .otherwise(mid))
    )


def with_outlier_variants(per_customer: DataFrame) -> DataFrame:
    """Two tagged copies of the customer base: one whole, one without the
    volume outliers.

    This is what lets an aggregate table publish both readings without
    duplicating a single aggregation: add `outliers` to the GROUP BY and every
    measure computes twice, once over each population. The `excluded` copy
    simply has no row for a flagged customer, so

      * a count or a sum over it omits them by construction, and
      * an ORDER-grain join against it drops their orders too, with no second
        filter to keep in step -- their orders find no `excluded` row to match.

    That second property is the reason this returns a frame rather than a
    boolean filter: `clv_tier_profile` joins orders to this to get order-grain
    medians, and a separate order-level exclusion list is exactly the kind of
    thing that drifts out of agreement with the customer-level one.

    Call AFTER `with_tier`, so both variants carry the same tier labels.
    """
    included, excluded = OUTLIER_VARIANTS
    return (
        per_customer.withColumn("outliers", F.lit(included))
        .unionByName(
            per_customer.filter(~F.col("is_volume_outlier"))
            .withColumn("outliers", F.lit(excluded)))
    )
