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

# The segment thresholds are imported, not copied, so this table and
# `customer_rfm` cannot drift into disagreeing about what a VIP is. The segment
# CASE below is a second implementation of the one there, because that one
# anchors to the dataset's last order date and this table needs the same logic
# evaluated at two different anchors. Factoring both onto one helper is the
# obvious follow-up; it would mean editing customer_rfm, so it is not done here.
from python_scripts.silver_to_gold.transforms.customer_rfm import (
    ACTIVE_WINDOW_DAYS,
    LOYAL_MIN_ORDERS_90D,
    NEW_MAX_TENURE_DAYS,
    PERIOD_DAYS,
    REENGAGE_DORMANCY_DAYS,
    SLIPPING_MIN_ORDERS_90D,
    VIP_SPEND_PERCENTILE,
)

PRE_LOYALTY = "Pre-loyalty"
POST_LOYALTY = "Post-loyalty"

ALL_CUSTOMERS = "All customers"
ALL_GROUP = "All"

MEDIAN = 0.5


def _per_order(items: DataFrame) -> DataFrame:
    """One row per (user_id, order_id) with its date, value, units and flag.

    Mirrors `_per_order` in loyalty_order_profile: anonymous orders are dropped
    because a per-customer measure cannot be built without a customer, and an
    order is flagged if ANY of its lines is.
    """
    identified = items.filter(
        F.col("user_id").isNotNull() & (F.trim(F.col("user_id")) != "")
    )
    return (
        identified
        .groupBy("user_id", "order_id")
        .agg(
            F.min(F.to_date(F.col("creation_time_utc"))).alias("order_date"),   # oldest order date of customer
            F.sum("item_price").cast("decimal(18,2)").alias("order_value"),     # total order revenue of customer
            F.sum("item_quantity").alias("units"),                              #total quantity of user
            F.sum(F.coalesce(F.col("is_loyalty"), F.lit(False)).cast("int"))    # is user a loyalty member for ANY order
             .alias("_flagged_lines"),
        )
        .withColumn("is_loyalty_order", F.col("_flagged_lines") > 0)            #if user was at any point a loyalty member produce this comn as a boolean
        .drop("_flagged_lines")
    )


def _rfm_features(per_order: DataFrame, anchor: date) -> DataFrame:
    """Per-customer RFM features as of `anchor`, ignoring everything after it.

    The `order_date <= anchor` filter is what makes this usable at a past
    anchor. `customer_rfm` has no equivalent because its anchor is the last
    order date, where there is nothing later to exclude -- here, without it,
    customers who first ordered AFTER the program launched would be counted
    into the pre-loyalty base as Dormant.

    Bucket boundaries are copied from customer_rfm deliberately: the window is
    half-open, (anchor - 90, anchor], so a customer cannot land in two periods.
    """
    history = per_order.filter(F.col("order_date") <= F.lit(anchor))    # filter the orders further only using the anchor of orders past

    p1_start = anchor - timedelta(days=PERIOD_DAYS)                     # create a timedelta that period days set in the rmf_anchor function so both will operate ssuming the same period
    p2_start = anchor - timedelta(days=2 * PERIOD_DAYS)                 # create a time period twice the time of the anchor period atart, creating a second window to observe historical customer behavior to compare
    window_start = anchor - timedelta(days=ACTIVE_WINDOW_DAYS)          # This is the entire window from the anchored time period used to examine customer behavior

    in_p1 = F.col("order_date") > F.lit(p1_start)                       # Creates a filter that assigns the order to one of the given periods above
    in_p2 = ((F.col("order_date") <= F.lit(p1_start))                   # 
             & (F.col("order_date") > F.lit(p2_start)))
    in_p3 = ((F.col("order_date") <= F.lit(p2_start))
             & (F.col("order_date") > F.lit(window_start)))

    return (
        history # Take only the orders before the anchor date
        .groupBy("user_id")
        .agg(
            F.max("order_date").alias("last_order_date"), #Find the most recent order date
            F.min("order_date").alias("first_order_date"), # find the last order date
            F.sum(F.when(in_p1, 1).otherwise(0)).alias("orders_last_30d"), #sum a total of all orders in p1 (this label could be an f_string populated by a string literal of the actual anchor date but works for this example)
            F.sum(F.when(in_p2, 1).otherwise(0)).alias("orders_31_60d"), #sum a total of all orders in p2
            F.sum(F.when(in_p3, 1).otherwise(0)).alias("orders_61_90d"), #sum a total of all orders in p3
            
            F.sum(F.when(in_p1 | in_p2 | in_p3, F.col("order_value")) # only sum the orders that fall within the anchor lookback window
                   .otherwise(0)).cast("decimal(18,2)").alias("spend_last90d"), #label the column 
            
            F.min(F.when(in_p1, F.col("order_date"))).alias("_first_in_last_30d"), # flag the fitst order_date in the last 30 days from anchor 
            
            F.max(F.when(F.col("order_date") <= F.lit(window_start), # last order date in the lookback period
                         F.col("order_date"))).alias("_last_before_window"),
        )
        .withColumn("orders_last90d", F.col("orders_last_30d")
                    + F.col("orders_31_60d") + F.col("orders_61_90d")) # get the total number of orders from each period and add them here. 
        
        .withColumn("tenure_days",
                    F.datediff(F.lit(anchor), F.col("first_order_date")))
        
        .withColumn("is_active", F.col("orders_last90d") > 0)
        
        .withColumn("dormancy_days",                            # This is only used if there are NO orders in p1 and p2 so if there are orders in between it won't be considered dormant lines 203 - 211 confirm this
                    F.datediff(F.col("_first_in_last_30d"),     # Used to calculate is user is re-engaging so not worred about missing orders in between 
                               F.col("_last_before_window")))
        .drop("_first_in_last_30d", "_last_before_window")
    )


def _vip_cut(per_order: DataFrame, anchor: date) -> Decimal:
    """The 80th percentile of 90-day spend among ACTIVE customers at `anchor`.

    Collected to a Python scalar rather than left as a window function, which
    is the difference from customer_rfm: a value that has to classify TWO eras
    identically cannot be recomputed per era. One small action, like the anchor
    lookup in build().

    Falls back to zero when there is no active customer to measure -- only
    reachable on a degenerate batch, and harmless there because the VIP branch
    also demands two orders in the last 30 days and one before that.
    """
    active = _rfm_features(per_order, anchor).filter(F.col("is_active"))
    cut = active.agg(
        F.percentile_approx(F.col("spend_last90d"), VIP_SPEND_PERCENTILE) #The spend percentile is calculate using the top 20% of users orders. If frequency and spend amount are in the top 80% we consider this a VIP
    ).collect()[0][0]
    return Decimal(0) if cut is None else cut


def _segments(per_order: DataFrame, anchor: date, vip_cut: Decimal) -> DataFrame:
    """One row per customer with the segment they held at `anchor`.

    First match wins, same order as customer_rfm, because the branches overlap
    by design -- a VIP also satisfies Loyal, and Loyal also satisfies
    Promising.
    """
    return (
        _rfm_features(per_order, anchor)
        .withColumn("segment",
                    F.when(~F.col("is_active"), "Dormant") #If customer does not have orders in the last 90 days they cannot be sconsdered active
                     .when((F.col("tenure_days") <= F.lit(NEW_MAX_TENURE_DAYS))
                           & (F.col("orders_last_30d") >= 1), "New") #If they only have one order in the last 30 days and no other they are considered new customers
                     .when((F.col("orders_last_30d") >= 1) # If they have a order in the last 30 days bot no other orders in the lsat 90 days but an order before the 6 months ago they ar giving us another chance by reengaging
                           & (F.col("orders_31_60d") == 0) 
                           & (F.col("orders_61_90d") == 0)
                           & (F.col("dormancy_days")
                              >= F.lit(REENGAGE_DORMANCY_DAYS)),
                           "Re-engaging")
                     .when((F.col("orders_last_30d") >= 2) #If they spend more than 80% of our customers in the last 90 days and they have more than one order in prior 2 months we consider these VIP
                           & ((F.col("orders_31_60d")
                               + F.col("orders_61_90d")) >= 1)
                           & (F.col("spend_last90d") >= F.lit(vip_cut)), "VIP")
                     .when((F.col("orders_last_30d") >= 1) #If customers have ordered with us in the last 30 days and have a minimum of 4 orders in 90 days but don't meet minimum spend threshold of VIPs they are considered loyal
                           & (F.col("orders_last90d")
                              >= F.lit(LOYAL_MIN_ORDERS_90D)), "Loyal")
                     .when(F.col("orders_last_30d") >= 1, "Promising") #If user doesn't meet the above criteria (not new, loyal, dormant, or a vip)
                     .when(F.col("orders_last90d")
                           >= F.lit(SLIPPING_MIN_ORDERS_90D), "Slipping") #If the user has ordered with us more than the average customer in the last 90 days but hasn't ordered with us in the last 30 days then we are losing them
                     .otherwise("Lapsing")) # Customers are not high spenders, nor frequent users and they have not ordered with us in the last 30 days
        .withColumn("segment_group",
                    F.when(F.col("segment").isin("VIP", "Loyal"), "Core") #Categorizes the segments into segment groups and determins if user is a churn risk 
                     .when(F.col("segment") == "New", "New Growth")
                     .when(F.col("segment").isin("Promising", "Re-engaging"),
                           "Developing")
                     .when(F.col("segment").isin("Slipping", "Lapsing"),
                           "Churn Risk")
                     .otherwise("Dormant"))
        .select("user_id", "segment", "segment_group")
    )


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
        .join(_segments(per_order, end, vip_cut), on="user_id", how="inner")
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
    per_order = _per_order(inputs["order_items"])

    # One action to find both ends of the data. The launch date is derived, not
    # configured, so a later batch that extends the data moves both windows
    # together and they stay equal.
    bounds = per_order.agg(
        F.max("order_date").alias("last_order"),
        F.min(F.when(F.col("is_loyalty_order"), F.col("order_date")))
         .alias("launch"),
    ).collect()[0]

    last_order = bounds["last_order"]
    launch = bounds["launch"]

    # Degenerate batches, kept to the same shape rather than special-cased.
    # No orders at all: fall back to the batch date, as customer_rfm does. No
    # flagged order: put the launch one day past the end, which empties the
    # post era and leaves every order in the pre era -- an honest answer for a
    # batch that contains no program.
    if last_order is None:
        last_order = batch_date.date()
    if launch is None:
        launch = last_order + timedelta(days=1)

    era_days = max((last_order - launch).days + 1, 1)
    pre_end = launch - timedelta(days=1)
    pre_start = pre_end - timedelta(days=era_days - 1)

    # Frozen once, at the PRE anchor, then used for both eras. See the module
    # docstring: a floating percentile would make growth unmeasurable.
    vip_cut = _vip_cut(per_order, pre_end)

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
