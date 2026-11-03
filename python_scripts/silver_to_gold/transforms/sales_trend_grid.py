"""Revenue on a clock: the only gold table that knows what time of day it is.

Grain is (sales_date_local, restaurant_id, item_category, hour_local) -- one
row per store, per menu category, per hour of the local day. 126,440 rows on
the current data, which is SMALLER than the silver fact table it is built
from, because a line item is the only thing finer than this cell.

Everything a sales-trend view needs rolls up from here: revenue by day, week
or month, split by location, by category, by time of day or by none of them,
plus the day-of-week by hour profile that resource planning actually asks
for. One table rather than four, so four readings cannot disagree on a total.

EVERY MEASURE IN THIS TABLE IS ADDITIVE
---------------------------------------
`orders`, `line_items`, `units` and `total_revenue` all SUM correctly over any
set of rows -- across hours, categories, stores, days, any combination.


Here each order is pinned to ONE cell -- the cell of its single largest line item 
-- and `orders` counts those
pins. So `orders` on a category row reads "orders whose biggest item was in
this category", and summing every category returns the true distinct order
count. `line_items` answers the other question: how many things were sold.

TIME OF DAY IS LOCAL TIME, AND THAT IS THE WHOLE POINT
------------------------------------------------------
The pipeline stores and dates everything in UTC deliberately, and every other
gold table is UTC. This one is not, because an hour-of-day column in UTC is
not an answer to any question anybody asks. 


America/New_York is not a guess: every row is USD and every row is one brand.
`from_utc_timestamp` resolves EST against EDT per instant, so a fixed -5
offset cannot be an hour wrong for half of each year.

WHY THIS WILL NOT TIE EXACTLY TO daily_restaurant_sales
-------------------------------------------------------
Two reasons, both measured, both small, both worth knowing before someone
reconciles the two by hand:

  * The local calendar day. An order at 2024-02-21 02:52 UTC is the evening of
    2024-02-20 in New York. $22,009.75 of revenue -- 1.24% -- sits on a
    different day in the two tables. `sales_date_utc` is published beside the
    local date so a reader can tie them together rather than wonder.
  * `NON_CATEGORIES`, imported from `category_period_sales` rather than
    redefined, because three tables disagreeing about what a menu category is
    would be worse than any one of them being wrong. It costs $35.57 across 10
    line items: "Test Items", and the "Most Popular" pseudo-category that is a
    merchandising shelf rather than a section of the menu.

Nothing else is filtered. In particular the `Alltown Fresh - DEVELOPMENT` app
is INCLUDED -- $7,221.54, 0.41% of revenue, 826 line items -- because every
other gold table includes it, and one table quietly disagreeing with the rest
of the warehouse is the worse problem. It is worth filtering, in silver and
everywhere at once, which is a separate decision from this table.


THE OUTLIER TOGGLE IS FOUR COLUMNS, NOT TWICE THE ROWS
------------------------------------------------------
The dashboard can exclude the three volume accounts everywhere, so this table
has to support it too. The CLV tables do that by carrying `outliers` as a
grain key and publishing both populations -- which they must, because a
published median cannot be un-aggregated.

FULL REBUILD, NO event_time_column
----------------------------------
An incremental run finds which UTC event dates a batch touched and replaces
those. This table is keyed on the LOCAL date, so a UTC-dated scope would miss
the four or five hours of each local day that belong to the previous UTC date
and would replace them under the wrong key. 
"""

from __future__ import annotations

from datetime import datetime

from pyspark.sql import Column, DataFrame, Window
from pyspark.sql import functions as F

from python_scripts.silver_to_gold.transforms import clv_history
from python_scripts.silver_to_gold.transforms.category_period_sales import (
    NON_CATEGORIES,
)

# The business runs on Eastern time. A zone NAME rather than an offset, so
# `from_utc_timestamp` applies EST or EDT per instant instead of being an hour
# out for half of every year.
LOCAL_ZONE = "America/New_York"

# Matched on the app name rather than the restaurant id, even though today it
# resolves to exactly one store. The id is an accident of this extract; the
# app name is what actually says "this is the test environment", and a second
# dev store appearing later should be caught rather than missed.
DEVELOPMENT_APP_PATTERN = "%DEVELOPMENT%"

# Hour buckets as (label, first_hour, last_hour), both inclusive, covering all
# 24 hours with no gap. Named for the clock rather than for meals: the
# measured revenue shares are Overnight 5.46%, Morning 30.22%, Midday 51.69%,
# Afternoon 10.67% and Evening 1.95%, and calling the 10am-1pm block "Lunch"
# would assert something about intent that the data does not say.
#
# Evening at 1.95% is itself a finding rather than a bucket that happens to be
# small: these are daytime locations, so an evening shift is not where the
# demand is, and the grid will show that per store.
DAYPARTS = (
    ("Overnight", 0, 5),
    ("Morning", 6, 9),
    ("Midday", 10, 13),
    ("Afternoon", 14, 17),
    ("Evening", 18, 23),
)

# The grain, so the group-by and the config cannot drift apart.
_GRAIN = ["sales_date_local", "restaurant_id", "item_category", "hour_local"]


def daypart_of(hour: Column) -> Column:
    """`DAYPARTS` as one CASE expression over an hour-of-day column.

    Built from the constant rather than written out, so the labels, the bucket
    edges and this expression cannot disagree. Public because the dashboard
    needs the same ordering and the tests assert the edges.
    """
    expr = F.lit(None).cast("string")
    for label, first, last in reversed(DAYPARTS):
        expr = F.when(
            (hour >= F.lit(first)) & (hour <= F.lit(last)), F.lit(label)
        ).otherwise(expr)
    return expr


def build(inputs: dict[str, DataFrame], batch_date: datetime) -> DataFrame:
    items = inputs["order_items"]
    calendar = inputs["date_dim"]


    is_non_category = F.coalesce(
        F.col("item_category").isin(*NON_CATEGORIES), F.lit(False))

    lines = (
        items
        .filter(~is_non_category)
        .withColumn("_ts_local",
                    F.from_utc_timestamp(F.col("creation_time_utc"),
                                         LOCAL_ZONE))
        .withColumn("sales_date_local", F.to_date(F.col("_ts_local")))
        .withColumn("hour_local", F.hour(F.col("_ts_local")))
        .withColumn("_line_value", F.col("item_price").cast("decimal(10,2)"))
    )

    # The volume-outlier flag, recomputed from silver through the shared rule.
    # `lifetime` is given the UNFILTERED items on purpose: whether an account
    # is a volume account is a property of everything it ever ordered, so
    # dropping the pseudo-categories first could in principle move a customer
    # across the floor and make this table disagree with the CLV tables.
    flagged = (clv_history.lifetime(items)
               .filter(F.col("is_volume_outlier"))
               .select("user_id")
               .withColumn("_flagged", F.lit(True)))

    # LEFT, then coalesced: an anonymous line has no user_id and so no flag,
    # and `NULL & anything` would make its contribution NULL rather than zero.
    lines = (lines.join(flagged, on="user_id", how="left")
             .withColumn("_outlier",
                         F.coalesce(F.col("_flagged"), F.lit(False))))

    # Pin every order to exactly one cell, which is what keeps `orders`
    # additive. Ordered by line value first -- "the order's main item" is the
    # sensible reading -- then by time, then by id purely to make it
    # deterministic, so a rebuild cannot silently move an order from one
    # category to another.
    #
    # This pins a CELL and not just a category on purpose. An order whose
    # lines straddle an hour boundary would otherwise be counted once in each
    # hour, and additivity would hold across categories but break across
    # hours, which is the harder bug to see.
    main_line_first = Window.partitionBy("order_id").orderBy(
        F.col("_line_value").desc_nulls_last(),
        F.col("_ts_local").asc_nulls_last(),
        F.col("lineitem_id").asc_nulls_last(),
    )

    grid = (
        lines
        .withColumn("_line_rank", F.row_number().over(main_line_first))
        .groupBy(*_GRAIN)
        .agg(
            F.sum(F.when(F.col("_line_rank") == 1, 1).otherwise(0))
             .cast("bigint").alias("orders"),
            F.count(F.lit(1)).cast("bigint").alias("line_items"),
            F.sum("item_quantity").cast("bigint").alias("units"),
            F.sum("item_price").cast("decimal(18,2)").alias("total_revenue"),
            # Functionally determined by the grain -- one local hour falls
            # inside exactly one UTC date -- so min() here is only a way to
            # carry the value through the group-by, not a choice between
            # candidates. Published as the bridge to the UTC-dated tables.
            F.min(F.to_date(F.col("creation_time_utc")))
             .alias("sales_date_utc"),
            # The flagged accounts' share OF THIS CELL, so the dashboard's
            # exclusion is a subtraction at whatever grain it has rolled up
            # to. Coalesced to zero AFTER the sum, never `.otherwise(0)`
            # inside the `when`: that unifies the two branches against an int
            # literal, and under ANSI mode a decimal or string price then
            # fails the cast outright rather than summing. A NULL here would
            # propagate through the subtraction and blank the measure instead
            # of leaving it unchanged, so zero is the only safe empty.
            F.coalesce(
                F.sum(F.when(F.col("_outlier") & (F.col("_line_rank") == 1),
                             F.lit(1))),
                F.lit(0)).cast("bigint").alias("outlier_orders"),
            F.coalesce(F.sum(F.when(F.col("_outlier"), F.lit(1))), F.lit(0))
             .cast("bigint").alias("outlier_line_items"),
            F.coalesce(F.sum(F.when(F.col("_outlier"),
                                    F.col("item_quantity"))), F.lit(0))
             .cast("bigint").alias("outlier_units"),
            F.coalesce(F.sum(F.when(F.col("_outlier"),
                                    F.col("_line_value"))), F.lit(0))
             .cast("decimal(18,2)").alias("outlier_revenue"),
            # ANY line, not every line: a cell mixing real and test traffic
            # is still not a cell to plan a shift from. Today the dev app is
            # one whole store so the distinction is moot, which is exactly
            # why the stricter reading is the one to publish.
            F.max(F.coalesce(
                F.upper(F.col("app_name")).like(DEVELOPMENT_APP_PATTERN),
                F.lit(False))).alias("is_development"),
        )
    )

    return (
        grid
        # LEFT, where category_holiday_sales joins the calendar INNER. The
        # trade is deliberate and runs the other way here: that table needs a
        # day it can call ordinary, while this one's contract is that its
        # revenue sums to the warehouse's. An inner join would drop real sales
        # on a date the calendar happens not to cover. A short calendar shows
        # up here as NULL flags, which is visible, rather than as missing
        # money, which is not.
        .join(calendar.select(F.col("date_key").alias("sales_date_local"),
                              "day_of_week", "is_weekend", "is_holiday",
                              "holiday_name"),
              on="sales_date_local", how="left")
        .withColumn("daypart", daypart_of(F.col("hour_local")))
        # ISO numbering, Monday = 1. Derived here rather than taken from the
        # calendar so the weekday axis still sorts if date_dim is ever short,
        # and because Spark's own dayofweek starts on Sunday -- sorting a
        # chart on the day NAME would read Friday, Monday, Saturday.
        .withColumn("dow_number",
                    ((F.dayofweek(F.col("sales_date_local")) + 5) % 7 + 1)
                    .cast("int"))
        .withColumn("week_start", F.trunc(F.col("sales_date_local"), "week"))
        .withColumn("month_start", F.trunc(F.col("sales_date_local"), "month"))
        .withColumn("batch_date",
                    F.lit(batch_date.strftime("%Y-%m-%d")).cast("date"))
        .withColumn("updated_at", F.current_timestamp())
    )
