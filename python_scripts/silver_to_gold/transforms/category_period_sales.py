"""Weekly, monthly and seasonal sales per menu category, in one table.

"""

from __future__ import annotations

from datetime import datetime
from functools import reduce

from pyspark.sql import Column, DataFrame, Window
from pyspark.sql import functions as F

# Normalising on days rather than on counted calendar weeks or months keeps the
# rate exact for a part-covered period, which is the whole point of having it.
# The cost is that on a Month row avg_orders_per_month differs from
# total_orders by up to 2% -- the amount by which a 28- or 31-day month differs
# from the mean Gregorian month of 365.25/12 days.
DAYS_PER_WEEK = 7.0
DAYS_PER_MONTH = 30.4375

# Meteorological seasons, which are whole calendar months and so nest exactly
# inside the Month grain. Astronomical seasons would start on a solstice or
# equinox that moves by a day or two each year and would split every boundary
# month in two.
WINTER_MONTHS = (12, 1, 2)
SPRING_MONTHS = (3, 4, 5)
SUMMER_MONTHS = (6, 7, 8)

# Values the source stores in item_category that are not menu categories:
# `Test Items` is test data that reached production, `Most Popular` is a
# merchandising tag. Excluded from category analysis, because a category chart
# with a row that is not a category is wrong at the source rather than in the
# dashboard -- gold is what gets queried.
#
# The cost is a deliberate reconciliation gap: 10 order lines and $35.57 of
# $1,778,729.14 (0.002%) do not appear in this table or in
# category_holiday_sales, so neither ties exactly to daily_restaurant_sales.
# Textual corruption in this column is repaired upstream instead, by
# silver_cleaners in config -- these two are judgements about scope, not
# repairs, which is why they live here.
NON_CATEGORIES = ("Test Items", "Most Popular")

# A category below this share of all-time sales is immaterial to the business.
# 1% is not arbitrary: it keeps 8 categories, which is also the ceiling on
# distinguishable categorical colours, and leaves 4.08% of sales in the tail.
# The next meaningful cut is 5 categories -- nothing at all sits between
# 1.67% and 7.75%, so every threshold from 2% to 5% gives the same answer, and
# that one discards Meal Prep Kits, Specialty Coffee Beverages and Drip
# Coffee, which are real lines of business rather than noise.
MATERIALITY_THRESHOLD_PCT = 1.0


def category_materiality(daily: DataFrame) -> DataFrame:
    """Each category's share of ALL-TIME sales, and whether it clears the bar.

    One row per category. Public because category_holiday_sales imports it:
    the two tables must not disagree about what is material, for the same
    reason they share NON_CATEGORIES.

    The share is deliberately global rather than per period. Measured per
    period it flickers -- 15 categories cross 1% in some months and not
    others, and `Burgers & Sandwiches` ranges from 0.15% to 73.42% across the
    five months it appears in -- so a per-period flag would move a category in
    and out of the tail as the reader changed the date range. Safe to compute
    here because both category tables are write: replace, so the transform
    always sees the whole of silver.

    Both columns are published, not just the boolean: a flag throws away the
    number, and a consumer may want to re-threshold without recomputing.
    """
    totals = (daily
              .groupBy("item_category")
              .agg(F.sum("item_price").alias("_category_sales")))

    # Unpartitioned on purpose -- the grand total is over every category, and
    # this frame is one row per category, a few dozen rows. Spark's
    # single-partition warning is the right trade here.
    grand_total = F.sum("_category_sales").over(Window.partitionBy())

    # TRY_DIVIDE and not `/`: ANSI mode is on in Spark 4, and a batch with no
    # sales at all would otherwise raise instead of returning null.
    share = F.try_divide(F.lit(100) * F.col("_category_sales"), grand_total)

    return (totals
            .withColumn("_share", share)
            .withColumn("pct_of_total_sales",
                        F.col("_share").cast("decimal(7,4)"))
            # Coalesced so a null share on a degenerate batch reads as "not
            # material" rather than as null, which would be a third state no
            # consumer expects from a boolean.
            .withColumn("is_material",
                        F.coalesce(
                            F.col("_share")
                            >= F.lit(MATERIALITY_THRESHOLD_PCT),
                            F.lit(False)))
            .select("item_category", "pct_of_total_sales", "is_material"))


def _season(date_col: Column) -> Column:
    """Which season a date falls in."""
    month = F.month(date_col)
    return (F.when(month.isin(*WINTER_MONTHS), "Winter")
             .when(month.isin(*SPRING_MONTHS), "Spring")
             .when(month.isin(*SUMMER_MONTHS), "Summer")
             .otherwise("Fall"))


def _season_year(date_col: Column) -> Column:
    """The year a season is filed under.
    """
    return F.when(F.month(date_col) == 12, F.year(date_col) + 1) \
            .otherwise(F.year(date_col))


def _season_start(date_col: Column) -> Column:
    """First day of the season containing this date."""
    year = _season_year(date_col)
    month = F.month(date_col)
    return (F.when(month.isin(*WINTER_MONTHS),
                   F.make_date(year - 1, F.lit(12), F.lit(1)))
             .when(month.isin(*SPRING_MONTHS),
                   F.make_date(year, F.lit(3), F.lit(1)))
             .when(month.isin(*SUMMER_MONTHS),
                   F.make_date(year, F.lit(6), F.lit(1)))
             .otherwise(F.make_date(year, F.lit(9), F.lit(1))))


def _periods(date_col: Column) -> list[tuple[str, Column, Column]]:
    """(period_type, first day, last day) for each grain this table carries.
    """
    week = F.trunc(date_col, "week")              # Monday
    month = F.trunc(date_col, "month")
    season = _season_start(date_col)
    return [
        ("Week", week, F.date_add(week, 6)),
        ("Month", month, F.last_day(month)),
        ("Season", season, F.last_day(F.add_months(season, 2))),
    ]


def _with_every_period(df: DataFrame, date_column: str) -> DataFrame:
    """Repeat each row once per grain, tagged with the period it belongs to.
    """
    frames = [
        df.withColumn("period_type", F.lit(name))
          .withColumn("period_start", start)
          .withColumn("period_end", end)
        for name, start, end in _periods(F.col(date_column))
    ]
    return reduce(DataFrame.unionByName, frames)


def build(inputs: dict[str, DataFrame], batch_date: datetime) -> DataFrame:
    items = inputs["order_items"]
    calendar = inputs["date_dim"]

    # Coalesced because isin returns NULL for a NULL category and `~NULL` is
    # NULL, which a filter drops -- so a batch with a missing category would
    # lose rows silently. There are none today; order_items is incremental.
    is_non_category = F.coalesce(
        F.col("item_category").isin(*NON_CATEGORIES), F.lit(False))

    daily = (items
             .filter(~is_non_category)
             .withColumn("sales_date",
                         F.to_date(F.col("creation_time_utc"))))

    span = daily.agg(F.min("sales_date"), F.max("sales_date")).collect()[0]
    span_start, span_end = span[0], span[1]
    if span_start is None:
        span_start = span_end = batch_date.date()

    sales = (
        _with_every_period(daily, "sales_date")
        .groupBy("period_type", "period_start", "period_end", "item_category")
        .agg(
            F.countDistinct("order_id").alias("total_orders"),
            F.count(F.lit(1)).alias("total_order_items"),
            F.sum("item_price").cast("decimal(18,2)").alias("total_sales"),
            F.countDistinct("sales_date").cast("int").alias("days_with_sales"),
        )
    )
    
    holidays = (
        _with_every_period(calendar, "date_key")
        .groupBy("period_type", "period_start")
        .agg(
            F.sum(F.col("is_holiday").cast("int")).cast("int")
             .alias("holiday_days"),
            F.sum(F.col("is_weekend").cast("int")).cast("int")
             .alias("weekend_days"),
        )
    )

    period_days = F.datediff(
        F.least(F.col("period_end"), F.lit(span_end)),
        F.greatest(F.col("period_start"), F.lit(span_start)),
    ) + 1

    return (
        sales
        .join(holidays, on=["period_type", "period_start"], how="left")
        # Left, though every category here is necessarily in `materiality` --
        # both derive from `daily`. An inner join would hide a future bug by
        # silently dropping period rows instead of showing a null share.
        .join(category_materiality(daily), on="item_category", how="left")
        .withColumn("period_days", period_days)
        .withColumn("season", _season(F.col("period_start")))
        .withColumn("season_year", _season_year(F.col("period_start")))
        # A period with no calendar coverage reads as no holidays known rather
        # than null, so a dashboard filter does not drop the row entirely.
        .withColumn("holiday_days", F.coalesce(F.col("holiday_days"), F.lit(0)))
        .withColumn("weekend_days", F.coalesce(F.col("weekend_days"), F.lit(0)))
        .withColumn("is_holiday_week",
                    F.when(F.col("period_type") == "Week",
                           F.col("holiday_days") > 0))
        .withColumn("avg_sales_per_order",
                    (F.col("total_sales") / F.col("total_orders"))
                    .cast("decimal(18,2)"))
        .withColumn("avg_order_items",
                    (F.col("total_order_items") / F.col("total_orders"))
                    .cast("decimal(10,2)"))
        .withColumn("avg_revenue_per_day",
                    (F.col("total_sales") / F.col("period_days"))
                    .cast("decimal(18,2)"))
        .withColumn("avg_orders_per_week",
                    (F.col("total_orders") * F.lit(DAYS_PER_WEEK)
                     / F.col("period_days")).cast("decimal(10,2)"))
        .withColumn("avg_orders_per_month",
                    (F.col("total_orders") * F.lit(DAYS_PER_MONTH)
                     / F.col("period_days")).cast("decimal(10,2)"))
        .withColumn("batch_date",
                    F.lit(batch_date.strftime("%Y-%m-%d")).cast("date"))
        .withColumn("updated_at", F.current_timestamp())
    )
