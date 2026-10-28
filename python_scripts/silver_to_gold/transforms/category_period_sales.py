"""Weekly, monthly and seasonal sales per menu category, in one table.

"""

from __future__ import annotations

from datetime import datetime
from functools import reduce

from pyspark.sql import Column, DataFrame
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

    daily = items.withColumn("sales_date", F.to_date(F.col("creation_time_utc")))

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
