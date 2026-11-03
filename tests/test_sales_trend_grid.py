"""`sales_trend_grid`: local time, and the additivity of `orders`.

Two properties carry this table and neither is visible by reading a chart, so
both are tested rather than documented alone.

The first is that time of day is America/New_York. Every other gold table is
UTC, so the natural edit -- "make this consistent with the rest" -- would move
the revenue peak five hours and nothing else in the suite would notice.

The second is that `orders` sums correctly. A distinct count per cell is the
obvious implementation and it is wrong in two different ways at once: it
double-counts an order across its categories, and again across hours if the
order's lines straddle one. Both are tested, because a dashboard that sums
this column has no way to tell.
"""

from __future__ import annotations

import pytest
from pyspark.sql import functions as F

from python_scripts.silver_to_gold.transforms import clv_history, sales_trend_grid

pytestmark = pytest.mark.spark


def _line(order_id, utc_ts, *, category="Side", price="9.99", lineitem="1",
          restaurant="r1", qty=1, user="u1", app="app"):
    return {"ORDER_ID": f"'{order_id}'", "LINEITEM_ID": f"'{lineitem}'",
            "RESTAURANT_ID": f"'{restaurant}'",
            "CREATION_TIME_UTC": f"'{utc_ts}'",
            "ITEM_CATEGORY": f"'{category}'", "ITEM_PRICE": f"'{price}'",
            "ITEM_QUANTITY": str(qty), "IS_LOYALTY": "false",
            "APP_NAME": f"'{app}'",
            # A typed NULL, not a bare one: conftest builds a SQL VALUES
            # list, so `None` is not a literal and an untyped NULL is Spark's
            # `void` type.
            "USER_ID": "CAST(NULL AS string)" if user is None
                       else f"'{user}'"}


def _calendar(dim_rows, *dates):
    return dim_rows(*[{"date_key": f"CAST('{d}' AS date)"} for d in dates])


@pytest.fixture
def build(dim_rows, batch_date):
    """Run the transform, with a calendar covering whatever dates are named."""
    def run(items, *calendar_dates):
        calendar = _calendar(dim_rows, *(calendar_dates or ("2020-04-21",)))
        return sales_trend_grid.build(
            {"order_items": items, "date_dim": calendar}, batch_date)
    return run


# --- local time -----------------------------------------------------------

def test_hour_of_day_is_local_not_utc(item_rows, build):
    """18:56 UTC in July is 14:56 in New York, and 14 is the answer.

    The assertion against 18 is the point of the test: an implementation that
    read the hour straight off `creation_time_utc` would pass every other
    test in this file.
    """
    out = build(item_rows(_line("o1", "2020-07-15T18:56:44")),
                "2020-07-15").collect()

    assert len(out) == 1
    assert out[0].hour_local == 14
    assert out[0].hour_local != 18
    assert str(out[0].sales_date_local) == "2020-07-15"


def test_a_small_hours_utc_order_belongs_to_the_previous_local_day(item_rows,
                                                                   build):
    """The real last order in silver: 2024-02-21 02:52 UTC is the 20th here.

    `sales_date_utc` is published alongside precisely so this row can be
    reconciled against the UTC-dated tables instead of looking like a loss.
    """
    out = build(item_rows(_line("o1", "2024-02-21T02:52:35")),
                "2024-02-20").collect()

    assert str(out[0].sales_date_local) == "2024-02-20"
    assert out[0].hour_local == 21
    assert str(out[0].sales_date_utc) == "2024-02-21"


def test_the_offset_follows_dst_rather_than_being_fixed(item_rows, build):
    """Same UTC hour, two seasons, two different local hours.

    A hard-coded -5 would put both at 11:00 and be an hour out for the summer
    half of every year. `from_utc_timestamp` resolves EST against EDT per
    instant, which is why LOCAL_ZONE is a zone name and not an offset.
    """
    out = build(item_rows(_line("o_win", "2024-01-15T16:00:00"),
                          _line("o_sum", "2024-07-15T16:00:00")),
                "2024-01-15", "2024-07-15").collect()

    by_month = {str(r.sales_date_local)[:7]: r.hour_local for r in out}
    assert by_month == {"2024-01": 11, "2024-07": 12}


# --- the additivity contract on `orders` ----------------------------------

def test_orders_sums_to_the_distinct_count_across_categories(item_rows, build):
    """One order spanning two categories is ONE order, not two.

    This is the trap `category_period_sales.total_orders` carries: a distinct
    count per cell is right per row and wrong the moment anything sums it.
    """
    out = build(item_rows(
        _line("o1", "2020-04-21T16:00:00", category="Sandwiches",
              price="12.00", lineitem="1"),
        _line("o1", "2020-04-21T16:00:00", category="Drip Coffee",
              price="2.50", lineitem="2"),
    ), "2020-04-21").collect()

    assert len(out) == 2, "two categories should be two cells"
    assert sum(r.orders for r in out) == 1

    # And the order is pinned to its largest line, so the single counted row
    # is the sandwich rather than whichever cell Spark happened to emit first.
    pinned = [r.item_category for r in out if r.orders == 1]
    assert pinned == ["Sandwiches"]


def test_orders_stays_additive_when_an_order_straddles_an_hour(item_rows,
                                                               build):
    """Lines either side of an hour boundary are still one order.

    `creation_time_utc` is per line item, so this really happens. Pinning the
    order to a CELL rather than to a category is what covers it -- pinning to
    a category alone would keep the count additive across categories and
    break it across hours.
    """
    out = build(item_rows(
        _line("o1", "2020-04-21T15:59:00", price="12.00", lineitem="1"),
        _line("o1", "2020-04-21T16:01:00", price="2.50", lineitem="2"),
    ), "2020-04-21").collect()

    assert sorted(r.hour_local for r in out) == [11, 12]
    assert sum(r.orders for r in out) == 1


def test_every_measure_sums_to_the_input(item_rows, build):
    """Revenue, line items and units are additive too, and nothing is lost."""
    out = build(item_rows(
        _line("o1", "2020-04-21T16:00:00", price="10.00", qty=2,
              lineitem="1"),
        _line("o1", "2020-04-21T17:00:00", price="5.00", qty=1,
              lineitem="2"),
        _line("o2", "2020-04-21T16:00:00", price="1.25", qty=3,
              restaurant="r2", lineitem="1"),
    ), "2020-04-21").collect()

    assert sum(r.orders for r in out) == 2
    assert sum(r.line_items for r in out) == 3
    assert sum(r.units for r in out) == 6
    assert float(sum(r.total_revenue for r in out)) == pytest.approx(16.25)


# --- the shared category definition ---------------------------------------

def test_non_categories_are_excluded(item_rows, build):
    """`NON_CATEGORIES` comes from category_period_sales, not a second copy.

    "Most Popular" is a merchandising shelf that overlaps real sections, so
    leaving it in would describe a menu category that does not exist.
    """
    out = build(item_rows(
        _line("o1", "2020-04-21T16:00:00", category="Side", price="9.99"),
        _line("o2", "2020-04-21T16:00:00", category="Most Popular",
              price="4.00"),
        _line("o3", "2020-04-21T16:00:00", category="Test Items",
              price="1.00"),
    ), "2020-04-21").collect()

    assert [r.item_category for r in out] == ["Side"]
    assert float(out[0].total_revenue) == pytest.approx(9.99)


# --- the derived calendar columns -----------------------------------------

def test_daypart_covers_every_hour_with_inclusive_edges(spark):
    """All 24 hours get a label, and the boundary hours fall where declared.

    Tested against DAYPARTS itself rather than against written-out numbers,
    so the constant stays the single definition.
    """
    labelled = (spark.range(0, 24)
                .withColumn("daypart",
                            sales_trend_grid.daypart_of(F.col("id")))
                .collect())
    got = {row.id: row.daypart for row in labelled}

    assert None not in got.values(), f"unlabelled hours: {got}"
    for label, first, last in sales_trend_grid.DAYPARTS:
        for hour in range(first, last + 1):
            assert got[hour] == label, f"hour {hour} is {got[hour]}, not {label}"


def test_dow_number_is_iso_with_monday_first(item_rows, build):
    """Spark's own dayofweek starts on Sunday, which sorts a chart wrongly."""
    out = build(item_rows(
        _line("o_mon", "2024-01-01T17:00:00", lineitem="1"),
        _line("o_sun", "2024-01-07T17:00:00", lineitem="2"),
    ), "2024-01-01", "2024-01-07").collect()

    numbers = {str(r.sales_date_local): r.dow_number for r in out}
    assert numbers == {"2024-01-01": 1, "2024-01-07": 7}


def test_a_date_absent_from_the_calendar_keeps_its_revenue(item_rows, build):
    """The calendar join is LEFT, so a short date_dim loses flags, not money.

    category_holiday_sales joins INNER for a good reason of its own. This
    table trades the other way: its contract is that revenue sums to the
    warehouse's, so a missing calendar row has to be visible as a NULL flag
    rather than as quietly absent sales.
    """
    out = build(item_rows(_line("o1", "2020-06-10T16:00:00")),
                "2020-04-21").collect()

    assert len(out) == 1
    assert float(out[0].total_revenue) == pytest.approx(9.99)
    assert out[0].day_of_week is None
    assert out[0].is_weekend is None
    # Derived locally, so the weekday axis still works without the calendar.
    assert out[0].dow_number == 3


# --- the volume-outlier portion -------------------------------------------

@pytest.fixture
def low_floor(monkeypatch):
    """Drop the volume-outlier floor so a three-row fixture can cross it.

    The real floor is 900 lifetime orders. Patching the constant tests the
    wiring -- that this table asks `clv_history` who is flagged rather than
    keeping its own copy of the rule -- without needing 900-row fixtures.
    Same approach as test_clv_history.
    """
    monkeypatch.setattr(clv_history, "OUTLIER_ORDER_FLOOR", 2)
    return 2


def test_the_outlier_portion_is_the_flagged_accounts_share(item_rows, build,
                                                           low_floor):
    """One cell, two customers, and the portion names the flagged one's part.

    Subtracting gives the excluded reading, which is the whole reason these
    columns exist: the dashboard's toggle is `measure - outlier_measure` at
    whatever grain it has rolled up to.
    """
    out = build(item_rows(
        _line("o1", "2020-04-21T12:00:00", user="u_big", price="10.00"),
        _line("o2", "2020-04-21T12:30:00", user="u_big", price="10.00"),
        _line("o3", "2020-04-21T12:15:00", user="u_small", price="5.00"),
    ), "2020-04-21").collect()

    assert len(out) == 1, "all three lines belong to one cell"
    cell = out[0]
    assert (cell.orders, float(cell.total_revenue)) == (3, 25.00)
    assert cell.outlier_orders == 2
    assert float(cell.outlier_revenue) == pytest.approx(20.00)
    assert cell.outlier_line_items == 2
    assert cell.outlier_units == 2

    # The subtraction the dashboard performs.
    assert cell.orders - cell.outlier_orders == 1
    assert float(cell.total_revenue - cell.outlier_revenue) == pytest.approx(5.00)


def test_a_cell_with_no_flagged_rows_reports_zero_not_null(item_rows, build,
                                                           low_floor):
    """NULL would propagate through the subtraction and blank the measure.

    That is the failure mode worth a test rather than a comment: a cell with
    no flagged customer must leave its measures untouched, not erase them.
    """
    out = build(item_rows(
        _line("o1", "2020-04-21T12:00:00", user="u_small", price="5.00"),
    ), "2020-04-21").collect()

    cell = out[0]
    assert cell.outlier_orders == 0
    assert cell.outlier_line_items == 0
    assert cell.outlier_units == 0
    assert float(cell.outlier_revenue) == pytest.approx(0.0)
    assert float(cell.total_revenue - cell.outlier_revenue) == pytest.approx(5.00)


def test_an_anonymous_line_is_never_part_of_the_outlier_portion(item_rows,
                                                                build,
                                                                low_floor):
    """No user_id means no flag, and the LEFT join must not make it NULL.

    `clv_history.lifetime` drops anonymous rows, so an unidentified line can
    never be attributed to a flagged account -- but it still has to carry its
    own revenue into the cell.
    """
    out = build(item_rows(
        _line("o1", "2020-04-21T12:00:00", user=None, price="7.00"),
    ), "2020-04-21").collect()

    cell = out[0]
    assert float(cell.total_revenue) == pytest.approx(7.00)
    assert cell.outlier_orders == 0
    assert float(cell.outlier_revenue) == pytest.approx(0.0)


# --- the development-traffic flag -----------------------------------------

def test_development_traffic_is_flagged_not_dropped(item_rows, build):
    """Flagged so the page can drop it; kept so the totals still tie.

    The dev app is one whole store that peaks at 1am, so on a per-location
    staffing chart it reads as a genuine 24-hour site. Filtering it here
    instead would make this the only gold table that disagrees with the rest
    of the warehouse about total revenue.
    """
    out = build(item_rows(
        _line("o1", "2020-04-21T16:00:00", restaurant="r_real",
              app="Alltown Fresh"),
        _line("o2", "2020-04-21T16:00:00", restaurant="r_dev",
              app="Alltown Fresh - DEVELOPMENT"),
    ), "2020-04-21").collect()

    flags = {r.restaurant_id: r.is_development for r in out}
    assert flags == {"r_real": False, "r_dev": True}
    # Kept, not dropped: both stores' revenue is still here.
    assert float(sum(r.total_revenue for r in out)) == pytest.approx(19.98)


def test_a_cell_mixing_real_and_test_traffic_is_flagged(item_rows, build):
    """ANY dev line flags the cell -- a mixed cell is still not plannable."""
    out = build(item_rows(
        _line("o1", "2020-04-21T16:00:00", app="Alltown Fresh", lineitem="1"),
        _line("o2", "2020-04-21T16:00:00",
              app="Alltown Fresh - DEVELOPMENT", lineitem="2"),
    ), "2020-04-21").collect()

    assert len(out) == 1
    assert out[0].is_development is True
