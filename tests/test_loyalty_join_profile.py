"""`loyalty_join_profile`: the period boundaries, and what the join day does.

Three properties carry this table and all three are invisible in a chart.

The first is that day +1 and day -1 land in DIFFERENT periods. The natural
implementation is `ceil(days / 30)`, and Spark's ceil rounds a negative
quotient toward zero, so day -1 and day +1 both come out as period 0 and the
first period either side of enrollment is silently merged.

The second is that the join day sits alone in period 0. It is by definition a
day the customer ordered -- the day of their first flagged order -- so letting
it into period +1 hands that side a guaranteed order the pre side has no
equivalent of, which manufactures a lift out of nothing.

The third is additivity across periods, which is what lets a consumer build a
+/-90 reading out of periods -3..3 instead of needing a second table.
"""

from __future__ import annotations

import pytest

from python_scripts.silver_to_gold.transforms import (
    clv_history, loyalty_join_profile,
)

pytestmark = pytest.mark.spark

JOIN = "2022-01-01"


def _line(order_id, utc_date, *, loyalty=False, user="u1", price="10.00",
          lineitem="1", qty=1):
    return {"ORDER_ID": f"'{order_id}'", "LINEITEM_ID": f"'{lineitem}'",
            "CREATION_TIME_UTC": f"'{utc_date}T12:00:00'",
            "ITEM_PRICE": f"'{price}'", "ITEM_QUANTITY": str(qty),
            "USER_ID": f"'{user}'",
            "IS_LOYALTY": "true" if loyalty else "false"}


def _anchors(user="u1"):
    """Orders that make `user` a switcher joining on JOIN, with wide coverage.

    The coverage test asks whether the DATA spans N days either side of the
    join date, so the span has to be stretched by real orders -- these two
    are what every other case in this file sits inside. A second customer
    carries them so they cannot be mistaken for the subject's own activity.
    """
    return [_line("span_lo", "2020-01-01", user="u9"),
            _line("span_hi", "2024-01-01", user="u9"),
            _line("join", JOIN, loyalty=True, user=user),
            _line("pre_far", "2021-01-01", user=user)]


@pytest.fixture
def build(item_rows, batch_date):
    def run(*lines):
        return loyalty_join_profile.build(
            {"order_items": item_rows(*lines)}, batch_date)
    return run


def _rows(out, coverage=180):
    return {r["period_index"]: r
            for r in out.where(f"coverage_days = {coverage}").collect()}


# --- the period boundaries ------------------------------------------------

def test_day_minus_one_and_day_plus_one_are_different_periods(build):
    """The ceil-toward-zero bug: both would land in period 0."""
    out = build(*_anchors(),
                _line("before", "2021-12-31"),
                _line("after", "2022-01-02"))
    rows = _rows(out)

    assert rows[-1]["orders"] == 1, "day -1 is not in period -1"
    assert rows[1]["orders"] == 1, "day +1 is not in period +1"


@pytest.mark.parametrize("date,period", [
    ("2021-12-02", -1),   # day -30, the far edge of the first pre period
    ("2021-12-01", -2),   # day -31, the first day of the second
    ("2022-01-31", 1),    # day +30, the far edge of the first post period
    ("2022-02-01", 2),    # day +31, the first day of the second
    ("2021-07-06", -6),   # day -179, just inside coverage
    ("2022-06-29", 6),    # day +179, just inside coverage
])
def test_period_edges(build, date, period):
    """Each 30-day period is exactly 30 days wide and starts where it should."""
    out = build(*_anchors(), _line("probe", date))
    rows = _rows(out)

    assert rows[period]["orders"] == 1, f"{date} is not in period {period}"
    assert rows[period]["period_start_day"] <= 0 or period > 0


def test_period_bounds_describe_their_own_period(build):
    """period_start_day/period_end_day are published, so they must be right."""
    rows = _rows(build(*_anchors()))

    assert (rows[1]["period_start_day"], rows[1]["period_end_day"]) == (1, 30)
    assert (rows[2]["period_start_day"], rows[2]["period_end_day"]) == (31, 60)
    assert (rows[-1]["period_start_day"], rows[-1]["period_end_day"]) == (-30, -1)
    assert (rows[-2]["period_start_day"], rows[-2]["period_end_day"]) == (-60, -31)
    assert (rows[0]["period_start_day"], rows[0]["period_end_day"]) == (0, 0)


# --- the join day ---------------------------------------------------------

def test_the_join_order_is_in_period_zero_and_not_in_period_one(build):
    """Otherwise the post side gets a free order and invents a lift."""
    rows = _rows(build(*_anchors()))

    assert rows[0]["orders"] == 1, "the join order left period 0"
    assert rows[1]["orders"] == 0, "the join order leaked into period +1"


def test_period_zero_is_one_day_wide_and_says_so(build):
    """A 30-day denominator on a one-day period overstates nothing else."""
    rows = _rows(build(*_anchors()))

    assert rows[0]["window_days"] == 1
    assert all(r["window_days"] == 30
               for p, r in rows.items() if p != 0)


def test_orders_per_customer_per_day_normalises_the_join_day(build):
    """The measure that is comparable across every row, join day included."""
    rows = _rows(build(*_anchors(), _line("p1a", "2022-01-02"),
                       _line("p1b", "2022-01-03")))

    # One order on one day against two orders over thirty.
    assert float(rows[0]["orders_per_customer_per_day"]) == pytest.approx(1.0)
    assert float(rows[1]["orders_per_customer_per_day"]) == pytest.approx(
        2 / 30, abs=1e-6)


# --- the cohort -----------------------------------------------------------

def test_a_customer_with_only_flagged_orders_is_not_a_switcher(build):
    """No before state means before and after are not the same customer."""
    out = build(_line("span_lo", "2020-01-01", user="u9"),
                _line("span_hi", "2024-01-01", user="u9"),
                _line("join", JOIN, loyalty=True, user="u1"),
                _line("more", "2022-02-01", loyalty=True, user="u1"))

    assert out.count() == 0


def test_a_customer_without_enough_coverage_is_excluded(build):
    """A period the data does not cover would read as zero orders."""
    # Data ends 40 days after the join, so +/-90 and +/-180 both fail.
    out = build(_line("pre", "2021-01-01", user="u1"),
                _line("join", JOIN, loyalty=True, user="u1"),
                _line("end", "2022-02-10", user="u9"))

    assert out.count() == 0


def test_the_ninety_day_cohort_can_be_wider_than_the_one_eighty(build):
    """The populations overlap: 180 is a subset, and that has to be visible."""
    # u1 joins with 180 days clear either side; u2 has only 90 after.
    out = build(_line("span_lo", "2020-01-01", user="u9"),
                _line("span_hi", "2024-01-01", user="u9"),
                _line("pre1", "2021-01-01", user="u1"),
                _line("join1", JOIN, loyalty=True, user="u1"),
                _line("pre2", "2023-01-01", user="u2"),
                _line("join2", "2023-09-01", loyalty=True, user="u2"))

    at90 = out.where("coverage_days = 90").collect()[0]["customers"]
    at180 = out.where("coverage_days = 180").collect()[0]["customers"]
    assert (at90, at180) == (2, 1)


def test_customers_is_constant_within_a_coverage_population(build):
    """The denominator is what makes the periods comparable to each other."""
    out = build(*_anchors(), _line("extra", "2022-03-01"))

    for coverage in (90, 180):
        counts = {r["customers"]
                  for r in out.where(f"coverage_days = {coverage}").collect()}
        assert len(counts) == 1, f"customers varies by period at {coverage}"


# --- zeros, and additivity ------------------------------------------------

def test_an_empty_period_is_a_zero_and_not_a_missing_row(build):
    """A gap in a decay curve reads as no data when it means no orders."""
    rows = _rows(build(*_anchors()))

    assert len(rows) == 13, "expected periods -6..6 for coverage 180"
    assert rows[5]["orders"] == 0
    assert rows[5]["median_order_value"] is None
    assert rows[5]["orders_per_active_day"] is None, \
        "no visits is not one order per visit"


def test_measures_are_additive_across_periods(build):
    """So a +/-90 reading needs no second table."""
    out = build(*_anchors(),
                _line("a", "2022-01-05", price="10.00"),
                _line("b", "2022-02-05", price="20.00"),
                _line("c", "2021-12-05", price="30.00"))
    rows = _rows(out)

    total = sum(r["orders"] for r in rows.values())
    revenue = sum(float(r["total_revenue"]) for r in rows.values())
    # 3 probes + the join order + pre_far at day -366 (outside +/-180).
    assert total == 4
    assert revenue == pytest.approx(10.0 + 20.0 + 30.0 + 10.0)


def test_active_days_counts_customer_days_not_orders(build):
    """Two orders on one day is one visit, which is the whole distinction."""
    rows = _rows(build(*_anchors(),
                       _line("x", "2022-01-05", lineitem="1"),
                       _line("y", "2022-01-05", lineitem="1")))

    assert rows[1]["orders"] == 2
    assert rows[1]["active_days"] == 1
    assert float(rows[1]["orders_per_active_day"]) == pytest.approx(2.0)


# --- the volume accounts --------------------------------------------------

def test_outlier_portions_net_out_to_the_excluded_reading(build,
                                                          monkeypatch):
    """`measure - outlier_measure` has to BE the excluded reading, exactly.

    The real floor is 900 lifetime orders, which a fixture would need 900 row
    literals to trip. Lowered deliberately: what this asserts is that the
    portion columns are wired to the shared flag and net out correctly, not
    where clv_history puts the floor -- that belongs to clv_history's tests.

    Two switchers joining the same day, both fully covered. `u1` has three
    lifetime orders and stays under the floor; `whale` has six and goes over
    it. Everything lands in period +1, so the expected values are exact
    rather than dependent on where the tiering happens to fall.
    """
    monkeypatch.setattr(clv_history, "OUTLIER_ORDER_FLOOR", 5)
    out = build(
        _line("span_lo", "2020-01-01", user="u9"),
        _line("span_hi", "2024-01-01", user="u9"),
        _line("join", JOIN, loyalty=True, user="u1"),
        _line("pre_far", "2021-01-01", user="u1"),
        _line("mine", "2022-01-10", price="10.00", user="u1"),
        _line("wjoin", JOIN, loyalty=True, user="whale"),
        _line("wpre", "2021-01-01", user="whale"),
        *[_line(f"bulk{i}", "2022-01-10", user="whale", price="5.00")
          for i in range(4)],
    )
    p1 = _rows(out)[1]

    assert (p1["orders"], p1["outlier_orders"]) == (5, 4)
    assert p1["orders"] - p1["outlier_orders"] == 1
    assert (float(p1["total_revenue"]), float(p1["outlier_revenue"])) == (30.0, 20.0)
    # A customer-day belongs to exactly one customer, so this nets too.
    assert (p1["active_days"], p1["outlier_active_days"]) == (2, 1)
    assert (p1["customers"], p1["outlier_customers"]) == (2, 1)

def test_outlier_portions_are_zero_and_not_null_when_nobody_is_flagged(build):
    """Null would make every netted measure null in a consumer's subtraction."""
    rows = _rows(build(*_anchors(), _line("p", "2022-01-05")))

    for period, row in rows.items():
        for column in ("outlier_orders", "outlier_revenue",
                       "outlier_line_items", "outlier_units",
                       "outlier_active_days", "outlier_customers"):
            assert row[column] is not None, \
                f"{column} is null in period {period}"


def test_outlier_customers_is_constant_within_a_population(build):
    """It is a property of the cohort, not of the period."""
    out = build(*_anchors(), _line("p", "2022-01-05"))

    for coverage in (90, 180):
        counts = {r["outlier_customers"]
                  for r in out.where(f"coverage_days = {coverage}").collect()}
        assert len(counts) == 1
