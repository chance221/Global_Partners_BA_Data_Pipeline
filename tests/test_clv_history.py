"""`clv_history`: the lifetime aggregation and the spend-tier cut.

The cut now has three consumers -- `customer_lifetime_value`,
`clv_tier_profile` and `clv_tier_segments` -- because a gold transform cannot
read another gold table and all three have to recompute it. These tests are
what keeps one definition honest.

The tie behaviour is tested rather than documented alone, because it is the
surprising part: the tiers are NOT 20/60/20 on real data, and a future edit
that "fixed" that by switching to `ntile(5)` would pass every other test in
this suite while splitting customers who spent identical amounts.
"""

from __future__ import annotations

import pytest

from python_scripts.silver_to_gold.transforms import clv_history

pytestmark = pytest.mark.spark


def _order(user, order_id, day, price="9.99", lineitem="1", qty=1):
    return {"USER_ID": f"'{user}'", "ORDER_ID": f"'{order_id}'",
            "LINEITEM_ID": f"'{lineitem}'",
            "CREATION_TIME_UTC": f"'{day}T12:00:00'",
            "ITEM_PRICE": f"'{price}'", "ITEM_QUANTITY": str(qty),
            "IS_LOYALTY": "false"}


@pytest.fixture
def spread(item_rows):
    """Five customers at five clearly separated spend levels."""
    return item_rows(
        _order("u_500", "o1", "2024-01-10", price="500.00"),
        _order("u_300", "o2", "2024-01-10", price="300.00"),
        _order("u_200", "o3", "2024-01-10", price="200.00"),
        _order("u_100", "o4", "2024-01-10", price="100.00"),
        _order("u_010", "o5", "2024-01-10", price="10.00"),
    )


def _by_user(frame, column="clv_tier"):
    return {r["user_id"]: r for r in frame.collect()}


def test_lifetime_counts_orders_not_line_items(item_rows):
    """`total_orders` is distinct orders; `line_items` counts the rows.

    A two-line order must not read as two orders, or avg_order_value halves
    and the customer drops a tier for having ordered a side dish.
    """
    batch = item_rows(
        _order("u", "o1", "2024-01-10", price="5.00", lineitem="1"),
        _order("u", "o1", "2024-01-10", price="7.00", lineitem="2"),
        _order("u", "o2", "2024-01-11", price="3.00", lineitem="1"),
    )
    row = clv_history.lifetime(batch).collect()[0]

    assert row["total_orders"] == 2
    assert row["line_items"] == 3
    assert float(row["total_lifetime_revenue"]) == 15.00
    assert float(row["avg_order_value"]) == 7.50


def test_lifetime_drops_anonymous_rows(item_rows):
    """A per-customer measure needs a customer."""
    batch = item_rows(
        _order("u_real", "o1", "2024-01-10", price="10.00"),
        {"USER_ID": "NULL", "ORDER_ID": "'o2'", "LINEITEM_ID": "'1'",
         "CREATION_TIME_UTC": "'2024-01-10T12:00:00'",
         "ITEM_PRICE": "'99.00'", "ITEM_QUANTITY": "1", "IS_LOYALTY": "false"},
        _order("   ", "o3", "2024-01-10", price="99.00"),
    )
    out = clv_history.lifetime(batch).collect()

    assert [r["user_id"] for r in out] == ["u_real"]


def test_top_tier_is_the_high_spend_end_at_percentile_zero(spread):
    """`value_percentile` 0.0 is the BIGGEST spender, not the smallest.

    The ranking is descending, so the top fifth is the low end of the
    percentile range. That reads backwards on first encounter and is the one
    thing a consumer of this column is likely to get wrong.
    """
    rows = _by_user(clv_history.with_tier(clv_history.lifetime(spread)))

    assert rows["u_500"]["clv_tier"] == "Top"
    assert float(rows["u_500"]["value_percentile"]) == 0.0
    assert rows["u_010"]["clv_tier"] == "Bottom"
    assert float(rows["u_010"]["value_percentile"]) == 1.0
    assert rows["u_200"]["clv_tier"] == "Mid"


def test_tied_spend_never_splits_across_tiers(item_rows):
    """Two customers who spent the same amount land in the same tier.

    This is why `percent_rank` is used and `ntile` is not. The cost is that
    the realised tier shares drift off 20/60/20 -- on the real data the
    Bottom tier is 18.04% because 475 customers tie at exactly $9.89 -- and
    that cost is accepted deliberately. See `clv_history`.
    """
    batch = item_rows(
        _order("u_big", "o1", "2024-01-10", price="500.00"),
        _order("u_mid", "o2", "2024-01-10", price="100.00"),
        _order("u_t1", "o3", "2024-01-10", price="10.00"),
        _order("u_t2", "o4", "2024-01-10", price="10.00"),
        _order("u_t3", "o5", "2024-01-10", price="10.00"),
    )
    rows = _by_user(clv_history.with_tier(clv_history.lifetime(batch)))
    tied = ("u_t1", "u_t2", "u_t3")

    assert len({rows[u]["clv_tier"] for u in tied}) == 1, (
        "a tie split across tiers: "
        f"{ {u: rows[u]['clv_tier'] for u in tied} }")
    assert len({float(rows[u]["value_percentile"]) for u in tied}) == 1

    # And the consequence: the bottom fifth is NOT one of five customers here,
    # because the tie that would have filled it sits above the 0.8 boundary.
    assert sum(1 for r in rows.values() if r["clv_tier"] == "Bottom") != 1


def test_every_consumer_gets_the_same_three_words(spread):
    """One vocabulary, whichever column name a table publishes it under.

    `customer_lifetime_value` writes the tier as `clv_segment` and the two
    aggregate tables write it as `clv_tier`. Those used to be different words
    for the identical cut -- High/Medium/Low against Top/Mid/Low -- which is
    the state this test exists to prevent returning to. The column name is a
    caller's choice; the labels are not.
    """
    per_customer = clv_history.lifetime(spread)
    as_tier = _by_user(clv_history.with_tier(per_customer))
    as_segment = _by_user(
        clv_history.with_tier(per_customer, column="clv_segment"))

    assert set(clv_history.TIERS) == {"Top", "Mid", "Bottom"}, (
        "the published vocabulary changed; gold.customer_lifetime_value."
        "clv_segment carries these values, so this is a data change")
    assert {r["clv_segment"] for r in as_segment.values()} <= set(
        clv_history.TIERS)
    for user, row in as_tier.items():
        assert as_segment[user]["clv_segment"] == row["clv_tier"], (
            f"{user} is {row['clv_tier']} as clv_tier but "
            f"{as_segment[user]['clv_segment']} as clv_segment")


# --- the volume-outlier flag and the two populations ----------------------
#
# The real floor is 900 lifetime orders, which no fixture is going to build.
# These patch it down to 4 and check the MECHANISM: that the rule is an order
# count, that the two populations differ by exactly the flagged rows, and that
# the tier label does not move between them. The threshold's own value is a
# judgement recorded in `clv_history`, not something a test can assert.

@pytest.fixture
def heavy(item_rows):
    """One account with four orders, three ordinary ones with a single order.

    Spends are separated so the tiers are unambiguous: u_a 100 (Top), u_b 50
    and u_heavy 20 (Mid), u_c 1 (Bottom).
    """
    rows = [_order("u_heavy", f"h{i}", f"2024-01-{10 + i:02d}", price="5.00")
            for i in range(4)]
    rows += [_order("u_a", "a1", "2024-01-10", price="100.00"),
             _order("u_b", "b1", "2024-01-10", price="50.00"),
             _order("u_c", "c1", "2024-01-10", price="1.00")]
    return item_rows(*rows)


@pytest.fixture
def low_floor(monkeypatch):
    """Drop the flag's threshold to 4 orders for the duration of a test.

    `lifetime` reads the constant when it is called, not at import, so
    patching the module attribute is enough.
    """
    monkeypatch.setattr(clv_history, "OUTLIER_ORDER_FLOOR", 4)
    return 4


def test_volume_flag_keys_off_order_count_not_spend(heavy, low_floor):
    """The flagged account is the FREQUENT one, not the big spender.

    u_a spends five times what u_heavy does and is not flagged; u_heavy is.
    That is the whole point of the rule -- a spend percentile would catch the
    best customers, and on the real data a p99 spend cut flags 64 current
    VIPs.
    """
    rows = _by_user(clv_history.lifetime(heavy))

    assert rows["u_heavy"]["is_volume_outlier"] is True
    assert rows["u_heavy"]["total_orders"] == 4
    assert rows["u_a"]["is_volume_outlier"] is False
    assert float(rows["u_a"]["total_lifetime_revenue"]) == 100.00
    assert not any(rows[u]["is_volume_outlier"] for u in ("u_a", "u_b", "u_c"))


def test_the_flag_is_inclusive_of_the_floor(heavy, low_floor):
    """`>=`, so an account exactly at the threshold is flagged.

    The boundary is where an off-by-one would hide, and on the real data the
    third account sits at exactly 1,000 orders -- which is why the floor is
    900 rather than 1000. See `clv_history.OUTLIER_ORDER_FLOOR`.
    """
    rows = _by_user(clv_history.lifetime(heavy))

    assert rows["u_heavy"]["total_orders"] == low_floor
    assert rows["u_heavy"]["is_volume_outlier"] is True


def test_variants_differ_by_exactly_the_flagged_rows(heavy, low_floor):
    """Two populations, and the excluded one is the included one minus three."""
    variants = clv_history.with_outlier_variants(
        clv_history.with_tier(clv_history.lifetime(heavy)))
    rows = variants.collect()

    assert {r["outliers"] for r in rows} == set(clv_history.OUTLIER_VARIANTS)
    included = {r["user_id"] for r in rows if r["outliers"] == "included"}
    excluded = {r["user_id"] for r in rows if r["outliers"] == "excluded"}

    assert included == {"u_heavy", "u_a", "u_b", "u_c"}
    assert excluded == {"u_a", "u_b", "u_c"}
    assert included - excluded == {"u_heavy"}


def test_the_tier_label_is_identical_in_both_variants(heavy, low_floor):
    """The cut is computed once, over the whole base, and both copies keep it.

    This is the invariant that keeps `customer_lifetime_value.clv_segment`
    agreeing with `clv_tier_profile` whichever population a reader is looking
    at. Calling `with_outlier_variants` BEFORE `with_tier` would re-rank the
    excluded copy and break it, which is why the ordering is asserted here
    rather than only documented.
    """
    variants = clv_history.with_outlier_variants(
        clv_history.with_tier(clv_history.lifetime(heavy))).collect()

    by_variant: dict[str, dict[str, str]] = {}
    for r in variants:
        by_variant.setdefault(r["outliers"], {})[r["user_id"]] = r["clv_tier"]

    shared = set(by_variant["included"]) & set(by_variant["excluded"])
    assert shared == {"u_a", "u_b", "u_c"}
    for user in shared:
        assert by_variant["included"][user] == by_variant["excluded"][user], (
            f"{user} changed tier between populations: "
            f"{by_variant['included'][user]} -> {by_variant['excluded'][user]}")
    assert by_variant["included"]["u_a"] == "Top"
    assert by_variant["included"]["u_c"] == "Bottom"


def test_an_order_grain_join_drops_the_flagged_orders(heavy, low_floor):
    """The property `clv_tier_profile` relies on for its order-grain medians.

    A flagged customer has no `excluded` row, so their orders find nothing to
    join to and fall out on their own. No separate order-level exclusion list
    exists, which is the point -- a second list is what would drift out of
    agreement with the customer-level one.
    """
    from python_scripts.silver_to_gold.transforms import segment_history

    variants = clv_history.with_outlier_variants(
        clv_history.with_tier(clv_history.lifetime(heavy)))
    orders = segment_history.per_order(heavy)

    joined = orders.join(
        variants.select("user_id", "clv_tier", "outliers"),
        on="user_id", how="inner")
    counts = {r["outliers"]: r["n"] for r in
              joined.groupBy("outliers").count()
              .withColumnRenamed("count", "n").collect()}

    # 7 orders in all; the flagged account placed 4 of them.
    assert counts["included"] == 7
    assert counts["excluded"] == 3


def test_orders_per_active_day_is_degenerate_for_one_order_customers(spread):
    """Documents the trap rather than pretending the column is clean.

    A single-order customer spans one day and scores exactly 1.00 orders/day,
    which outranks the real volume accounts at 1.4-2.4. The column is evidence
    for a judgement about high-order accounts, never a rule on its own -- and
    the median of 1.00 across the real base is an artifact of this, not a
    baseline anyone should quote.
    """
    rows = _by_user(clv_history.lifetime(spread))

    for user, row in rows.items():
        assert row["total_orders"] == 1
        assert row["active_days"] == 1, f"{user} spans {row['active_days']}"
        assert float(row["orders_per_active_day"]) == 1.0
        assert row["is_volume_outlier"] is False
