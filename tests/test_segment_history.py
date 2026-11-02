"""`segment_history` against `customer_rfm`, and the anchor cutoffs.

The segment CASE exists twice on purpose -- customer_rfm anchors to the
dataset's last order date and reads a windowed percentile, so it cannot take an
anchor argument without changing a published table. These tests are what keeps
the second copy honest: given the same rows and the same VIP bar, segmenting at
customer_rfm's own anchor must produce customer_rfm's own answer, segment for
segment.

Without this, a threshold edited in one file and not the other would not fail
anything -- it would quietly publish two different definitions of Slipping.
"""

from datetime import date, timedelta

import pytest
from pyspark.sql import functions as F

from python_scripts.silver_to_gold.transforms import customer_rfm, segment_history

pytestmark = pytest.mark.spark

# The latest order in `mixed_batch`, and therefore customer_rfm's anchor.
# Asserted in the test rather than assumed: if a future edit to the fixture
# adds a later order, every window shifts and the segment expectations below
# would quietly stop meaning what they say.
ANCHOR = "2024-02-20"


def _order(user, order_id, day, price="9.99", loyalty="false", qty=1):
    return {"USER_ID": f"'{user}'", "ORDER_ID": f"'{order_id}'",
            "LINEITEM_ID": "'1'",
            "CREATION_TIME_UTC": f"'{day}T12:00:00'",
            "ITEM_PRICE": f"'{price}'", "ITEM_QUANTITY": str(qty),
            "IS_LOYALTY": loyalty}


@pytest.fixture
def mixed_batch(item_rows):
    """Customers spread across the segment branches, not just one of them.

    Hand-positioned dates rather than a random spread: a test that happens to
    produce three segments proves much less than one that produces seven, and
    the assertion below checks the spread is still there.
    """
    return item_rows(
        # two in the last 30d plus one before it, high spend -> VIP branch
        _order("u_vip", "o1", "2024-02-20", price="80.00"),
        _order("u_vip", "o2", "2024-02-10", price="80.00"),
        _order("u_vip", "o3", "2024-01-10", price="80.00"),
        # four in 90d with one in the last 30d, cheap -> Loyal branch
        _order("u_loyal", "o4", "2024-02-18", price="4.00"),
        _order("u_loyal", "o5", "2024-02-01", price="4.00"),
        _order("u_loyal", "o6", "2024-01-15", price="4.00"),
        _order("u_loyal", "o7", "2023-12-20", price="4.00"),
        # long tenure, one recent order -> Promising
        _order("u_promising", "o8", "2022-06-01", price="10.00"),
        _order("u_promising", "o9", "2024-02-15", price="10.00"),
        # first order inside 30 days -> New
        _order("u_new", "o10", "2024-02-14", price="12.00"),
        # three in 31-90d, nothing in the last 30d -> Slipping
        _order("u_slip", "o11", "2024-01-20", price="7.00"),
        _order("u_slip", "o12", "2024-01-05", price="7.00"),
        _order("u_slip", "o13", "2023-12-15", price="7.00"),
        # two in 31-90d -> Lapsing
        _order("u_lapse", "o14", "2024-01-12", price="6.00"),
        _order("u_lapse", "o15", "2023-12-14", price="6.00"),
        # exactly one lifetime order, inside the window -> Never Returned
        _order("u_never", "o16", "2024-01-02", price="15.00"),
        # a long gap cleared by one fresh order -> Re-engaging
        _order("u_reeng", "o17", "2022-01-05", price="11.00"),
        _order("u_reeng", "o18", "2024-02-19", price="11.00"),
        # nothing inside 90 days -> Dormant
        _order("u_dormant", "o19", "2023-01-17", price="20.00"),
        _order("u_dormant", "o20", "2022-11-17", price="20.00"),
    )


def _rfm_rows(batch, batch_date):
    out = customer_rfm.build({"order_items": batch}, batch_date).collect()
    return {r["user_id"]: r for r in out}


def test_segments_agree_with_customer_rfm_at_the_same_anchor(
        mixed_batch, batch_date):
    """The whole reason the second implementation is allowed to exist."""
    published = _rfm_rows(mixed_batch, batch_date)

    orders = segment_history.per_order(mixed_batch)
    anchor = orders.agg(F.max("order_date")).collect()[0][0]
    # customer_rfm's cut for the ACTIVE partition. Taken from its output rather
    # than recomputed, so the two sides cannot differ because of the bar.
    cut = next(r["vip_spend_cut"] for r in published.values() if r["is_active"])

    rebuilt = {r["user_id"]: r
               for r in segment_history.segments(orders, anchor, cut).collect()}

    assert str(anchor) == ANCHOR
    assert set(rebuilt) == set(published)

    disagreements = {
        uid: (published[uid]["segment"], rebuilt[uid]["segment"])
        for uid in published
        if (published[uid]["segment"] != rebuilt[uid]["segment"]
            or published[uid]["segment_group"] != rebuilt[uid]["segment_group"])
    }
    assert not disagreements, (
        f"segment_history and customer_rfm disagree (published, rebuilt): "
        f"{disagreements}")

    # Guard against the test passing because everyone landed in one branch.
    assert len({r["segment"] for r in published.values()}) >= 7


def test_segments_at_a_past_anchor_ignore_later_orders(mixed_batch):
    """`order_date <= anchor` is the filter that makes the past readable.

    Without it a customer whose first order came after the anchor is counted
    into the cohort as Dormant, which silently inflates every denominator.
    """
    orders = segment_history.per_order(mixed_batch)
    anchor = orders.agg(F.max("order_date")).collect()[0][0]
    past = anchor - timedelta(days=180)

    at_past = {r["user_id"] for r in
               segment_history.segments(orders, past, 0).collect()}

    # u_new's only order is 2024-02-14, inside the 180 days before the anchor,
    # so at the earlier anchor that customer does not exist at all.
    assert "u_new" not in at_past
    assert "u_dormant" in at_past


def test_members_as_of_ignores_enrolment_after_the_anchor(item_rows):
    """Membership read from all history credits the program with the period
    before someone joined. The cutoff is the point of the function."""
    batch = item_rows(
        _order("u_early", "a1", "2023-01-10", loyalty="true"),
        _order("u_late", "a2", "2023-01-10", loyalty="false"),
        _order("u_late", "a3", "2024-02-10", loyalty="true"),
    )
    orders = segment_history.per_order(batch)

    early = {r["user_id"]: r["is_member"] for r in
             segment_history.members_as_of(orders, date(2023, 6, 1)).collect()}
    assert early["u_early"] is True
    assert early["u_late"] is False, \
        "u_late enrolled in 2024 and must not read as a member in mid-2023"

    later = {r["user_id"]: r["is_member"] for r in
             segment_history.members_as_of(orders, date(2024, 3, 1)).collect()}
    assert later["u_late"] is True


def test_frozen_vip_cut_is_measured_the_day_before_launch(mixed_batch):
    """One bar for every table that compares across time."""
    orders = segment_history.per_order(mixed_batch)
    launch = date(2024, 1, 15)

    assert (segment_history.frozen_vip_cut(orders, launch)
            == segment_history.vip_cut(orders, launch - timedelta(days=1)))
