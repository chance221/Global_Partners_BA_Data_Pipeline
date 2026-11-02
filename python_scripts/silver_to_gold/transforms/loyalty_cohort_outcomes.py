"""Loyalty members against non-members, measured over the SAME 180 days.

One row per (cohort, seg_t0). Customers are split by whether they held loyalty
membership at T0, then followed to T1 and scored on where they ended up. Both
cohorts are observed over one identical window, which is the whole point:

  loyalty_era_profile      splits on CALENDAR -- before the program against
                           after it -- so it cannot separate the program from
                           the January-2023 reprice or the chain's growth.
  loyalty_order_profile    splits WITHIN a customer, before and after their own
                           enrolment, which holds the customer fixed but not
                           the calendar.
  this table               splits on MEMBERSHIP inside one shared window, which
                           holds the calendar fixed but not the customer.

Three different confounds, so the three tables are read together. None of them
is a randomised trial and this one is not either -- see the self-selection
warning below.

MEMBERSHIP IS TAKEN AS OF T0, NOT FROM ALL HISTORY
--------------------------------------------------
`segment_history.members_as_of` reads the flag from orders placed on or before
T0. Reading it from all history would file a customer who enrolled DURING the
window as a member throughout, crediting the program with the period before
they joined. That single mistake is the easiest way to manufacture a loyalty
effect that is not there, so the cutoff is explicit and T0 is published.

WHY THE `All customers` ROWS ARE THE ONES NOT TO QUOTE
------------------------------------------------------
Rows ROLL UP: every customer is counted once under their T0 segment and once
under `All customers`, so summing `customers` over this table reports twice the
base. The rollup rows exist because they are what a reader reaches for first,
and they are badly misleading on their own:

Members are not spread evenly across segments -- they concentrate in the
engaged ones, which retain better whether or not anyone enrols. So the pooled
gap is mostly COMPOSITION, and it overstates the within-segment effect by
several times. The per-segment rows are the comparison; the pooled row is the
thing it corrects. Both are published so the difference between them is
visible rather than something a reader has to be warned about in prose.

WHAT THIS STILL CANNOT TELL YOU
-------------------------------
Enrolment is a choice, so members differ from non-members in ways no column
here captures -- a customer who signs up for a rewards program was likelier to
return anyway. A positive per-segment gap is consistent with the program
working AND with it attracting the customers who were going to stay. This table
bounds the effect; it does not identify it. Saying so is the point of the
docstring.

The VIP bar is `segment_history.frozen_vip_cut` -- the day before launch, the
same dollar value loyalty_era_profile and customer_segment_transitions read --
so one bar classifies both ends of the window and all three tables agree on
what a VIP is. Full rebuild, no
event_time_column -- a slice of one batch cannot say who was a member 180 days
before it. Twenty rows, so unpartitioned.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from python_scripts.silver_to_gold.transforms.customer_segment_transitions import (
    TRANSITION_DAYS,
)
from python_scripts.silver_to_gold.transforms.segment_history import (
    CORE,
    DORMANT,
    frozen_vip_cut,
    launch_date,
    members_as_of,
    per_order,
    segments,
)

MEMBER = "Member at T0"
NON_MEMBER = "Non-member at T0"
ALL_CUSTOMERS = "All customers"


def _with_rollup(paired: DataFrame) -> DataFrame:
    """Repeat every customer once more under the `All customers` label.

    The same relabel-and-union trick loyalty_era_profile uses, so one groupBy
    produces both the per-segment rows and the pooled ones.
    """
    return paired.unionByName(
        paired.withColumn("seg_t0", F.lit(ALL_CUSTOMERS)))


def build(inputs: dict[str, DataFrame], batch_date: datetime) -> DataFrame:
    orders = per_order(inputs["order_items"])

    last_order = orders.agg(F.max("order_date")).collect()[0][0]
    if last_order is None:
        last_order = batch_date.date()

    t1 = last_order
    t0 = t1 - timedelta(days=TRANSITION_DAYS)

    # The same frozen bar the other two time-comparing tables read, rather than
    # one measured at this window's own T0: three tables disagreeing about what
    # a VIP is would make their segment rows incomparable.
    launch = launch_date(orders, last_order + timedelta(days=1))
    cut = frozen_vip_cut(orders, launch)

    at_t0 = (segments(orders, t0, cut)
             .select("user_id", F.col("segment").alias("seg_t0")))
    at_t1 = (segments(orders, t1, cut)
             .select("user_id",
                     F.col("segment_group").alias("_grp_t1"),
                     F.col("orders_last90d").alias("_orders_t1"),
                     F.col("spend_last90d").alias("_spend_t1")))

    # Left join on membership rather than inner: every T0 customer has an
    # order by T0 and so appears in members_as_of, but coalescing to False
    # means a change upstream cannot silently drop a cohort.
    paired = (
        at_t0
        .join(at_t1, on="user_id", how="inner")
        .join(members_as_of(orders, t0), on="user_id", how="left")
        .withColumn("cohort",
                    F.when(F.coalesce(F.col("is_member"), F.lit(False)),
                           F.lit(MEMBER))
                     .otherwise(F.lit(NON_MEMBER)))
        .drop("is_member")
    )

    return (
        _with_rollup(paired)
        .groupBy("cohort", "seg_t0")
        .agg(
            F.count(F.lit(1)).alias("customers"),
            # still_active and to_dormant are complements by construction --
            # a group is Dormant or it is not. Both published because a reader
            # asking "did they stay?" and one asking "did we lose them?" should
            # not have to subtract.
            (F.avg((F.col("_grp_t1") != F.lit(DORMANT)).cast("int")) * 100)
            .cast("decimal(5,2)").alias("still_active_pct"),
            (F.avg((F.col("_grp_t1") == F.lit(CORE)).cast("int")) * 100)
            .cast("decimal(5,2)").alias("to_core_pct"),
            (F.avg((F.col("_grp_t1") == F.lit(DORMANT)).cast("int")) * 100)
            .cast("decimal(5,2)").alias("to_dormant_pct"),
            F.avg("_orders_t1").cast("decimal(10,2)").alias("avg_orders_90d_t1"),
            F.avg("_spend_t1").cast("decimal(10,2)").alias("avg_spend_90d_t1"),
        )
        .withColumn("t0", F.lit(t0))
        .withColumn("t1", F.lit(t1))
        .withColumn("transition_days", F.lit(TRANSITION_DAYS))
        .withColumn("vip_spend_cut", F.lit(cut).cast("decimal(18,2)"))
        .withColumn("batch_date",
                    F.lit(batch_date.strftime("%Y-%m-%d")).cast("date"))
        .withColumn("updated_at", F.current_timestamp())
    )
