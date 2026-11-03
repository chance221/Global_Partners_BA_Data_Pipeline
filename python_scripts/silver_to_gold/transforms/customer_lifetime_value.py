"""
Lifetime value per customer, with a Top/Mid/Bottom spend tier.

The aggregation and the tier cut both live in `clv_history` now, because
`clv_tier_profile` and `clv_tier_segments` need the same tiering and a gold
transform cannot read another gold table. This module is what publishes the
per-customer row.

`clv_segment` publishes `clv_history.TIERS`, the same words as
`clv_tier_profile` and `clv_tier_segments`. It used to say High/Medium/Low for
the identical cut, which left the warehouse carrying two vocabularies for one
concept. Realigning it CHANGES THE VALUES in a published column rather than
just a display label, so it was a deliberate decision: anything matching on
'High' or 'Low' has to be updated in the same change.

EVERY CUSTOMER IS STILL HERE, INCLUDING THE THREE VOLUME ACCOUNTS
-----------------------------------------------------------------
`is_volume_outlier` marks the three accounts that order 1.4-2.4 times a day
for over a year (see `clv_history` for the measurements and for why the rule
is an order count rather than a percentile). This table FLAGS them and keeps
them: it is the grain of record, so dropping a row here would make the
exclusion unmeasurable and leave the warehouse with no way to answer "how much
were we setting aside". The aggregate tables publish both populations, and the
dashboard reads this flag to filter its own customer-grain frames.

`clv_segment` is computed over the whole base, outliers included, so this
column means the same thing whichever way a reader is looking at the
aggregates. Three rows out of 20,174 do not move a percentile -- the cuts are
$75.93 and $9.89 either way -- so there is nothing to gain from a second
definition and a real cost to having one.
"""

from __future__ import annotations

from datetime import datetime

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from python_scripts.silver_to_gold.transforms import clv_history


def build(inputs: dict[str, DataFrame], batch_date: datetime) -> DataFrame:
    per_customer = clv_history.lifetime(inputs["order_items"])

    return (
        clv_history.with_tier(per_customer, column="clv_segment")
        .withColumn("batch_date",
                    F.lit(batch_date.strftime("%Y-%m-%d")).cast("date"))
        .withColumn("updated_at", F.current_timestamp())
    )
