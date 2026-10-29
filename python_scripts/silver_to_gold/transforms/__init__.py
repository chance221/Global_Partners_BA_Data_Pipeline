"""Registry of gold transforms.

One module per gold table. Each exposes `build(inputs, batch_date) -> DataFrame`
where `inputs` is {silver_table_name: DataFrame}, already narrowed to the rows
this run must recompute.

A transform aggregates and returns the result. It does not read, does not write,
and does not know whether the run is incremental -- silver_read decides what it
sees and gold_write decides what happens to the output. That is what keeps a
transform testable with a handful of literal rows.

Register a new one by importing it and adding it below, keyed by the same name
used in the `gold:` section of config.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

from pyspark.sql import DataFrame

from python_scripts.silver_to_gold.transforms import (
    category_holiday_sales, category_period_sales, customer_lifetime_value,
    customer_rfm, daily_restaurant_sales, loyalty_era_profile,
    loyalty_order_profile, restaurant_revenue_rank,
)

Transform = Callable[[dict[str, DataFrame], datetime], DataFrame]

TRANSFORMS: dict[str, Transform] = {
    "daily_restaurant_sales": daily_restaurant_sales.build,
    "restaurant_revenue_rank": restaurant_revenue_rank.build,
    "customer_lifetime_value": customer_lifetime_value.build,
    "customer_rfm": customer_rfm.build,
    "category_period_sales": category_period_sales.build,
    "category_holiday_sales": category_holiday_sales.build,
    "loyalty_order_profile": loyalty_order_profile.build,
    "loyalty_era_profile": loyalty_era_profile.build,
}


def get_transform(name: str) -> Transform:
    """Look up a transform, with an error that lists the registered names."""
    try:
        return TRANSFORMS[name]
    except KeyError:
        raise KeyError(
            f"no transform registered for gold table {name!r}; "
            f"registered: {sorted(TRANSFORMS)}"
        ) from None
