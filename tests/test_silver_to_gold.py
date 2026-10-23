"""Tests for the dependency guard in run_silver_to_gold.

A gold table is only as correct as the silver it reads, so the job refuses to
build one whose inputs are not in the catalog yet. These tests pin down three
things about that refusal: it fires, it names the missing table, and it is not
vacuous -- the same call succeeds once the dependency exists.

The last point is the one worth having. A guard that always trips looks
identical to a working guard from the failing side only.

Measured by deleting the guard: only
`test_missing_dependency_names_the_silver_table` fails. The other tests keep
passing, because Spark's own TABLE_OR_VIEW_NOT_FOUND reaches the same
RuntimeError by a worse route. So the guard buys a readable message and a
failure before any read, not failure detection -- Spark detects it either way.
"""

from dataclasses import replace

import pytest

from python_scripts.common.models.config_models import GoldTableConfig
from python_scripts.raw_to_silver.raw_transforms import apply_dynamic_transform
from python_scripts.raw_to_silver.silver_write import write_silver
from python_scripts.silver_to_gold.silver_to_gold import run_silver_to_gold

pytestmark = pytest.mark.spark

GOLD = "daily_restaurant_sales"


def seed_silver(spark, cfg, batch_date, table, df):
    """Put a real batch into silver, the way run_raw_to_silver would."""
    table_config = cfg.get_table_config(table)
    transformed = apply_dynamic_transform(df, table_config, batch_date)
    write_silver(spark, transformed, cfg, table, table_config)


# --- the dependency is absent ---------------------------------------------

def test_missing_dependency_fails_the_table(spark, cfg, batch_date,
                                            silver_table, gold_table):
    """The `silver_table` fixture drops order_items, so the input is gone.

    Note the type: the ValueError raised inside the loop is collected into
    `failures` rather than propagating, and what reaches the caller is the
    RuntimeError raised after the loop.
    """
    silver_table("order_items")
    gold_table(GOLD)

    with pytest.raises(RuntimeError, match=r"silver_to_gold failed for: \['daily_restaurant_sales'\]"):
        run_silver_to_gold(spark, cfg, batch_date, [GOLD])


def test_missing_dependency_names_the_silver_table(spark, cfg, batch_date,
                                                   silver_table, gold_table,
                                                   capsys):
    """Which input is missing is printed, not carried on the exception.

    `capsys` is a pytest built-in that captures stdout; `readouterr()` returns
    what was written and clears the buffer. Asserting on print output is only
    worth doing where the message is the whole diagnostic, as it is here.
    """
    silver_table("order_items")
    gold_table(GOLD)

    with pytest.raises(RuntimeError):
        run_silver_to_gold(spark, cfg, batch_date, [GOLD])

    assert "silver tables not built yet: ['order_items']" in capsys.readouterr().out


def test_missing_dependency_writes_no_gold_table(spark, cfg, batch_date,
                                                 silver_table, gold_table):
    """The guard runs before the read and the write, so gold stays untouched.

    Without this, a guard that raised *after* write_gold would still pass the
    two tests above while leaving a half-built table behind.
    """
    silver_table("order_items")
    fqn = gold_table(GOLD)

    with pytest.raises(RuntimeError):
        run_silver_to_gold(spark, cfg, batch_date, [GOLD])

    assert not spark.catalog.tableExists(fqn)


# --- the dependency is present --------------------------------------------

def test_present_dependency_builds_the_table(spark, cfg, batch_date, item_rows,
                                             silver_table, gold_table):
    """The control case: same call, same config, dependency now in the catalog."""
    silver_table("order_items")
    fqn = gold_table(GOLD)
    seed_silver(spark, cfg, batch_date, "order_items", item_rows(
        {"LINEITEM_ID": "'1'", "ITEM_PRICE": "'9.99'", "ITEM_QUANTITY": "2"},
        {"LINEITEM_ID": "'2'", "ITEM_PRICE": "'4.50'", "ITEM_QUANTITY": "1"},
    ))

    run_silver_to_gold(spark, cfg, batch_date, [GOLD])

    row = spark.table(fqn).collect()[0]
    assert row["total_orders"] == 1
    assert row["line_items"] == 2
    assert row["units"] == 3
    # item_price is the fully loaded line total -- options and quantity are
    # already in it -- so revenue is a plain sum, NOT price * quantity.
    assert str(row["total_revenue"]) == "14.49"     # 9.99 + 4.50
    assert str(row["avg_order_total"]) == "14.49"   # one order
    # 2020-04-21 is a Tuesday, so its Monday-start week begins 2020-04-20.
    assert str(row["week_start"]) == "2020-04-20"


# --- one missing input among several tables -------------------------------

def test_a_missing_dependency_does_not_stop_the_other_tables(
        spark, cfg, batch_date, item_rows, silver_table, gold_table):
    """The loop collects failures instead of aborting on the first one.

    Worth stating explicitly because it is the opposite of what "the job stops"
    suggests: a gold table whose inputs are missing is skipped, every other
    table is still built, and the run fails at the end.

    `replace` on a frozen dataclass returns a copy with one field changed, so
    this adds a second gold table without editing config/local.yaml.
    """
    two_tables = replace(cfg, gold={
        **cfg.gold,
        "unbuilt_table": GoldTableConfig(
            depends_on=["never_loaded"],
            grain_keys=["sales_date"],
            scope_keys=["sales_date"],
        ),
    })
    silver_table("order_items")
    fqn = gold_table(GOLD)
    seed_silver(spark, cfg, batch_date, "order_items", item_rows({}))

    with pytest.raises(RuntimeError, match=r"silver_to_gold failed for: \['unbuilt_table'\]"):
        run_silver_to_gold(spark, two_tables, batch_date,
                           ["unbuilt_table", GOLD])

    assert spark.table(fqn).count() == 1
