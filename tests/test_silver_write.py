"""Tests for the three ways a batch is applied to a silver Iceberg table.

Each test gets an empty table from the `silver_table` fixture and drops it
afterwards, so order of execution cannot matter.
"""

import pytest

from python_scripts.raw_to_silver.raw_transforms import apply_dynamic_transform
from python_scripts.raw_to_silver.silver_write import write_silver

pytestmark = pytest.mark.spark

ORDER = "'5e9f41ec505ee9c65088f162'"
OTHER_ORDER = "'aaaaaaaaaaaaaaaaaaaaaaaa'"


def apply(spark, cfg, batch_date, table, df):
    """Transform a raw batch and write it, as run_raw_to_silver does."""
    table_config = cfg.get_table_config(table)
    transformed = apply_dynamic_transform(df, table_config, batch_date)
    return write_silver(spark, transformed, cfg, table, table_config)


# --- first run ------------------------------------------------------------

def test_first_run_creates_the_table(spark, cfg, item_rows, batch_date,
                                     silver_table):
    """MERGE and DELETE both need a target, and a first run has none."""
    fqn = silver_table("order_items")

    outcome = apply(spark, cfg, batch_date, "order_items", item_rows({}))

    assert outcome == f"created {fqn}"
    assert spark.table(fqn).count() == 1


def test_creation_applies_the_configured_casts(spark, cfg, item_rows, batch_date,
                                               silver_table):
    """required_columns describes the source; silver_transforms overrides types."""
    fqn = silver_table("order_items")

    apply(spark, cfg, batch_date, "order_items", item_rows({}))
    schema = dict(spark.table(fqn).dtypes)

    assert schema["creation_time_utc"] == "timestamp"
    assert schema["item_price"] == "decimal(10,2)"
    assert schema["batch_date"] == "date"


def test_creation_applies_the_partition_transform(spark, cfg, item_rows,
                                                  batch_date, silver_table):
    """months(creation_time_utc) lets a merge prune to one month's files.

    Partitioning is applied at creation only, so this is the one place it can
    be checked. Two rows in different months must land in two partitions.
    """
    fqn = silver_table("order_items")
    df = item_rows(
        {"LINEITEM_ID": "'1'", "CREATION_TIME_UTC": "'2020-04-21T18:56:44'"},
        {"LINEITEM_ID": "'2'", "CREATION_TIME_UTC": "'2020-05-02T10:00:00'"},
    )

    apply(spark, cfg, batch_date, "order_items", df)

    assert spark.sql(f"SELECT * FROM {fqn}.partitions").count() == 2


def test_same_month_rows_share_one_partition(spark, cfg, item_rows, batch_date,
                                             silver_table):
    """The counterpart: months() must NOT split within a month.

    Guards against someone putting days() back -- these two rows are ten days
    apart and belong in the same file.
    """
    fqn = silver_table("order_items")
    df = item_rows(
        {"LINEITEM_ID": "'1'", "CREATION_TIME_UTC": "'2020-04-01T00:00:00'"},
        {"LINEITEM_ID": "'2'", "CREATION_TIME_UTC": "'2020-04-11T00:00:00'"},
    )

    apply(spark, cfg, batch_date, "order_items", df)

    assert spark.sql(f"SELECT * FROM {fqn}.partitions").count() == 1


# --- merge ----------------------------------------------------------------

def test_merge_updates_matched_and_inserts_unmatched(spark, cfg, item_rows,
                                                     batch_date, silver_table):
    """SCD1: the matched row is overwritten, not duplicated or appended."""
    fqn = silver_table("order_items")
    apply(spark, cfg, batch_date, "order_items", item_rows(
        {"LINEITEM_ID": "'1'", "ITEM_PRICE": "'9.99'"},
        {"LINEITEM_ID": "'2'", "ITEM_PRICE": "'4.50'"},
    ))

    outcome = apply(spark, cfg, batch_date, "order_items", item_rows(
        {"LINEITEM_ID": "'1'", "ITEM_PRICE": "'11.25'"},   # changed
        {"LINEITEM_ID": "'3'", "ITEM_PRICE": "'2.00'"},    # new
    ))

    assert outcome == f"merged into {fqn}"
    assert spark.table(fqn).count() == 3

    def price(lineitem):
        return str(spark.sql(
            f"SELECT item_price FROM {fqn} WHERE lineitem_id='{lineitem}'"
        ).collect()[0][0])

    assert price("1") == "11.25"   # updated
    assert price("2") == "4.50"    # untouched by this batch
    assert price("3") == "2.00"    # inserted


def test_merge_is_idempotent(spark, cfg, item_rows, batch_date, silver_table):
    """Re-running a day is the documented repair procedure, so it must be safe."""
    fqn = silver_table("order_items")
    batch = item_rows({})

    apply(spark, cfg, batch_date, "order_items", batch)
    apply(spark, cfg, batch_date, "order_items", batch)
    apply(spark, cfg, batch_date, "order_items", batch)

    assert spark.table(fqn).count() == 1


# --- delete_insert --------------------------------------------------------

def test_delete_insert_replaces_only_the_scoped_parents(spark, cfg, option_rows,
                                                        batch_date, silver_table):
    """The scoped order is rewritten; every other order is left alone.

    This is the behaviour that makes delete_insert usable on a table with no
    unique key -- and the reason the extract must pull whole parents.
    """
    fqn = silver_table("order_item_options")
    apply(spark, cfg, batch_date, "order_item_options", option_rows(
        {"OPTION_PRICE": "1.00"},
        {"OPTION_PRICE": "1.50"},                       # duplicate key, allowed
        {"ORDER_ID": OTHER_ORDER, "OPTION_NAME": "'Fries'"},
    ))

    outcome = apply(spark, cfg, batch_date, "order_item_options", option_rows(
        {"OPTION_NAME": "'Small'", "OPTION_PRICE": "0.50"}))

    assert outcome == f"delete-inserted into {fqn}"

    def count_for(order):
        return spark.sql(
            f"SELECT count(*) FROM {fqn} WHERE order_id={order}"
        ).collect()[0][0]

    assert count_for(ORDER) == 1          # two rows replaced by one
    assert count_for(OTHER_ORDER) == 1    # outside the batch, untouched


def test_delete_insert_is_idempotent(spark, cfg, option_rows, batch_date,
                                     silver_table):
    fqn = silver_table("order_item_options")
    batch = option_rows({})

    apply(spark, cfg, batch_date, "order_item_options", batch)
    apply(spark, cfg, batch_date, "order_item_options", batch)

    assert spark.table(fqn).count() == 1


# --- replace --------------------------------------------------------------

def test_replace_rewrites_every_row(spark, cfg, dim_rows, batch_date,
                                    silver_table):
    """date_dim is small, so a merge buys nothing over rewriting it."""
    fqn = silver_table("date_dim")
    apply(spark, cfg, batch_date, "date_dim", dim_rows(
        {"date_key": "CAST('2020-04-21' AS date)"},
        {"date_key": "CAST('2020-04-22' AS date)"},
    ))

    outcome = apply(spark, cfg, batch_date, "date_dim", dim_rows(
        {"date_key": "CAST('2020-04-21' AS date)", "is_holiday": "true"}))

    assert outcome == f"replaced {fqn}"
    assert spark.table(fqn).count() == 1


# --- the contract ---------------------------------------------------------

def test_missing_required_column_raises(spark, cfg, rows, batch_date):
    """A column the contract promises but raw lacks must fail, not be dropped.

    pytest.raises asserts that the block raises -- the test fails if it does
    not. `match` is a regex checked against the message.
    """
    df = rows("ORDER_ID, LINEITEM_ID", f"{ORDER},'1'")

    with pytest.raises(ValueError, match="required_columns missing from raw"):
        apply_dynamic_transform(df, cfg.get_table_config("order_items"), batch_date)
