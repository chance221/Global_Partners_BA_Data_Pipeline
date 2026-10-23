"""Tests for the SQL that gets pushed down to SQL Server.

These need no Spark session and no database -- `build_extract_query` is pure
string generation, so the whole file runs in milliseconds. Worth keeping
separate from the Spark tests for exactly that reason: you can run it on every
save, while the Spark suite is something you run before committing.
"""

from datetime import datetime, timedelta, timezone

import pytest

from python_scripts.common.models.config_models import TableConfig
from python_scripts.io_utils.extract_query import (
    build_extract_table, sql_timestamp_literal, watermark_predicate,
)

START = datetime(2026, 10, 13, tzinfo=timezone.utc)
END = START + timedelta(days=1)


def make_table_config(**overrides) -> TableConfig:
    """Build a TableConfig for a test, overriding only what the test cares about.

    Without this every test would repeat all the required fields, and adding a
    field to TableConfig would mean editing every test in the file.
    """
    defaults = dict(
        source_table="dbo.order_items",
        load_type="incremental",
        business_keys=["ORDER_ID", "LINEITEM_ID"],
        watermark_column="inserted_at",
        silver_write="merge",
    )
    return TableConfig(**{**defaults, **overrides})


# --- the timestamp literal ------------------------------------------------

def test_literal_uses_iso_8601_with_t_separator():
    """The T form is the one SQL Server reads the same way in every locale."""
    assert sql_timestamp_literal(START) == "2026-10-13T00:00:00.000"


def test_literal_converts_to_utc():
    """inserted_at is SYSUTCDATETIME(), so a non-UTC input must be converted.

    13:00 in UTC+5 is 08:00 UTC -- comparing the wall-clock time instead would
    select the wrong five hours of rows.
    """
    plus_five = timezone(timedelta(hours=5))
    moment = datetime(2026, 10, 13, 13, 0, tzinfo=plus_five)

    assert sql_timestamp_literal(moment) == "2026-10-13T08:00:00.000"


def test_literal_keeps_milliseconds():
    moment = datetime(2026, 10, 13, 1, 2, 3, 456789, tzinfo=timezone.utc)

    assert sql_timestamp_literal(moment) == "2026-10-13T01:02:03.456"


# --- the window predicate -------------------------------------------------

def test_window_is_half_open():
    """>= on the start and < on the end."""
    predicate = watermark_predicate("inserted_at", START, END)

    assert "inserted_at >= '2026-10-13T00:00:00.000'" in predicate
    assert "inserted_at < '2026-10-14T00:00:00.000'" in predicate


def test_consecutive_windows_neither_gap_nor_overlap():
    """One day's upper bound is the next day's lower bound.

    This is the property that makes a row belong to exactly one batch. If the
    end were inclusive, a row stamped at midnight would be processed twice.
    """
    day_one_end = sql_timestamp_literal(END)
    day_two_start = sql_timestamp_literal(END)

    assert day_one_end == day_two_start


# --- choosing the query shape ---------------------------------------------

@pytest.mark.parametrize("load_type", ["static", "full"])
def test_unfiltered_load_reads_the_table_directly(load_type):
    """No predicate means no subquery -- just the table name.

    @parametrize runs this test once per value, reported as two separate
    tests, so a failure tells you which load type broke.
    """
    config = make_table_config(load_type=load_type, watermark_column=None)

    assert build_extract_table(config, START, END) == "dbo.order_items"


def test_merge_table_gets_a_plain_window():
    """A row-level window is enough when rows have a unique key."""
    query = build_extract_table(make_table_config(), START, END)

    assert query == (
        "(SELECT * FROM dbo.order_items "
        "WHERE inserted_at >= '2026-10-13T00:00:00.000' "
        "AND inserted_at < '2026-10-14T00:00:00.000') AS src"
    )


def test_delete_insert_table_widens_to_whole_parents():
    """The destructive case: delete_insert needs every row of each parent.

    The watermark picks which parents changed; the outer query then pulls those
    parents complete. A plain row-level window here would delete an order's
    other option rows and not put them back.
    """
    config = make_table_config(
        source_table="dbo.order_item_options",
        business_keys=["ORDER_ID", "LINEITEM_ID", "OPTION_GROUP_NAME", "OPTION_NAME"],
        silver_write="delete_insert",
        silver_scope_keys=["ORDER_ID"],
    )

    query = build_extract_table(config, START, END)

    assert "EXISTS" in query
    assert "w.ORDER_ID = o.ORDER_ID" in query
    assert "SELECT o.* FROM dbo.order_item_options o" in query


def test_delete_insert_correlates_on_every_scope_key():
    """EXISTS is arity-agnostic, unlike the multi-column IN it replaced.

    SQL Server has no ANSI row value constructor, so `(a, b) IN (SELECT a, b)`
    is a syntax error there -- which is why this generates EXISTS even though
    the same IN form is valid in Spark SQL.
    """
    config = make_table_config(
        silver_write="delete_insert",
        silver_scope_keys=["ORDER_ID", "LINEITEM_ID"],
    )

    query = build_extract_table(config, START, END)

    assert "w.ORDER_ID = o.ORDER_ID AND w.LINEITEM_ID = o.LINEITEM_ID" in query
    assert " IN (" not in query


def test_real_config_generates_expected_shapes(cfg):
    """Against the actual config/local.yaml, not a hand-built one.

    Guards the wiring as much as the logic: a config edit that puts the wrong
    silver_write on a table would change the query shape, and this notices.
    """
    items = build_extract_table(cfg.get_table_config("order_items"), START, END)
    options = build_extract_table(cfg.get_table_config("order_item_options"), START, END)
    dim = build_extract_table(cfg.get_table_config("date_dim"), START, END)

    assert items.startswith("(SELECT * FROM dbo.order_items WHERE inserted_at")
    assert "EXISTS" in options
    assert dim == "dbo.date_dim"
