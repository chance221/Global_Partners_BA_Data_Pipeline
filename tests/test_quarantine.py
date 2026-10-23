"""Tests for which rows get held back and why.

Marked `spark` because every test here needs a real session. Run only the fast
tests with:  pytest -m "not spark"
"""

import pytest

from python_scripts.io_utils.paths import build_path
from python_scripts.io_utils.quarantine import (
    REASON_COLUMN, split_rejects, write_quarantine,
)
from python_scripts.raw_to_silver.raw_transforms import (
    apply_dynamic_transform, silver_reject_reasons,
)
from python_scripts.raw_to_silver.silver_write import write_silver
from tests.conftest import LOCAL_PROTOCOL

pytestmark = pytest.mark.spark   # applies the marker to every test in the file

ITEM_COLUMNS = ("ORDER_ID, LINEITEM_ID, RESTAURANT_ID, "
                "CREATION_TIME_UTC, ITEM_PRICE, ITEM_QUANTITY")
OPTION_COLUMNS = ("ORDER_ID, LINEITEM_ID, OPTION_GROUP_NAME, "
                  "OPTION_NAME, OPTION_PRICE, OPTION_QUANTITY")

GOOD_ITEM = "'5e9f41ec505ee9c65088f162','1','r1','2020-04-21T18:56:44','9.99',1"


def reasons_for(df, table_config):
    """Run the checks and return the reason strings of the rejected rows."""
    _accepted, rejected = split_rejects(df, silver_reject_reasons(df, table_config))
    return [r[REASON_COLUMN] for r in rejected.select(REASON_COLUMN).collect()]


def test_clean_rows_are_all_accepted(rows, cfg):
    df = rows(ITEM_COLUMNS, GOOD_ITEM,
              "'5e9f41ec505ee9c65088f162','2','r1','2020-04-21T19:00:00','4.50',2")

    accepted, rejected = split_rejects(
        df, silver_reject_reasons(df, cfg.get_table_config("order_items"))
    )

    assert accepted.count() == 2
    assert rejected.count() == 0


def test_accepted_rows_keep_their_original_columns(rows, cfg):
    """The reason column is an artefact of the split and must not leak through."""
    df = rows(ITEM_COLUMNS, GOOD_ITEM)

    accepted, _ = split_rejects(
        df, silver_reject_reasons(df, cfg.get_table_config("order_items"))
    )

    assert REASON_COLUMN not in accepted.columns


def test_unparseable_timestamp_is_held(rows, cfg):
    """A US-format date cannot become a timestamp.

    Under Spark 4's ANSI defaults a bare cast would abort the whole run, so
    this is the check that turns a job failure into one held row.
    """
    df = rows(ITEM_COLUMNS,
              "'5e9f41ec505ee9c65088f162','1','r1','04/21/2020 18:56:44','9.99',1")

    assert reasons_for(df, cfg.get_table_config("order_items")) == [
        "CREATION_TIME_UTC is not castable to timestamp"
    ]


def test_non_numeric_price_is_held(rows, cfg):
    df = rows(ITEM_COLUMNS,
              "'5e9f41ec505ee9c65088f162','1','r1','2020-04-21T18:56:44','abc',1")

    assert reasons_for(df, cfg.get_table_config("order_items")) == [
        "ITEM_PRICE is not castable to decimal(10,2)"
    ]


def test_null_in_a_business_key_is_held(rows, cfg):
    """`ON t.k = s.k` is never true for null, so the row would insert forever."""
    df = rows(ITEM_COLUMNS,
              "'5e9f41ec505ee9c65088f162',NULL,'r1','2020-04-21T18:56:44','9.99',1")

    assert reasons_for(df, cfg.get_table_config("order_items")) == [
        "LINEITEM_ID is null"
    ]


def test_null_input_is_not_a_cast_failure(rows, cfg):
    """A value that was already null stays null -- nothing failed to convert."""
    df = rows(ITEM_COLUMNS,
              "'5e9f41ec505ee9c65088f162','1','r1',NULL,'9.99',1")

    assert reasons_for(df, cfg.get_table_config("order_items")) == []


def test_every_member_of_a_duplicate_group_is_held(rows, cfg):
    """Both copies, not one.

    With no ordering column in the source there is no non-arbitrary way to pick
    the survivor, and row_number() would choose differently between runs --
    which SCD1 would see as a change every time.
    """
    df = rows(ITEM_COLUMNS,
              "'5e9f41ec505ee9c65088f162','1','r1','2020-04-21T18:56:44','3.00',1",
              "'5e9f41ec505ee9c65088f162','1','r1','2020-04-21T18:56:44','7.00',1")

    held = reasons_for(df, cfg.get_table_config("order_items"))

    assert len(held) == 2
    assert all("duplicate business key" in reason for reason in held)


def test_delete_insert_tolerates_duplicate_keys(rows, cfg):
    """order_item_options has no unique key, which is why it uses delete_insert.

    The mirror image of the test above: the same duplication that disqualifies
    a merge table is legitimate here, so the duplicate check must not run.
    """
    df = rows(OPTION_COLUMNS,
              "'5e9f41ec505ee9c65088f162','1','Size','Large',1.00,1",
              "'5e9f41ec505ee9c65088f162','1','Size','Large',1.50,1")

    accepted, rejected = split_rejects(
        df, silver_reject_reasons(df, cfg.get_table_config("order_item_options"))
    )

    assert accepted.count() == 2
    assert rejected.count() == 0


def test_a_row_reports_all_of_its_defects(rows, cfg):
    """Reasons are joined, not short-circuited, so one pass finds everything."""
    df = rows(ITEM_COLUMNS,
              "'5e9f41ec505ee9c65088f162',NULL,'r1','bad-ts','abc',1")

    held = reasons_for(df, cfg.get_table_config("order_items"))

    assert len(held) == 1
    assert held[0].split("; ") == [
        "CREATION_TIME_UTC is not castable to timestamp",
        "ITEM_PRICE is not castable to decimal(10,2)",
        "LINEITEM_ID is null",
    ]


def test_bad_rows_removed_so_the_strict_cast_survives(item_rows, cfg, batch_date):
    """The point of the whole mechanism.

    Detection uses try_cast; the transform still uses cast, which raises under
    ANSI mode. This passes only because the bad row was removed first -- it is
    what proves the two agree.
    """
    df = item_rows(
        {},
        {"LINEITEM_ID": "'2'", "CREATION_TIME_UTC": "'04/21/2020 00:00:00'"},
    )
    table_config = cfg.get_table_config("order_items")

    accepted, _ = split_rejects(df, silver_reject_reasons(df, table_config))
    transformed = apply_dynamic_transform(accepted, table_config, batch_date)

    assert transformed.count() == 1
    assert str(transformed.collect()[0]["item_price"]) == "9.99"


# --- rejected rows actually reaching the quarantine zone -------------------
#
# Everything above stops at split_rejects, which only proves bad rows were
# *identified*. These follow them through write_quarantine onto disk and read
# them back, which is the part that would silently do nothing if the path,
# the metadata or the writer were wrong.

BAD_TIMESTAMP = f"'{'5e9f41ec505ee9c65088f162'}','9','r1','04/21/2020 18:56:44','1.00',1"
BAD_PRICE = f"'{'5e9f41ec505ee9c65088f162'}','8','r1','2020-04-21T18:56:44','abc',1"


def quarantine_path(local_cfg, table, batch_date, stage):
    """The path write_quarantine will have written to."""
    return build_path(LOCAL_PROTOCOL, local_cfg.bucket, local_cfg.env,
                      local_cfg.layer("quarantine"), f"{stage}/{table}",
                      batch_date, "full")


def test_bad_rows_are_written_to_the_quarantine_zone(spark, local_cfg, rows,
                                                     batch_date):
    """The whole point: a bad row ends up on disk where someone can review it."""
    df = rows(ITEM_COLUMNS, GOOD_ITEM, BAD_TIMESTAMP, BAD_PRICE)
    table_config = local_cfg.get_table_config("order_items")

    _accepted, rejected = split_rejects(df, silver_reject_reasons(df, table_config))
    held = write_quarantine(rejected, LOCAL_PROTOCOL, local_cfg, "order_items",
                            batch_date, "raw_to_silver")

    assert held == 2

    landed = spark.read.parquet(
        quarantine_path(local_cfg, "order_items", batch_date, "raw_to_silver"))
    assert landed.count() == 2
    assert sorted(r["LINEITEM_ID"] for r in landed.collect()) == ["8", "9"]


def test_quarantined_rows_keep_the_values_the_source_had(spark, local_cfg, rows,
                                                         batch_date):
    """Not a converted copy -- the unparseable text itself must survive.

    If the split ran after the cast, this column would be null and the file
    would be useless for working out what went wrong.
    """
    df = rows(ITEM_COLUMNS, BAD_TIMESTAMP)
    table_config = local_cfg.get_table_config("order_items")

    _, rejected = split_rejects(df, silver_reject_reasons(df, table_config))
    write_quarantine(rejected, LOCAL_PROTOCOL, local_cfg, "order_items",
                     batch_date, "raw_to_silver")

    landed = spark.read.parquet(
        quarantine_path(local_cfg, "order_items", batch_date, "raw_to_silver")
    ).collect()[0]

    assert landed["CREATION_TIME_UTC"] == "04/21/2020 18:56:44"
    assert landed["ITEM_PRICE"] == "1.00"


def test_quarantined_rows_carry_provenance(spark, local_cfg, rows, batch_date):
    """Reason, stage, table and batch date -- enough to triage without guessing."""
    df = rows(ITEM_COLUMNS, BAD_PRICE)
    table_config = local_cfg.get_table_config("order_items")

    _, rejected = split_rejects(df, silver_reject_reasons(df, table_config))
    write_quarantine(rejected, LOCAL_PROTOCOL, local_cfg, "order_items",
                     batch_date, "raw_to_silver")

    landed = spark.read.parquet(
        quarantine_path(local_cfg, "order_items", batch_date, "raw_to_silver")
    ).collect()[0]

    assert landed[REASON_COLUMN] == "ITEM_PRICE is not castable to decimal(10,2)"
    assert landed["_quarantine_stage"] == "raw_to_silver"
    assert landed["_quarantine_table"] == "order_items"
    assert str(landed["_batch_date"]) == "2020-04-21"


def test_each_stage_writes_to_its_own_location(spark, local_cfg, rows, batch_date):
    """raw_ingest and raw_to_silver rejects must not land on top of each other.

    Both write with overwrite, so a shared path would mean the second stage
    silently erased the first stage's rejects.
    """
    df = rows(ITEM_COLUMNS, BAD_PRICE)
    table_config = local_cfg.get_table_config("order_items")
    _, rejected = split_rejects(df, silver_reject_reasons(df, table_config))

    write_quarantine(rejected, LOCAL_PROTOCOL, local_cfg, "order_items",
                     batch_date, "raw_ingest")
    write_quarantine(rejected, LOCAL_PROTOCOL, local_cfg, "order_items",
                     batch_date, "raw_to_silver")

    for stage in ("raw_ingest", "raw_to_silver"):
        landed = spark.read.parquet(
            quarantine_path(local_cfg, "order_items", batch_date, stage))
        assert landed.count() == 1
        assert landed.collect()[0]["_quarantine_stage"] == stage


def test_rerunning_a_batch_replaces_its_quarantine(spark, local_cfg, rows,
                                                   batch_date):
    """Re-running a day is the repair procedure, so rejects must not accumulate."""
    table_config = local_cfg.get_table_config("order_items")

    two_bad = rows(ITEM_COLUMNS, BAD_TIMESTAMP, BAD_PRICE)
    _, rejected = split_rejects(two_bad, silver_reject_reasons(two_bad, table_config))
    write_quarantine(rejected, LOCAL_PROTOCOL, local_cfg, "order_items",
                     batch_date, "raw_to_silver")

    # The data was fixed at source, so the re-run holds only one row.
    one_bad = rows(ITEM_COLUMNS, GOOD_ITEM, BAD_PRICE)
    _, rejected = split_rejects(one_bad, silver_reject_reasons(one_bad, table_config))
    write_quarantine(rejected, LOCAL_PROTOCOL, local_cfg, "order_items",
                     batch_date, "raw_to_silver")

    landed = spark.read.parquet(
        quarantine_path(local_cfg, "order_items", batch_date, "raw_to_silver"))
    assert landed.count() == 1


def test_a_clean_batch_leaves_no_quarantine_behind(local_cfg, rows, batch_date):
    """0 rejects writes nothing, rather than an empty partition to review."""
    df = rows(ITEM_COLUMNS, GOOD_ITEM)
    table_config = local_cfg.get_table_config("order_items")

    _, rejected = split_rejects(df, silver_reject_reasons(df, table_config))
    held = write_quarantine(rejected, LOCAL_PROTOCOL, local_cfg, "order_items",
                            batch_date, "raw_to_silver")

    assert held == 0
    from pathlib import Path
    assert not (Path(local_cfg.bucket.lstrip("/")) / local_cfg.env
                / "quarantine").exists()


def test_bad_rows_quarantined_while_good_rows_reach_silver(spark, local_cfg, item_rows,
                                                           batch_date, silver_table):
    """One pass, both destinations -- the behaviour run_raw_to_silver depends on.

    A batch containing bad rows must still deliver the good ones, which is the
    difference between quarantine and simply failing the table.
    """
    fqn = silver_table("order_items")
    df = item_rows(
        {},
        {"LINEITEM_ID": "'9'", "CREATION_TIME_UTC": "'04/21/2020 18:56:44'"},
        {"LINEITEM_ID": "'8'", "ITEM_PRICE": "'abc'"},
    )
    table_config = local_cfg.get_table_config("order_items")

    accepted, rejected = split_rejects(df, silver_reject_reasons(df, table_config))
    held = write_quarantine(rejected, LOCAL_PROTOCOL, local_cfg, "order_items",
                            batch_date, "raw_to_silver")
    write_silver(spark, apply_dynamic_transform(accepted, table_config, batch_date),
                 local_cfg, "order_items", table_config)

    assert held == 2
    assert spark.table(fqn).count() == 1
    assert str(spark.table(fqn).collect()[0]["item_price"]) == "9.99"
