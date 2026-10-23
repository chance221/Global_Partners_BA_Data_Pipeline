"""Fixtures shared by every test in this directory.

pytest discovers this file automatically -- tests never import it. Any function
decorated with @pytest.fixture here can be requested by a test simply by
naming it as a parameter.

The Spark session is the reason this file exists. Starting one costs roughly
half a minute (JVM boot, then Ivy resolving the Iceberg and JDBC jars), so it
is built once per test run and shared by every test, rather than once per test.
"""

import os
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from python_scripts.common.config_loader import load_config
from python_scripts.raw_to_silver.silver_write import qualified_table_name
from python_scripts.silver_to_gold.gold_write import gold_table_name
from python_scripts.spark.spark_session import create_spark_session

# Protocol paired with `local_cfg` below. build_path renders
# "{protocol}://{bucket}/..." so a local run needs "file" here.
LOCAL_PROTOCOL = "file"


@pytest.fixture(scope="session")
def spark(tmp_path_factory):
    """One Spark session for the whole test run.

    scope="session" is what makes it shared. The default scope is "function",
    which would rebuild it for every test and turn a 20-second suite into a
    20-minute one.

    Deliberately the real `create_spark_session()` rather than a hand-built
    session, so the Iceberg catalog configuration is itself under test.

ICEBERG_CATALOG_TYPE and ICEBERG_WAREHOUSE pin the run to a hadoop
    catalog in a throwaway directory. Both matter:

      * the type, because config now selects a Glue catalog, and a test suite
        that reached AWS would be slow, need credentials, and write to a real
        account.
      * the warehouse, because the default is the repo's warehouse/, and the
        teardown that used to clean up after a run would delete the silver and
        gold tables a real job had written.

    pytest manages retention of tmp_path_factory directories, so there is
    nothing to clean up here.

    The variable has to be set before `create_spark_session()` is called,
    because the warehouse path is fixed when the JVM starts.

    Everything before `yield` is setup; everything after is teardown, and it
    runs even if tests fail.
    """
    warehouse = tmp_path_factory.mktemp("iceberg_warehouse")
    os.environ["ICEBERG_CATALOG_TYPE"] = "hadoop"
    os.environ["ICEBERG_WAREHOUSE"] = f"file:///{warehouse.as_posix()}"

    session, _protocol = create_spark_session()
    session.sparkContext.setLogLevel("ERROR")

    yield session

    session.stop()


@pytest.fixture(scope="session")
def cfg():
    """The real local config, so tests exercise the same table contracts as a run."""
    return load_config("local")


@pytest.fixture
def local_cfg(cfg, tmp_path):
    """The real config with `bucket` pointed at a throwaway local directory.

    Lets a test exercise the genuine write path -- build_path, Spark's parquet
    writer, partition layout -- without touching S3, needing credentials, or
    leaving anything behind. `tmp_path` is a pytest built-in fixture giving a
    fresh empty directory per test.

    The leading "/" matters: build_path produces "{protocol}://{bucket}/...",
    so a Windows path has to become "/C:/Users/..." for the result to be
    "file:///C:/Users/..." rather than "file://C:/...", where Hadoop would read
    "C:" as a hostname.
    """
    return replace(cfg, bucket=f"/{tmp_path.as_posix()}")


@pytest.fixture
def batch_date():
    """A fixed logical batch date.

    Hard-coded rather than `datetime.now()`: a test whose expected values
    depend on today's date fails on a different day for no useful reason.
    """
    return datetime(2020, 4, 21, tzinfo=timezone.utc)


# One SQL literal per contract column, so a test states only what it cares
# about. Adding a column to required_columns means adding it here once, instead
# of editing every row literal in every test.
ITEM_DEFAULTS = {
    "ORDER_ID": "'5e9f41ec505ee9c65088f162'",
    "LINEITEM_ID": "'1'",
    "RESTAURANT_ID": "'r1'",
    "CREATION_TIME_UTC": "'2020-04-21T18:56:44'",
    "ITEM_PRICE": "'9.99'",
    "ITEM_QUANTITY": "1",
    "APP_NAME": "'app'",
    "USER_ID": "'u1'",
    "CURRENCY": "'USD'",
    "ITEM_NAME": "'Fries'",
    "ITEM_CATEGORY": "'Side'",
    "INSERTED_AT": "CAST('2026-10-16 17:03:39' AS timestamp)",
    "IS_LOYALTY": "false",
}

OPTION_DEFAULTS = {
    "ORDER_ID": "'5e9f41ec505ee9c65088f162'",
    "LINEITEM_ID": "'1'",
    "OPTION_GROUP_NAME": "'Size'",
    "OPTION_NAME": "'Large'",
    "OPTION_PRICE": "1.00",
    "OPTION_QUANTITY": "CAST(1 AS smallint)",
}

DIM_DEFAULTS = {
    "date_key": "CAST('2020-04-21' AS date)",
    "year": "CAST(2020 AS smallint)",
    "month": "CAST(4 AS smallint)",
    "is_weekend": "false",
    "is_holiday": "false",
    # Cast, not a bare NULL: an untyped NULL literal is Spark's `void` type,
    # which Iceberg v2 rejects as "unknown is not supported until v3".
    "holiday_name": "CAST(NULL AS string)",
}


def _build(spark, defaults: dict[str, str], overrides: tuple[dict, ...]):
    columns = list(defaults)
    values = []
    for row in overrides or ({},):
        merged = {**defaults, **row}
        unknown = set(row) - set(defaults)
        if unknown:
            raise KeyError(f"not contract columns: {sorted(unknown)}")
        values.append("(" + ", ".join(merged[c] for c in columns) + ")")

    return spark.sql(
        f"SELECT * FROM VALUES {', '.join(values)} AS t({', '.join(columns)})"
    )


@pytest.fixture
def item_rows(spark):
    """order_items rows with every contract column; pass dicts to override."""
    return lambda *overrides: _build(spark, ITEM_DEFAULTS, overrides)


@pytest.fixture
def option_rows(spark):
    """order_item_options rows with every contract column."""
    return lambda *overrides: _build(spark, OPTION_DEFAULTS, overrides)


@pytest.fixture
def dim_rows(spark):
    """date_dim rows with every contract column."""
    return lambda *overrides: _build(spark, DIM_DEFAULTS, overrides)


@pytest.fixture
def rows(spark):
    """Build a DataFrame from literal SQL values.

    `spark.sql(... FROM VALUES ...)` rather than `spark.createDataFrame`,
    because createDataFrame serializes Python objects to an executor and so
    needs a Python worker process -- which needs PYSPARK_PYTHON set, and it is
    not set in this environment. VALUES is evaluated entirely inside the JVM.

    A fixture can depend on another fixture just by naming it, which is how
    this one gets `spark`.
    """
    def _rows(columns: str, *value_rows: str):
        values = ", ".join(f"({row})" for row in value_rows)
        return spark.sql(f"SELECT * FROM VALUES {values} AS t({columns})")

    return _rows


@pytest.fixture
def silver_table(spark, cfg):
    """Hand a test an empty silver table, and drop it afterwards.

    Iceberg tables persist in the warehouse between tests, so without this a
    test would inherit whatever the previous one wrote and pass or fail for the
    wrong reason. Returning a function (rather than a value) lets one test ask
    for several tables.
    """
    requested = []

    def _table(name: str) -> str:
        fqn = qualified_table_name(cfg, name)
        spark.sql(f"DROP TABLE IF EXISTS {fqn}")
        requested.append(fqn)
        return fqn

    yield _table

    for fqn in requested:
        spark.sql(f"DROP TABLE IF EXISTS {fqn}")


@pytest.fixture
def gold_table(spark, cfg):
    """The gold counterpart of `silver_table`: an empty table, dropped after.

    Same reason it exists for silver -- an Iceberg table outlives the test that
    created it, so a test that asserts "no gold table was written" has to know
    none was there to begin with.
    """
    requested = []

    def _table(name: str) -> str:
        fqn = gold_table_name(cfg, name)
        spark.sql(f"DROP TABLE IF EXISTS {fqn}")
        requested.append(fqn)
        return fqn

    yield _table

    for fqn in requested:
        spark.sql(f"DROP TABLE IF EXISTS {fqn}")
