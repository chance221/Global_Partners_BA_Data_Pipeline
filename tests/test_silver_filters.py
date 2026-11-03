"""Tests for `silver_filters`: the rows silver does not carry at all.

Two groups, deliberately. The TableConfig tests are pure Python and run
without a session, because a config that would empty a table should fail in
milliseconds and not after a JVM start. The rest are marked `spark`.

Run only the fast ones with:  pytest -m "not spark"
"""

from dataclasses import replace

import pytest

from python_scripts.common.models.config_models import TableConfig
from python_scripts.raw_to_silver.raw_transforms import (
    apply_dynamic_transform, silver_exclusion, split_excluded,
)

DEV = "Alltown Fresh - DEVELOPMENT"
REAL = "Alltown Fresh"
PERKS = "Alltown Neighborhood Perks"


def app_names(frame):
    """The app_name values of a transformed frame, sorted."""
    return sorted(r.app_name for r in frame.select("app_name").collect())


def split(cfg, item_rows, batch_date, *overrides, table="order_items"):
    """Transform a batch of literal rows and split it the way the job does."""
    tableConfig = cfg.get_table_config(table)
    transformed = apply_dynamic_transform(
        item_rows(*overrides), tableConfig, batch_date)
    return split_excluded(transformed, tableConfig)


# --------------------------------------------------------------------------
# The real config, against the app it was written for
# --------------------------------------------------------------------------

@pytest.mark.spark
def test_development_app_is_excluded(cfg, item_rows, batch_date):
    kept, excluded = split(cfg, item_rows, batch_date,
                           {"APP_NAME": f"'{DEV}'"})

    assert kept.count() == 0
    assert excluded.count() == 1


@pytest.mark.spark
def test_the_two_production_apps_are_kept(cfg, item_rows, batch_date):
    """The regression guard against over-filtering.

    `Alltown Neighborhood Perks` is a real app on real stores -- 1,076 orders
    and $8,212.12 of genuine revenue -- so a pattern loose enough to catch it
    would quietly delete production money. app_name has three values, not two.
    """
    kept, excluded = split(cfg, item_rows, batch_date,
                           {"APP_NAME": f"'{REAL}'", "LINEITEM_ID": "'1'"},
                           {"APP_NAME": f"'{PERKS}'", "LINEITEM_ID": "'2'"})

    assert app_names(kept) == sorted([REAL, PERKS])
    assert excluded.count() == 0


@pytest.mark.spark
def test_match_is_case_insensitive_and_unanchored(cfg, item_rows, batch_date):
    """`(?i)DEVELOPMENT` is a substring match, which is the point.

    An exact comparison against the full app name would have to be revised the
    first time the source adds a suffix or changes the casing.
    """
    kept, excluded = split(cfg, item_rows, batch_date,
                           {"APP_NAME": "'alltown fresh - development'",
                            "LINEITEM_ID": "'1'"},
                           {"APP_NAME": "'Alltown Fresh - Development (iOS)'",
                            "LINEITEM_ID": "'2'"})

    assert kept.count() == 0
    assert excluded.count() == 2


@pytest.mark.spark
def test_null_app_name_is_kept(cfg, item_rows, batch_date):
    """The bug this guards is silent and total.

    `rlike` returns NULL against a NULL column, and `filter` drops a row whose
    predicate is NULL -- so negating a bare `rlike` without the coalesce would
    discard every row with a NULL app_name. app_name is nullable in the
    contract, so the loss would be real data, and nothing in the run log would
    say so.
    """
    kept, excluded = split(cfg, item_rows, batch_date,
                           {"APP_NAME": "CAST(NULL AS string)"})

    assert kept.count() == 1
    assert excluded.count() == 0


# --------------------------------------------------------------------------
# The split itself
# --------------------------------------------------------------------------

@pytest.mark.spark
def test_split_is_a_partition(cfg, item_rows, batch_date):
    """Nothing is lost and nothing is counted twice."""
    kept, excluded = split(cfg, item_rows, batch_date,
                           {"APP_NAME": f"'{REAL}'", "LINEITEM_ID": "'1'"},
                           {"APP_NAME": f"'{DEV}'", "LINEITEM_ID": "'2'"},
                           {"APP_NAME": "CAST(NULL AS string)",
                            "LINEITEM_ID": "'3'"})

    assert kept.count() + excluded.count() == 3
    assert kept.count() == 2


@pytest.mark.spark
def test_excluded_frame_keeps_the_full_schema(cfg, item_rows, batch_date):
    """So a caller could write or quarantine it later without reshaping it."""
    kept, excluded = split(cfg, item_rows, batch_date,
                           {"APP_NAME": f"'{DEV}'"})

    assert excluded.columns == kept.columns


@pytest.mark.spark
def test_a_table_with_no_filters_is_untouched(cfg, option_rows, batch_date):
    """order_item_options configures none, so the split must be a no-op.

    Asserted on the real config rather than a stub: the cost of the filter has
    to be zero for the tables that do not use it.
    """
    tableConfig = cfg.get_table_config("order_item_options")
    assert silver_exclusion(option_rows(), tableConfig) is None

    transformed = apply_dynamic_transform(option_rows(), tableConfig, batch_date)
    kept, excluded = split_excluded(transformed, tableConfig)

    assert kept.count() == 1
    assert excluded.count() == 0
    assert excluded.columns == kept.columns


# --------------------------------------------------------------------------
# Composition, and the ordering the docstring promises
# --------------------------------------------------------------------------

@pytest.mark.spark
def test_several_patterns_on_one_column_are_ored(cfg, item_rows, batch_date):
    tableConfig = replace(cfg.get_table_config("order_items"),
                          silver_filters={"app_name": ["DEVELOPMENT", "^Test"]})
    transformed = apply_dynamic_transform(
        item_rows({"APP_NAME": f"'{DEV}'", "LINEITEM_ID": "'1'"},
                  {"APP_NAME": "'Test Harness'", "LINEITEM_ID": "'2'"},
                  {"APP_NAME": f"'{REAL}'", "LINEITEM_ID": "'3'"}),
        tableConfig, batch_date)

    kept, excluded = split_excluded(transformed, tableConfig)

    assert app_names(kept) == [REAL]
    assert excluded.count() == 2


@pytest.mark.spark
def test_several_columns_are_ored(cfg, item_rows, batch_date):
    tableConfig = replace(
        cfg.get_table_config("order_items"),
        silver_filters={"app_name": ["DEVELOPMENT"],
                        "restaurant_id": ["^test-store$"]})
    transformed = apply_dynamic_transform(
        item_rows({"APP_NAME": f"'{DEV}'", "LINEITEM_ID": "'1'"},
                  {"RESTAURANT_ID": "'test-store'", "LINEITEM_ID": "'2'"},
                  {"LINEITEM_ID": "'3'"}),
        tableConfig, batch_date)

    kept, excluded = split_excluded(transformed, tableConfig)

    assert kept.count() == 1
    assert excluded.count() == 2


@pytest.mark.spark
def test_filters_see_the_cleaned_value_not_the_source_spelling(
        cfg, item_rows, batch_date):
    """Filters run AFTER silver_cleaners, and this pins that order.

    `item_category` has a cleaner rewriting the childrens-menu label. A filter
    written against the repaired label must therefore match a row the source
    spelled the broken way -- which is only true if the filter runs second.
    Reverse the two and this test fails, which is the whole reason it exists.
    """
    repaired = cfg.get_table_config("order_items").silver_cleaners[
        "item_category"][-1]
    broken, fixed = repaired[0].strip("^$"), repaired[1]

    tableConfig = replace(cfg.get_table_config("order_items"),
                          silver_filters={"item_category": [f"^{fixed}$"]})
    transformed = apply_dynamic_transform(
        item_rows({"ITEM_CATEGORY": f"'{broken}'"}), tableConfig, batch_date)

    kept, excluded = split_excluded(transformed, tableConfig)

    assert kept.count() == 0
    assert excluded.count() == 1


# --------------------------------------------------------------------------
# Config validation -- no Spark, because an unsafe config must fail instantly
# --------------------------------------------------------------------------

def a_config(**overrides) -> dict:
    base = dict(
        source_table="dbo.t", load_type="static", business_keys=["ID"],
        required_columns={"ID": "string", "APP_NAME": "string"},
    )
    return {**base, **overrides}


def test_empty_pattern_string_is_rejected():
    """An empty pattern matches every row, emptying the table successfully."""
    with pytest.raises(ValueError, match="would drop the whole table"):
        TableConfig(**a_config(silver_filters={"app_name": [""]}))


def test_empty_pattern_list_is_rejected():
    with pytest.raises(ValueError, match="has no patterns"):
        TableConfig(**a_config(silver_filters={"app_name": []}))


def test_non_string_pattern_is_rejected():
    with pytest.raises(ValueError, match="must be a non-empty string"):
        TableConfig(**a_config(silver_filters={"app_name": [None]}))


def test_filtering_an_uncarried_column_is_rejected():
    """Worse than a no-op: the exclusion would silently never happen."""
    with pytest.raises(ValueError, match="absent from required_columns"):
        TableConfig(**a_config(silver_filters={"nope": ["x"]}))


def test_no_filters_is_valid_and_defaults_empty():
    assert TableConfig(**a_config()).silver_filters == {}


def test_the_shipped_config_still_filters_the_development_app(cfg):
    """A config edit that drops the filter should fail here, not in the data."""
    filters = cfg.get_table_config("order_items").silver_filters

    assert "app_name" in filters
    assert any("DEVELOPMENT" in p.upper() for p in filters["app_name"])
