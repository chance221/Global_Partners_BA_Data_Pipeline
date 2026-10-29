"""Contract tests that hold for every gold transform.
"""

import pytest

from python_scripts.silver_to_gold.transforms import TRANSFORMS

pytestmark = pytest.mark.spark

GOLD_TABLES = sorted(TRANSFORMS)


# --- inputs ---------------------------------------------------------------

@pytest.fixture
def build_inputs(cfg, dim_rows):
    
    calendar = dim_rows(
        {"date_key": "CAST('2020-04-20' AS date)", "day_of_week": "'Monday'",
         "week": "CAST(17 AS smallint)"},
        {"date_key": "CAST('2020-04-21' AS date)", "is_holiday": "true",
         "holiday_name": "'Test Holiday'"},
        {"date_key": "CAST('2020-04-22' AS date)", "day_of_week": "'Wednesday'",
         "week": "CAST(17 AS smallint)"},
    )
    supplied = {"date_dim": calendar}

    def inputs(table, df):
        depends_on = cfg.get_gold_config(table).depends_on
        unsupported = set(depends_on) - {"order_items"} - set(supplied)
        if unsupported:
            raise AssertionError(
                f"{table} depends on {sorted(unsupported)}, which these tests "
                f"have no fixture for; add one rather than passing an empty frame"
            )
        return {name: supplied.get(name, df) for name in depends_on}

    return inputs


@pytest.fixture
def degenerate(item_rows):
    """Batches that are valid silver but carry nothing a transform can group."""
    return {
        "empty table": item_rows().where("1=0"),
        "all user_id null": item_rows({"USER_ID": "CAST(NULL AS string)"}),
        "all user_id blank": item_rows({"USER_ID": "''"}),
    }


# --- the registry and config agree ----------------------------------------

def test_every_gold_table_in_config_has_a_transform(cfg):
    """A config entry with no transform fails at run time, in Glue, mid-job."""
    assert set(cfg.gold) - set(TRANSFORMS) == set()


def test_every_transform_is_declared_in_config(cfg):
    """The reverse: an unreachable transform is dead code, and `--all` skips it."""
    assert set(TRANSFORMS) - set(cfg.gold) == set()


# --- building -------------------------------------------------------------

@pytest.mark.parametrize("table", GOLD_TABLES)
def test_transform_builds_from_one_row(spark, cfg, batch_date, item_rows,
                                       build_inputs, table):
    
    out = TRANSFORMS[table](build_inputs(table, item_rows({})), batch_date)

    assert out.count() >= 0
    assert out.columns


@pytest.mark.parametrize("table", GOLD_TABLES)
def test_transform_survives_degenerate_input(cfg, batch_date, degenerate,
                                             build_inputs, table):
    """Nothing to aggregate is an empty answer, never an exception.

    An empty gold table is correct here and a failed run is not: the batch is
    well-formed, it just has no identifiable customers or no rows at all.
    """
    for label, df in degenerate.items():
        out = TRANSFORMS[table](build_inputs(table, df), batch_date)
        assert out.count() >= 0, f"{table} produced no frame for {label}"


@pytest.mark.parametrize("table", GOLD_TABLES)
def test_degenerate_input_keeps_the_full_schema(cfg, batch_date, degenerate,
                                                item_rows, build_inputs,
                                                table):
    """An empty result must still carry every column.
    """
    full = TRANSFORMS[table](build_inputs(table, item_rows({})), batch_date)

    for label, df in degenerate.items():
        out = TRANSFORMS[table](build_inputs(table, df), batch_date)
        assert out.columns == full.columns, f"{table} changed schema on {label}"


# --- the declared grain ---------------------------------------------------

@pytest.mark.parametrize("table", GOLD_TABLES)
def test_grain_keys_exist_in_the_output(spark, cfg, batch_date, item_rows,
                                       build_inputs, table):
    """config names the grain; the transform has to actually emit it."""
    gold_config = cfg.get_gold_config(table)
    out = TRANSFORMS[table](build_inputs(table, item_rows({})), batch_date)

    missing = [k for k in gold_config.grain_keys if k not in out.columns]
    assert not missing, f"{table} does not emit grain_keys {missing}"


@pytest.mark.parametrize("table", GOLD_TABLES)
def test_grain_is_unique(spark, cfg, batch_date, item_rows, build_inputs,
                         table):
    """One row per grain value, over a batch built to collide.
    """
    rows = item_rows(
        {"ORDER_ID": "'o1'", "LINEITEM_ID": "'1'"},
        {"ORDER_ID": "'o1'", "LINEITEM_ID": "'2'"},
        {"ORDER_ID": "'o2'", "LINEITEM_ID": "'1'"},
    )
    grain = cfg.get_gold_config(table).grain_keys
    out = TRANSFORMS[table](build_inputs(table, rows), batch_date).cache()

    try:
        assert out.count() == out.select(*grain).distinct().count(), \
            f"{table} emits duplicate {grain}"
    finally:
        out.unpersist()


@pytest.mark.parametrize("table", GOLD_TABLES)
def test_grain_is_never_null(spark, cfg, batch_date, item_rows, build_inputs,
                             table):
    """A null grain key breaks MERGE matching and every downstream join."""
    grain = cfg.get_gold_config(table).grain_keys
    out = TRANSFORMS[table](build_inputs(table, item_rows({})), batch_date)

    predicate = " OR ".join(f"{k} IS NULL" for k in grain)
    assert out.where(predicate).count() == 0, f"{table} emits a null {grain}"


# --- category materiality -------------------------------------------------

# A category's share is of ALL-TIME sales, so these build a batch whose
# category split is known and check the arithmetic rather than a magic number.
MATERIALITY_TABLES = ["category_period_sales", "category_holiday_sales"]


@pytest.fixture
def split_categories(item_rows):
    """Three categories at 96 / 3 / 1 percent of sales, on one day.

    Chosen so exactly one category sits above the 1% bar and one sits exactly
    ON it -- the boundary is where an off-by-one in the comparison shows up.
    """
    return item_rows(
        {"LINEITEM_ID": "'1'", "ITEM_CATEGORY": "'Big'",
         "ITEM_PRICE": "'96.00'"},
        {"LINEITEM_ID": "'2'", "ITEM_CATEGORY": "'Small'",
         "ITEM_PRICE": "'3.00'"},
        {"LINEITEM_ID": "'3'", "ITEM_CATEGORY": "'Boundary'",
         "ITEM_PRICE": "'1.00'"},
    )


@pytest.mark.parametrize("table", MATERIALITY_TABLES)
def test_materiality_share_is_all_time_not_per_period(spark, cfg, batch_date,
                                                      build_inputs,
                                                      split_categories, table):
    """One share per category, repeated on every row that category has.

    This is the property that keeps a category from flickering in and out of
    the tail as a reader changes the date range, so it is asserted directly
    rather than inferred from a single row.
    """
    out = TRANSFORMS[table](build_inputs(table, split_categories), batch_date)

    shares = {r["item_category"]: float(r["pct_of_total_sales"])
              for r in out.select("item_category", "pct_of_total_sales")
                          .distinct().collect()}

    assert shares == {"Big": 96.0, "Small": 3.0, "Boundary": 1.0}


@pytest.mark.parametrize("table", MATERIALITY_TABLES)
def test_is_material_includes_the_threshold_itself(spark, cfg, batch_date,
                                                   build_inputs,
                                                   split_categories, table):
    """The comparison is >=, so a category exactly at 1% is material."""
    out = TRANSFORMS[table](build_inputs(table, split_categories), batch_date)

    flags = {r["item_category"]: r["is_material"]
             for r in out.select("item_category", "is_material")
                         .distinct().collect()}

    assert flags == {"Big": True, "Small": True, "Boundary": True}


@pytest.mark.parametrize("table", MATERIALITY_TABLES)
def test_is_material_is_false_below_the_threshold(spark, cfg, batch_date,
                                                  build_inputs, item_rows,
                                                  table):
    """0.5% of sales is immaterial, and reads False rather than null."""
    df = item_rows(
        {"LINEITEM_ID": "'1'", "ITEM_CATEGORY": "'Big'",
         "ITEM_PRICE": "'199.00'"},
        {"LINEITEM_ID": "'2'", "ITEM_CATEGORY": "'Tiny'",
         "ITEM_PRICE": "'1.00'"},
    )
    out = TRANSFORMS[table](build_inputs(table, df), batch_date)

    flags = {r["item_category"]: r["is_material"]
             for r in out.select("item_category", "is_material")
                         .distinct().collect()}

    assert flags == {"Big": True, "Tiny": False}


@pytest.mark.parametrize("table", MATERIALITY_TABLES)
def test_non_categories_are_excluded(spark, cfg, batch_date, build_inputs,
                                     item_rows, table):
    """`Test Items` and `Most Popular` are not menu categories.

    Excluded from the rows AND from the denominator -- if they still counted
    toward total sales, every real category's share would be understated.
    """
    df = item_rows(
        {"LINEITEM_ID": "'1'", "ITEM_CATEGORY": "'Real'",
         "ITEM_PRICE": "'50.00'"},
        {"LINEITEM_ID": "'2'", "ITEM_CATEGORY": "'Test Items'",
         "ITEM_PRICE": "'25.00'"},
        {"LINEITEM_ID": "'3'", "ITEM_CATEGORY": "'Most Popular'",
         "ITEM_PRICE": "'25.00'"},
    )
    out = TRANSFORMS[table](build_inputs(table, df), batch_date)
    rows = out.select("item_category", "pct_of_total_sales").distinct().collect()

    assert [r["item_category"] for r in rows] == ["Real"]
    assert float(rows[0]["pct_of_total_sales"]) == 100.0


# --- the published contract -----------------------------------------------

@pytest.mark.parametrize("table", GOLD_TABLES)
def test_output_columns_are_declared(cfg, table):
    """Every gold table carries a contract. Without this the tests below pass
    vacuously for a table whose config nobody filled in."""
    assert cfg.get_gold_config(table).output_columns, \
        f"{table} declares no output_columns"


@pytest.mark.parametrize("table", GOLD_TABLES)
def test_output_matches_the_declared_contract(spark, cfg, batch_date, item_rows,
                                              build_inputs, table):
    """Declared columns are present and of the declared type.
    """
    contract = cfg.get_gold_config(table).output_columns
    out = TRANSFORMS[table](build_inputs(table, item_rows({})), batch_date)
    actual = dict(out.dtypes)

    missing = sorted(set(contract) - set(actual))
    assert not missing, f"{table} is missing contract columns {missing}"

    wrong = {name: (declared, actual[name])
             for name, declared in contract.items()
             if actual[name] != declared}
    assert not wrong, f"{table} contract type mismatch (declared, actual): {wrong}"
