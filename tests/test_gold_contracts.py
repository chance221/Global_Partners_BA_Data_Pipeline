"""Contract tests that hold for every gold transform, not just one.

test_silver_to_gold.py tests the dependency guard, using one table as its
vehicle. These tests are the other axis: the same assertions applied to every
registered transform, so adding a gold table adds coverage automatically rather
than leaving it untested until something breaks in Glue.

Three things are checked, in rising order of what they cost to get wrong.

Building at all. Every transform is called with a one-row batch and with
degenerate ones -- an empty silver table, and a batch where every user_id is
null or blank. This is not hypothetical: customer_rfm collects MAX(order_date)
to the driver to anchor its window, and on an empty frame that is None, which
raised TypeError before the guard in that transform.

The declared grain. `grain_keys` is config's claim about what makes a row
unique, and the write strategies depend on it: `merge` requires it (Iceberg
raises when two source rows match one target row) and `replace` will happily
persist duplicates. Nothing else in the suite checks a transform honours it.

The published contract. `output_columns` names the columns consumers read, with
types. A gold table is an interface -- renaming a column consumers depend on is
a breaking change, and this is what makes it cost a deliberate config edit. The
contract is a subset by design, so an additive change needs no edit here.

All four gold tables declare `depends_on: [order_items]`, so one fixture drives
every case. If a future transform reads something else, `build_inputs` raises
rather than silently passing it an empty frame.
"""

import pytest

from python_scripts.silver_to_gold.transforms import TRANSFORMS

pytestmark = pytest.mark.spark

GOLD_TABLES = sorted(TRANSFORMS)


# --- inputs ---------------------------------------------------------------

def build_inputs(cfg, table, df):
    """Map every silver table the transform declares to `df`.

    Raises on a dependency this module cannot supply, so a new transform that
    reads a second silver table fails loudly here instead of being handed an
    empty frame and quietly producing nothing.
    """
    depends_on = cfg.get_gold_config(table).depends_on
    unsupported = set(depends_on) - {"order_items"}
    if unsupported:
        raise AssertionError(
            f"{table} depends on {sorted(unsupported)}, which these tests have "
            f"no fixture for; add one rather than passing an empty frame"
        )
    return {name: df for name in depends_on}


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
def test_transform_builds_from_one_row(spark, cfg, batch_date, item_rows, table):
    """The baseline: a transform returns a frame and the frame is readable.

    `.count()` rather than a bare call, because a transform is lazy -- an error
    in the plan does not surface until something forces execution.
    """
    out = TRANSFORMS[table](build_inputs(cfg, table, item_rows({})), batch_date)

    assert out.count() >= 0
    assert out.columns


@pytest.mark.parametrize("table", GOLD_TABLES)
def test_transform_survives_degenerate_input(cfg, batch_date, degenerate, table):
    """Nothing to aggregate is an empty answer, never an exception.

    An empty gold table is correct here and a failed run is not: the batch is
    well-formed, it just has no identifiable customers or no rows at all.
    """
    for label, df in degenerate.items():
        out = TRANSFORMS[table](build_inputs(cfg, table, df), batch_date)
        assert out.count() >= 0, f"{table} produced no frame for {label}"


@pytest.mark.parametrize("table", GOLD_TABLES)
def test_degenerate_input_keeps_the_full_schema(cfg, batch_date, degenerate,
                                                item_rows, table):
    """An empty result must still carry every column.

    This is what lets the write create a correct empty table. A transform that
    returned a narrower frame when it had no rows would create a table with the
    wrong schema on a first run against empty silver.
    """
    full = TRANSFORMS[table](build_inputs(cfg, table, item_rows({})), batch_date)

    for label, df in degenerate.items():
        out = TRANSFORMS[table](build_inputs(cfg, table, df), batch_date)
        assert out.columns == full.columns, f"{table} changed schema on {label}"


# --- the declared grain ---------------------------------------------------

@pytest.mark.parametrize("table", GOLD_TABLES)
def test_grain_keys_exist_in_the_output(spark, cfg, batch_date, item_rows, table):
    """config names the grain; the transform has to actually emit it."""
    gold_config = cfg.get_gold_config(table)
    out = TRANSFORMS[table](build_inputs(cfg, table, item_rows({})), batch_date)

    missing = [k for k in gold_config.grain_keys if k not in out.columns]
    assert not missing, f"{table} does not emit grain_keys {missing}"


@pytest.mark.parametrize("table", GOLD_TABLES)
def test_grain_is_unique(spark, cfg, batch_date, item_rows, table):
    """One row per grain value, over a batch built to collide.

    Several orders for one customer on one day at one restaurant: whatever the
    grain is, these rows share it, so a transform that failed to aggregate
    would return more rows than distinct keys.
    """
    rows = item_rows(
        {"ORDER_ID": "'o1'", "LINEITEM_ID": "'1'"},
        {"ORDER_ID": "'o1'", "LINEITEM_ID": "'2'"},
        {"ORDER_ID": "'o2'", "LINEITEM_ID": "'1'"},
    )
    grain = cfg.get_gold_config(table).grain_keys
    out = TRANSFORMS[table](build_inputs(cfg, table, rows), batch_date).cache()

    try:
        assert out.count() == out.select(*grain).distinct().count(), \
            f"{table} emits duplicate {grain}"
    finally:
        out.unpersist()


@pytest.mark.parametrize("table", GOLD_TABLES)
def test_grain_is_never_null(spark, cfg, batch_date, item_rows, table):
    """A null grain key breaks MERGE matching and every downstream join."""
    grain = cfg.get_gold_config(table).grain_keys
    out = TRANSFORMS[table](build_inputs(cfg, table, item_rows({})), batch_date)

    predicate = " OR ".join(f"{k} IS NULL" for k in grain)
    assert out.where(predicate).count() == 0, f"{table} emits a null {grain}"


# --- the published contract -----------------------------------------------

@pytest.mark.parametrize("table", GOLD_TABLES)
def test_output_columns_are_declared(cfg, table):
    """Every gold table carries a contract. Without this the tests below pass
    vacuously for a table whose config nobody filled in."""
    assert cfg.get_gold_config(table).output_columns, \
        f"{table} declares no output_columns"


@pytest.mark.parametrize("table", GOLD_TABLES)
def test_output_matches_the_declared_contract(spark, cfg, batch_date, item_rows,
                                              table):
    """Declared columns are present and of the declared type.

    A subset check, deliberately: adding a column is not a breaking change and
    needs no config edit, while renaming or retyping a declared one fails here.
    """
    contract = cfg.get_gold_config(table).output_columns
    out = TRANSFORMS[table](build_inputs(cfg, table, item_rows({})), batch_date)
    actual = dict(out.dtypes)

    missing = sorted(set(contract) - set(actual))
    assert not missing, f"{table} is missing contract columns {missing}"

    wrong = {name: (declared, actual[name])
             for name, declared in contract.items()
             if actual[name] != declared}
    assert not wrong, f"{table} contract type mismatch (declared, actual): {wrong}"
