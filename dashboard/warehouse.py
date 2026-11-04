"""Live reads of the pipeline's own Iceberg tables, with no Spark and no CSVs.

PyIceberg against the Glue catalog, which is the arrangement `config/*.yaml`
describes: the jobs write `iceberg.gold.<table>` through Spark and this reads
`gold.<table>` through pure Python, so the dashboard never carries a second
copy of the pipeline's output. No JVM, no Ivy resolve, and it runs on a small
instance.

WHY THIS IS CHEAP ENOUGH TO DO LIVE
-----------------------------------
Measured against the real warehouse:

  catalog connect                            0.50s
  gold.customer_rfm, 20,174 x 25             0.89s
  silver.order_items, 203,518 rows / 7.1 MB  1.57s  (41 MB in pandas)
  the heaviest groupby below                 0.01s

The whole fact table is seven megabytes. Spark was the expensive part, not the
data, and `build_data.py` pre-materialising CSVs was solving a problem that
only existed because it went through Spark to reach the same rows.

ONE SNAPSHOT, NOT ELEVEN
------------------------
`datasets()` reads each source table ONCE and derives every frame from it.
Caching eleven separate queries would let a pipeline run land between two of
them, giving a page whose segments and whose orders disagree. One read means
every number on the page comes from one state of the warehouse.

TWO THINGS THAT MUST MATCH SPARK, AND SILENTLY WOULD NOT
--------------------------------------------------------
1. `creation_time_utc` is resolved to a date in UTC, explicitly. The pipeline
   pins `spark.sql.session.timeZone` to UTC for exactly this reason: the last
   order in silver is 2024-02-21 02:52 UTC, which is 2024-02-20 in New York,
   and taking the machine's zone here would move the anchor a day and change
   every window on the page. `.dt.tz_convert(None)` or a naive parse is the
   bug; `utc=True` is the fix.

2. A percentage whose numerator can be NULL takes its denominator from the
   rows where it is NOT null. Spark's `avg()` skips nulls, pandas counts
   `NaN >= 2` as False. On `pct_2x_overdue` for the New segment -- 362 of 443
   customers have no second order, so no cadence -- the two readings are 37.0%
   and 6.8%. The Spark reading is the intended one, which is why `no_cadence`
   is published beside it, and `_pct_where` below reproduces it.

Medians use `interpolation="lower"` because Spark's `percentile_approx` returns
an actual data point while pandas interpolates between the two middle values.
Without it every even-sized group differs by half a unit from the published
gold.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
from pyiceberg.catalog.glue import GlueCatalog
from pyiceberg.exceptions import NoSuchTableError

from python_scripts.common.config_loader import load_config

PERIODS = ["P1 last 30d", "P2 31-60d", "P3 61-90d"]

CLIFF_BUCKETS = {"same day": 0, "1-3 days": 1, "4-7 days": 2, "8-14 days": 3,
                 "15-30 days": 4, "31+ days": 5, "never": 6}

# The gold table labels these with the anchor's nature; the charts want the
# plainer words. Mapped here rather than in the chart code so the gold table
# stays the one place the eras are named.
ERA_LABELS = {"Pre-launch": "Pre-launch", "Recent": "Recent",
              "Prior": "Prior"}
COHORT_LABELS = {"Member at T0": "member", "Non-member at T0": "non-member"}
ALL_CUSTOMERS = "All customers"

# Reading order for the lifetime-spend tiers, highest spend first. The gold
# table stores the tier as a string with no natural sort, so without this the
# charts come out alphabetical: Bottom, Mid, Top. Must match
# `clv_history.TIERS`, which is what the pipeline writes.
CLV_TIERS = ["Top", "Mid", "Bottom"]

# Must match clv_history.OUTLIER_VARIANTS. Not imported from it: the dashboard
# runs without pyspark installed -- that is the whole point of reading gold
# through PyIceberg -- and importing a transform module would pull the JVM
# bindings in. The contract is the published column, and `gold_clv` fails loud
# below if the table does not carry the variant it was asked for.
OUTLIERS_INCLUDED = "included"
OUTLIERS_EXCLUDED = "excluded"

_ORDER_COLS = ("user_id", "order_id", "creation_time_utc", "item_price",
               "item_quantity", "is_loyalty")

# Hour buckets in clock order. Must match sales_trend_grid.DAYPARTS, and for
# the same reason as CLV_TIERS above: the published column is a string with no
# natural sort, so without this the charts come out alphabetical -- Afternoon,
# Evening, Midday, Morning, Overnight -- which reads as nonsense on a time
# axis. Not imported from the transform, which would pull in pyspark.
DAYPART_ORDER = ["Overnight", "Morning", "Midday", "Afternoon", "Evening"]

# Measure -> the column holding its volume-outlier portion. The names are not
# uniform (`total_revenue` pairs with `outlier_revenue`), so the pairing is
# written out rather than derived with a prefix.
SALES_MEASURES = (
    ("orders", "outlier_orders"),
    ("line_items", "outlier_line_items"),
    ("units", "outlier_units"),
    ("total_revenue", "outlier_revenue"),
)

# The same pairing for `loyalty_join_profile`, which carries two more. Note
# `customers` is in here: netting the volume accounts out of a per-customer
# rate means removing them from the DENOMINATOR as well, and leaving them in
# it would understate every netted rate on the page.
JOIN_MEASURES = (
    ("customers", "outlier_customers"),
    ("orders", "outlier_orders"),
    ("line_items", "outlier_line_items"),
    ("units", "outlier_units"),
    ("active_days", "outlier_active_days"),
    ("total_revenue", "outlier_revenue"),
)

# The trailing window `member_frequency` compares in, and the recency rule
# that makes the comparison fair. 90 matches customer_rfm's
# ACTIVE_WINDOW_DAYS so the figures sit beside the published 90-day columns
# rather than inventing a second convention.
MEMBER_WINDOW_DAYS = 90
# A customer only enters the comparison if they ordered within this many days
# of the window opening. Without it the non-member side is mostly people who
# last ordered in 2020 and could never have enrolled -- see the docstring.
MEMBER_RECENCY_DAYS = 365
# Prior-window order counts, as (label, lower, upper) with both inclusive.
# Holding this fixed is what separates "members order more" from "engaged
# customers order more, and they are the ones who enrol".
MEMBER_PRIOR_BANDS = (
    ("0", 0, 0),
    ("1", 1, 1),
    ("2-3", 2, 3),
    ("4-9", 4, 9),
    ("10+", 10, 10 ** 9),
)

# The half-widths `loyalty_join_profile` publishes, widest first so the
# default selection shows the whole decay rather than the first third of it.
JOIN_COVERAGE = (180, 90)


def catalog(env: str = "prod"):
    """The Glue catalog, configured from the same YAML the jobs read."""
    cfg = load_config(env)
    if cfg.catalog_type != "glue":
        raise RuntimeError(
            f"the dashboard needs catalog_type 'glue', config says "
            f"{cfg.catalog_type!r}. PyIceberg has no hadoop catalog "
            f"implementation, so it cannot see a directory warehouse."
        )
    cat = GlueCatalog("glue", region_name=cfg.region,
                      warehouse=cfg.warehouse)
    return cat, cfg


def scan(cat, cfg, layer: str, table: str,
         cols: tuple[str, ...] | None = None) -> pd.DataFrame:
    """One Iceberg table into pandas, projected to the columns needed."""
    tbl = cat.load_table(f"{cfg.layer(layer)}.{table}")
    s = tbl.scan(selected_fields=cols) if cols else tbl.scan()
    return s.to_pandas()


def _floats(frame: pd.DataFrame, *cols: str) -> pd.DataFrame:
    """Decimal columns to float. Vega cannot serialise decimal.Decimal, and it
    raises TypeError the moment it meets a float."""
    for c in cols:
        if c in frame.columns:
            frame[c] = pd.to_numeric(frame[c], errors="coerce").astype("float64")
    return frame


def _median(series: pd.Series) -> float:
    """Spark's percentile_approx, near enough: a real data point, not an
    interpolation between two of them."""
    s = series.dropna()
    return float("nan") if s.empty else float(
        s.quantile(0.5, interpolation="lower"))


def _pct_where(flag: pd.Series, defined: pd.Series) -> float:
    """A percentage over the rows where the measure EXISTS.

    `defined` is the not-null mask. This is what Spark's avg() does and what
    a naive pandas mean() does not -- see the module docstring.
    """
    n = int(defined.sum())
    return float("nan") if n == 0 else round(100.0 * float(
        (flag & defined).sum()) / n, 1)


# --- source frames --------------------------------------------------------

def per_order(cat, cfg) -> pd.DataFrame:
    """One row per (user_id, order_id), mirroring the transforms' own grain.

    Anonymous orders are dropped, as every transform does: a per-customer
    measure cannot be built without a customer.
    """
    li = scan(cat, cfg, "silver", "order_items", _ORDER_COLS)
    li = li[li.user_id.notna() & (li.user_id.astype("string").str.strip() != "")]

    # UTC explicitly. See the module docstring -- this is the pinned-timezone
    # bug, and it would move the anchor by a day.
    li = li.assign(
        order_date=pd.to_datetime(li.creation_time_utc, utc=True).dt.date,
        item_price=pd.to_numeric(li.item_price).astype("float64"),
        is_loyalty=li.is_loyalty.fillna(False).astype(bool))

    return (li.groupby(["user_id", "order_id"], as_index=False)
            .agg(order_date=("order_date", "min"),
                 order_revenue=("item_price", "sum"),
                 order_units=("item_quantity", "sum"),
                 loyalty_order=("is_loyalty", "max")))


def customers(cat, cfg) -> pd.DataFrame:
    """gold.customer_rfm. Carries 25 columns although it declares 7 -- the
    transform drops only its two intermediates -- so recency and cadence are
    available here without recomputing anything."""
    df = scan(cat, cfg, "gold", "customer_rfm", (
        "user_id", "segment", "segment_group", "is_active", "recency_days",
        "avg_gap_days", "total_orders", "total_revenue", "orders_last90d",
        "spend_last90d", "vip_spend_cut", "anchor_date"))
    return _floats(df, "avg_gap_days", "total_revenue", "spend_last90d",
                   "vip_spend_cut")


def volume_outliers(cat, cfg) -> pd.DataFrame:
    """The accounts `gold.customer_lifetime_value` has flagged as non-consumer.

    Read from the published flag rather than re-derived from an order count
    here, so the dashboard and the three gold tables cannot disagree about
    which accounts are set aside. `clv_history.OUTLIER_ORDER_FLOOR` is the one
    place the rule is written down; this is a consumer of its output.

    Three rows at the time of writing, ordering 1.4-2.4 times a day for over a
    year each. Returned whole rather than as a set of ids because the page
    names them and quotes their totals -- an exclusion the reader cannot see
    the size of is worse than no exclusion.
    """
    df = scan(cat, cfg, "gold", "customer_lifetime_value",
              ("user_id", "total_orders", "total_lifetime_revenue",
               "avg_order_value", "orders_per_active_day", "clv_segment",
               "first_order_date", "last_order_date", "is_volume_outlier"))
    df = df[df.is_volume_outlier.fillna(False).astype(bool)]
    df = _floats(df, "total_lifetime_revenue", "avg_order_value",
                 "orders_per_active_day")
    return df.sort_values("total_orders", ascending=False).reset_index(drop=True)


# --- derived frames -------------------------------------------------------

def _order_stats(orders: pd.DataFrame, key: str) -> pd.DataFrame:
    """Order-level measures, which gold.customer_rfm cannot supply.

    It holds one row per customer, so a median taken from it would be a median
    of per-customer averages -- a different and much less useful number than
    the median order.
    """
    total = float(orders.order_revenue.sum()) or 1.0
    out = (orders.groupby(key)
           .apply(lambda d: pd.Series({
               "orders": len(d),
               "revenue": round(float(d.order_revenue.sum()), 2),
               "avg_order": round(float(d.order_revenue.mean()), 2),
               "med_order": round(_median(d.order_revenue), 2),
               "units": int(d.order_units.sum()),
               "avg_units": round(float(d.order_units.mean()), 2),
               "med_units": _median(d.order_units),
           }), include_groups=False)
           .reset_index())
    out["rev_share_pct"] = (100.0 * out.revenue / total).round(2)
    return out


def window_frames(cust: pd.DataFrame, orders: pd.DataFrame,
                  anchor: date) -> dict[str, pd.DataFrame]:
    """The 90-day window, whole and split into its three 30-day periods."""
    p1, p2, w0 = (anchor - timedelta(days=d) for d in (30, 60, 90))

    # Dormant drops out via the inner join on an active-only frame. It has no
    # orders inside the window by definition, so it would be a row of zeros --
    # which is the whole reason is_active exists.
    active = cust[cust.is_active]
    win = (orders[orders.order_date > w0]
           .merge(active[["user_id", "segment", "segment_group"]],
                  on="user_id", how="inner"))
    win = win.assign(period=[
        PERIODS[0] if d > p1 else PERIODS[1] if d > p2 else PERIODS[2]
        for d in win.order_date])

    out: dict[str, pd.DataFrame] = {}
    for key in ("segment", "segment_group"):
        counts = (active.groupby(key).size()
                  .rename("customers").reset_index())
        whole = _order_stats(win, key).merge(counts, on=key, how="left")
        whole["orders_per_cust"] = (whole.orders / whole.customers).round(2)
        whole["rev_per_cust"] = (whole.revenue / whole.customers).round(2)
        out[f"window_{key}"] = whole

        parts = [_order_stats(win[win.period == p], key).assign(period=p)
                 for p in PERIODS]
        out[f"period_{key}"] = pd.concat(parts, ignore_index=True)
    return out


def cadence_frames(cust: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Recency against each customer's OWN gap, divided per customer.

    The ratio of two medians is not the median of the ratio, and here they
    differ by enough to matter -- Slipping reads 3.16x one way and 3.01x the
    other -- so the division happens per customer before aggregating.

    `avg_gap_days` is NULL for a one-order customer: there is no gap to be
    late against. That is why `no_cadence` sits beside the medians, and why
    the percentage uses `_pct_where`.
    """
    active = cust[cust.is_active].copy()
    active["overdue"] = (active.recency_days / active.avg_gap_days).where(
        active.avg_gap_days > 0)

    out: dict[str, pd.DataFrame] = {}
    for key in ("segment", "segment_group"):
        rows = []
        for name, d in active.groupby(key):
            has = d.overdue.notna()
            rows.append({
                key: name,
                "customers": len(d),
                "med_recency": _median(d.recency_days),
                "med_gap": round(_median(d.avg_gap_days), 1),
                "med_overdue": round(_median(d.overdue), 2),
                "pct_2x_overdue": _pct_where(d.overdue >= 2, has),
                "no_cadence": int((~has).sum()),
                "lifetime_revenue": round(float(d.total_revenue.sum()), 2),
            })
        out[f"cadence_{key}"] = pd.DataFrame(rows)
    return out


def segment_sizes(cust: pd.DataFrame) -> pd.DataFrame:
    """One row per segment, for the definitions reference at the top of the page.

    Dormant is included here and nowhere else. Every other frame drops it
    because its 90-day figures are zero by definition, but the reference exists
    to say what each segment MEANS, and omitting the one that holds four fifths
    of the base would misrepresent the shape of the book.

    Counted live rather than written into the page, so a definition and its
    headcount cannot drift apart after a pipeline run.
    """
    out = (cust.groupby(["segment", "segment_group"], as_index=False)
           .agg(customers=("user_id", "size"), is_active=("is_active", "max")))
    out["pct_of_base"] = (100.0 * out.customers / len(cust)).round(1)
    return out.sort_values("customers", ascending=False).reset_index(drop=True)


def reorder_cliff(orders: pd.DataFrame, anchor: date) -> pd.DataFrame:
    """How long a customer took to order again, and what they became.

    Restricted to customers whose FIRST order is at least 90 days before the
    anchor. Without that cut the recent joiners are right-censored -- someone
    who first ordered three days ago cannot have reordered within seven, and
    counting them inflates the never-reordered share.
    """
    srt = orders.sort_values(["user_id", "order_date"]).copy()
    # An explicit row number, mirroring the row_number() window this replaces.
    # groupby().nth(1) returns original-indexed ROWS rather than a per-group
    # value, so it cannot be merged back on user_id.
    srt["rn"] = srt.groupby("user_id").cumcount()
    second = (srt.loc[srt.rn == 1, ["user_id", "order_date"]]
              .rename(columns={"order_date": "second_date"}))
    cust = (srt.groupby("user_id", as_index=False)
            .agg(first_date=("order_date", "min"),
                 life_orders=("order_id", "size"),
                 life_revenue=("order_revenue", "sum"))
            .merge(second, on="user_id", how="left"))

    cust["days_to_second"] = [
        (s - f).days if pd.notna(s) else None
        for s, f in zip(cust.second_date, cust.first_date)]
    cust["observed_days"] = [(anchor - f).days for f in cust.first_date]
    cust = cust[cust.observed_days >= 90]

    def bucket(d):
        if d is None or pd.isna(d):
            return "never"
        for limit, name in ((0, "same day"), (3, "1-3 days"), (7, "4-7 days"),
                            (14, "8-14 days"), (30, "15-30 days")):
            if d <= limit:
                return name
        return "31+ days"

    cust["bucket"] = [bucket(d) for d in cust.days_to_second]
    total = len(cust)
    frame = (cust.groupby("bucket", as_index=False)
             .agg(customers=("user_id", "size"),
                  avg_life_orders=("life_orders", "mean"),
                  avg_life_revenue=("life_revenue", "mean"),
                  total_life_revenue=("life_revenue", "sum")))
    frame["avg_life_orders"] = frame.avg_life_orders.round(2)
    frame["avg_life_revenue"] = frame.avg_life_revenue.round(2)
    frame["total_life_revenue"] = frame.total_life_revenue.round(2)
    frame["bucket_order"] = frame.bucket.map(CLIFF_BUCKETS)
    frame["pct_of_customers"] = (100.0 * frame.customers / total).round(1)
    frame["pct_of_revenue"] = (
        100.0 * frame.total_life_revenue / frame.total_life_revenue.sum()
    ).round(1)
    return frame.sort_values("bucket_order").reset_index(drop=True)


def gold_transitions(cat, cfg) -> pd.DataFrame:
    """gold.customer_segment_transitions, straight off the catalog.

    Nothing is recomputed. Segmenting customers at a past anchor is the
    pipeline's job, in the Spark that owns the segment definitions -- porting
    it to pandas here would be a second implementation of the rules and the
    first thing to drift.
    """
    df = scan(cat, cfg, "gold", "customer_segment_transitions")
    df = _floats(df, "row_pct", "vip_spend_cut")
    df["era"] = df.era.map(lambda e: ERA_LABELS.get(e, e))
    # The old CSV called it n_at_t0 and the charts still do.
    return df.rename(columns={"customers_at_t0": "n_at_t0"})


def gold_loyalty(cat, cfg) -> dict[str, pd.DataFrame]:
    """gold.loyalty_cohort_outcomes, split into its per-segment and pooled rows.

    The table carries both: every customer is counted once under their T0
    segment and once under `All customers`. Splitting them here keeps a chart
    from accidentally summing a base that is counted twice.
    """
    df = scan(cat, cfg, "gold", "loyalty_cohort_outcomes")
    df = _floats(df, "still_active_pct", "to_core_pct", "to_dormant_pct",
                 "avg_orders_90d_t1", "avg_spend_90d_t1", "vip_spend_cut")
    df["cohort"] = df.cohort.map(lambda c: COHORT_LABELS.get(c, c))
    pooled = df[df.seg_t0 == ALL_CUSTOMERS].copy()
    pooled["seg_t0"] = "ALL (pooled)"
    return {"loyalty_by_segment": df[df.seg_t0 != ALL_CUSTOMERS].copy(),
            "loyalty_pooled": pooled}


def gold_clv(cat, cfg, variant: str = OUTLIERS_INCLUDED) -> dict[str, pd.DataFrame]:
    """The two lifetime-spend-tier tables, straight off the catalog.

    Nothing is recomputed, for the same reason as `gold_transitions`: the tier
    cut is a percentile over the whole customer base, so reproducing it here
    would be a second definition of "top fifth" and the first thing to drift
    when someone edits `clv_history`.

    `median_spend_per_order` in particular could not be rebuilt here from the
    other frames at all -- it needs every order joined to its customer's tier,
    and `customers()` holds one row per customer. That is also why the outlier
    exclusion has to be a published variant rather than a filter applied here:
    a median cannot be adjusted after the fact, only recomputed.

    `variant` picks the population. Both tables carry three rows per variant,
    so NOT filtering would double every count on the page -- which is the one
    hazard the new grain introduces, and the reason this raises instead of
    returning whatever it finds.

    Rows are ordered Top, Mid, Bottom. The gold table has no natural sort and
    the tier is a string, so leaving it unordered puts Bottom first
    alphabetically and every chart reads backwards.
    """
    def pick(frame: pd.DataFrame, table: str) -> pd.DataFrame:
        if "outliers" not in frame.columns:
            raise RuntimeError(
                f"gold.{table} has no `outliers` column, so the volume-outlier "
                f"toggle cannot be honoured. It predates that grain -- re-run "
                f"`silver_to_gold --full --tables {table}`.")
        kept = frame[frame.outliers == variant]
        if kept.empty and not frame.empty:
            raise RuntimeError(
                f"gold.{table} carries no rows for outliers={variant!r}; it "
                f"has {sorted(frame.outliers.unique())}")
        return kept.drop(columns="outliers").reset_index(drop=True)

    profile = pick(scan(cat, cfg, "gold", "clv_tier_profile"),
                   "clv_tier_profile")
    profile = _floats(profile, "pct_of_customers", "total_spend",
                      "pct_of_spend", "pct_of_orders", "median_total_spend",
                      "avg_total_spend", "spend_floor", "spend_ceiling",
                      "avg_orders", "avg_spend_per_order",
                      "median_spend_per_order", "one_order_pct",
                      "avg_units_per_order")
    segs = pick(scan(cat, cfg, "gold", "clv_tier_segments"),
                "clv_tier_segments")
    segs = _floats(segs, "pct_of_tier", "pct_of_segment", "total_spend",
                   "pct_of_tier_spend", "median_total_spend", "avg_orders",
                   "avg_customer_order_value", "vip_spend_cut")

    rank = {t: i for i, t in enumerate(CLV_TIERS)}
    profile = (profile.assign(_o=profile.clv_tier.map(rank))
               .sort_values("_o").drop(columns="_o").reset_index(drop=True))
    segs = (segs.assign(_o=segs.clv_tier.map(rank))
            .sort_values(["_o", "customers"], ascending=[True, False])
            .drop(columns="_o").reset_index(drop=True))
    return {"clv_profile": profile, "clv_segments": segs}


# --- the one entry point --------------------------------------------------

def sales_trend(cat, cfg, exclude_outliers: bool = False) -> pd.DataFrame:
    """The hourly sales grid, with the volume accounts netted out on request.

    Returned WHOLE rather than pre-aggregated. The tab rolls it up by day, by
    week and by month, and by store, category, weekday and hour, and 126,433
    rows is a third of the silver read this page already does. One frame per
    combination would be a dozen frames that can disagree about a total.

    The exclusion here is a SUBTRACTION, not a filter, and that is the whole
    reason this tab can honour the toggle when the era tables cannot. Every
    measure in `sales_trend_grid` is additive and each cell publishes its own
    flagged-account portion, so `measure - outlier_measure` is the excluded
    reading at ANY rollup. There is no customer grain here to filter on -- the
    grid has no user_id -- so without those columns the only options would be
    a second table or an honest refusal.

    The `outlier_*` columns are left on the frame after the subtraction, so
    the page can say what it removed rather than only showing what remains.

    `week_complete` and `month_complete` are added here because the trap they
    cover is not optional: silver ends 2024-02-20, so the last month holds 20
    days and the last week holds 2. Plotted unmarked, a complete-looking
    series falls off a cliff at the right-hand edge and the drop is the
    calendar, not the business. The first period has the same problem at the
    other end -- April 2020 is 2 days -- so completeness is computed against
    both ends of the data's own span.
    """
    df = scan(cat, cfg, "gold", "sales_trend_grid")
    df = _floats(df, "total_revenue", "outlier_revenue")

    if exclude_outliers:
        for measure, portion in SALES_MEASURES:
            df[measure] = df[measure] - df[portion]

    span_start, span_end = df.sales_date_local.min(), df.sales_date_local.max()
    week = pd.to_datetime(df.week_start)
    month = pd.to_datetime(df.month_start)
    # Spark's `trunc(date, 'week')` starts the week on Monday, so the period
    # ends six days later; MonthEnd(0) on a month-start date returns that
    # month's own last day rather than the next one's.
    df["week_complete"] = ((week.dt.date >= span_start)
                           & ((week + pd.Timedelta(days=6)).dt.date
                              <= span_end))
    df["month_complete"] = ((month.dt.date >= span_start)
                            & ((month + pd.offsets.MonthEnd(0)).dt.date
                               <= span_end))
    return df


def join_profile(cat, cfg, exclude_outliers: bool = False) -> pd.DataFrame:
    """The order rate in 30-day periods around each customer's own join date.

    Returned whole, both `coverage_days` populations together, because the tab
    lets the reader pick one and they are 20 rows in total.

    THE PUBLISHED RATES ARE THE UNNETTED ONES, SO THEY ARE RECOMPUTED HERE
    ---------------------------------------------------------------------
    `orders_per_customer` and friends are computed in the transform from the
    whole cohort. Subtracting the `outlier_*` portion changes the numerator
    AND the denominator, so the published rate is no longer the rate of what
    is left. Every rate this function returns is therefore derived after the
    subtraction, and the published ones are dropped rather than left on the
    frame to be picked up by accident.

    `median_order_value` is NOT netted and cannot be: a published median
    cannot be un-aggregated. It is dropped when excluding rather than shown
    as if it had been corrected.

    WHY THIS TABLE HONOURS THE TOGGLE WHEN THE OTHER TWO ERA TABLES DO NOT
    ---------------------------------------------------------------------
    It publishes the flagged accounts' portion of every additive measure, the
    way `sales_trend_grid` does, so the exclusion is a subtraction rather than
    a filter. It matters more here than anywhere else on the page: two
    accounts are 826 of 2,778 orders in period -2 alone, which unnetted is
    the single most prominent feature of the curve.
    """
    df = scan(cat, cfg, "gold", "loyalty_join_profile")
    counts = [c for pair in JOIN_MEASURES for c in pair]
    for column in counts:
        df[column] = pd.to_numeric(df[column])

    if exclude_outliers:
        for measure, portion in JOIN_MEASURES:
            df[measure] = df[measure] - df[portion]
        df = df.drop(columns=["median_order_value"])
    else:
        df = _floats(df, "median_order_value")

    base = df.customers.clip(lower=1)
    df["orders_per_customer"] = df.orders / base
    df["visits_per_customer"] = df.active_days / base
    # Null rather than a division by zero: a period with no visits has no
    # orders-per-visit, which is not the same as one order per visit.
    df["orders_per_visit"] = (df.orders / df.active_days).where(
        df.active_days > 0)
    df["avg_order_value"] = (df.total_revenue / df.orders).where(df.orders > 0)
    df["revenue_per_customer"] = df.total_revenue / base
    return df.drop(columns=["orders_per_customer_per_day",
                            "active_days_per_customer",
                            "orders_per_active_day"])


def member_frequency(orders: pd.DataFrame, anchor: date) -> pd.DataFrame:
    """Members against non-members in the most recent 90 days, like for like.

    One row per (prior-order band, cohort). The question is whether members
    are still ordering and visiting more often NOW, rather than over the
    180-day span the anchored cohort tables measure.

    FOUR THINGS DECIDE THIS NUMBER, AND THREE OF THEM INFLATE IT IF IGNORED
    ----------------------------------------------------------------------
    1. Membership is fixed AS OF THE WINDOW START -- a customer counts as a
       member if their first flagged order is on or before it. Defining it
       from a flagged order inside the window would be circular: being
       active in the window would be part of being a member.
    2. The base is customers who already EXISTED before the window. Someone
       whose first ever order falls inside it could not have been a member
       beforehand, so including them would mix acquisition into a frequency
       comparison.
    3. Customers with ZERO orders in the window stay in the denominator.
       Dropping them silently turns every measure into "per ACTIVE customer",
       which flatters whichever cohort is more likely to be active -- and
       that is most of what is being measured.
    4. THE RECENCY RULE IS THE ONE THAT MATTERS MOST. Enrolling requires an
       order, so to be a member by the window start a customer had to
       transact between the program launch and then. Non-members carry no
       such requirement, so unrestricted they include the whole dormant pile:
       median recency runs 152 days for members against 477 for non-members,
       and the raw gap in the zero-order band comes out at +1066% on that
       difference alone. Restricting both cohorts to customers who ordered
       within `MEMBER_RECENCY_DAYS` of the window start cuts that band to
       +310% and is the only version of this comparison worth publishing.

    Even so this is NOT causal. Holding prior orders and recency fixed does
    nothing about self-selection on who chooses to enrol, so read it as an
    upper bound on the association rather than an effect.
    """
    window_start = anchor - timedelta(days=MEMBER_WINDOW_DAYS)
    prior_start = window_start - timedelta(days=MEMBER_WINDOW_DAYS)

    per_customer = orders.groupby("user_id").agg(
        first_order=("order_date", "min"),
        last_order=("order_date", "max"))
    joined = (orders[orders.loyalty_order]
              .groupby("user_id").order_date.min().rename("join_date"))
    per_customer = per_customer.join(joined)

    in_window = orders[(orders.order_date > window_start)
                       & (orders.order_date <= anchor)]
    in_prior = orders[(orders.order_date > prior_start)
                      & (orders.order_date <= window_start)]

    panel = (per_customer
             .join(in_window.groupby("user_id").agg(
                 orders=("order_id", "count"),
                 visits=("order_date", "nunique")))
             .join(in_prior.groupby("user_id")
                   .agg(prior_orders=("order_id", "count"))))
    panel[["orders", "visits", "prior_orders"]] = panel[
        ["orders", "visits", "prior_orders"]].fillna(0)

    # Existed beforehand, and was still alive when the window opened.
    recency = [(window_start - d).days for d in panel.last_order]
    panel = panel[(panel.first_order <= window_start)
                  & (pd.Series(recency, index=panel.index)
                     <= MEMBER_RECENCY_DAYS)]

    # Fixed at the window start. A customer who enrolled INSIDE the window is
    # neither cohort -- they are active by construction, which would answer
    # the question with its own premise.
    def cohort_of(join_date) -> str:
        if pd.isna(join_date):
            return "non-member"
        return "member" if join_date <= window_start else "joined in window"

    panel["cohort"] = [cohort_of(j) for j in panel.join_date]
    panel = panel[panel.cohort != "joined in window"]
    panel["active"] = panel.orders > 0

    rows = []
    for position, (label, low, high) in enumerate(MEMBER_PRIOR_BANDS):
        band = panel[panel.prior_orders.between(low, high)]
        for cohort in ("member", "non-member"):
            sub = band[band.cohort == cohort]
            customers = int(len(sub))
            n_orders = float(sub.orders.sum())
            n_visits = float(sub.visits.sum())
            n_active = int(sub.active.sum())
            rows.append({
                "prior_band": label,
                "band_order": position,
                "cohort": cohort,
                "customers": customers,
                "orders": n_orders,
                "visits": n_visits,
                "active": n_active,
                "pct_active": (100.0 * n_active / customers
                               if customers else float("nan")),
                "orders_per_customer": (n_orders / customers
                                        if customers else float("nan")),
                "visits_per_customer": (n_visits / customers
                                        if customers else float("nan")),
                "orders_per_visit": (n_orders / n_visits
                                     if n_visits else float("nan")),
                # Among the active only, so "more of them come back" reads
                # apart from "they order more once they are back".
                "orders_per_active": (n_orders / n_active
                                      if n_active else float("nan")),
                "window_start": str(window_start),
                "window_end": str(anchor),
                "recency_limit_days": MEMBER_RECENCY_DAYS,
            })
    return pd.DataFrame(rows)


def datasets(env: str = "prod",
             exclude_outliers: bool = False) -> dict[str, pd.DataFrame]:
    """Every frame the dashboard draws, from one read of the warehouse.

    `exclude_outliers` drops the three volume accounts from every aggregate on
    the page. It is applied HERE, to `cust` and `orders`, rather than in each
    chart: every derived frame below is computed from those two, so one filter
    covers the whole page and a chart cannot be added later that silently
    ignores the toggle. The two pre-aggregated CLV tables cannot be filtered
    this way -- a published median is not adjustable -- so they carry the
    variant as grain and `gold_clv` picks it.

    The era tables (`transitions`, `loyalty_*`) are pre-aggregated and have no
    such variant, so they do NOT honour the toggle. That is reported in `meta`
    and said on the page rather than left for the reader to assume either way.
    """
    cat, cfg = catalog(env)

    orders = per_order(cat, cfg)
    cust = customers(cat, cfg)

    # Reported, never raised, like the two blocks at the end of this function.
    # The flag is a recent addition to `customer_lifetime_value`, so a
    # warehouse built before it should still draw the whole page with the
    # toggle inert rather than fail on the first read.
    try:
        outliers = volume_outliers(cat, cfg)
        # Their CURRENT 90-day segment, merged from the unfiltered `cust`
        # before the exclusion below removes them from it. This is the part
        # that makes the flag legible: two of the three are Slipping and one
        # is long Dormant, which is why excluding them barely touches the
        # 90-day figures but moves the lifetime ones.
        outliers = outliers.merge(cust[["user_id", "segment"]],
                                  on="user_id", how="left")
        outliers["segment"] = outliers.segment.fillna("n/a")
        outlier_state = "present"
    except (NoSuchTableError, ValueError, KeyError) as exc:
        outliers = pd.DataFrame(columns=[
            "user_id", "total_orders", "total_lifetime_revenue",
            "avg_order_value", "orders_per_active_day", "clv_segment",
            "first_order_date", "last_order_date", "is_volume_outlier",
            "segment"])
        outlier_state = f"unavailable: {exc}"

    # The anchor the page reports is silver's own last order date, never
    # batch_date -- which is pinned to 2026-10-16 in the job arguments and
    # would claim data from the future.
    #
    # Taken BEFORE the exclusion below, deliberately. If one of the flagged
    # accounts happened to hold the last order in silver, filtering first
    # would move the anchor and shift every 90-day window on the page -- so
    # the toggle would change the segmentation itself rather than just which
    # customers are counted. The anchor is a property of the data, not of the
    # reader's filter.
    anchor = max(orders.order_date)

    excluded_ids = set(outliers.user_id) if exclude_outliers else set()
    if excluded_ids:
        orders = orders[~orders.user_id.isin(excluded_ids)]
        cust = cust[~cust.user_id.isin(excluded_ids)]

    # gold was built at ITS anchor. If silver has moved on since the last Glue
    # run the two disagree, and every segment on the page is stale by that
    # many days. Surfaced rather than silently reconciled.
    gold_anchor = pd.to_datetime(cust.anchor_date.iloc[0]).date()

    frames: dict[str, pd.DataFrame] = {}
    frames.update(window_frames(cust, orders, anchor))
    frames.update(cadence_frames(cust))
    frames["reorder_cliff"] = reorder_cliff(orders, anchor)
    frames["segment_sizes"] = segment_sizes(cust)
    frames["member_frequency"] = member_frequency(orders, anchor)

    active = cust[cust.is_active]
    frames["meta"] = pd.DataFrame([{
        "anchor_date": str(anchor),
        "gold_anchor_date": str(gold_anchor),
        "stale_days": (anchor - gold_anchor).days,
        # The FLOATING cut, recomputed by customer_rfm on every run. Taken from
        # an ACTIVE row on purpose: the column is a percentile over
        # Window.partitionBy("is_active"), so the dormant partition carries its
        # own cut of 0.00 and reading row zero of the whole frame returns that.
        # The frozen cut the era comparisons use is a different number and is
        # filled in below from the transitions table, which publishes it.
        "live_vip_cut": float(active.vip_spend_cut.dropna().iloc[0]),
        "frozen_vip_cut": float("nan"),
        "active_customers": int(len(active)),
        "all_customers": int(len(cust)),
        "spend_90d": round(float(active.spend_last90d.sum()), 2),
        "orders_90d": int(active.orders_last90d.sum()),
        "lifetime_revenue": round(float(cust.total_revenue.sum()), 2),
        # The toggle's state and what it is worth, so the page can say what it
        # is leaving out instead of quietly leaving it out. These three totals
        # are of the FLAGGED accounts themselves and do not change with the
        # toggle -- they are what was set aside, not what remains.
        "outliers_excluded": bool(exclude_outliers) and bool(len(outliers)),
        "outlier_flag": outlier_state,
        "outlier_accounts": int(len(outliers)),
        "outlier_orders": int(outliers.total_orders.sum()),
        "outlier_revenue": round(float(outliers.total_lifetime_revenue.sum()), 2),
    }])

    # Kept whole so the page can name the accounts and quote their rates.
    frames["volume_outliers"] = outliers

    # The two anchored tables are published by Glue, so a repo cloned before
    # that job has run will not find them. Missing is reported, never raised:
    # three of the five tabs do not depend on them and should still draw.
    try:
        frames["transitions"] = gold_transitions(cat, cfg)
        frames.update(gold_loyalty(cat, cfg))
        frames["meta"]["anchored_tables"] = "present"
        frames["meta"]["frozen_vip_cut"] = float(
            frames["transitions"].vip_spend_cut.iloc[0])
    except NoSuchTableError as exc:
        frames["transitions"] = pd.DataFrame(columns=[
            "era", "seg_t0", "grp_t1", "t0", "t1", "transition_days",
            "customers", "n_at_t0", "row_pct", "vip_spend_cut"])
        empty = pd.DataFrame(columns=[
            "cohort", "seg_t0", "customers", "still_active_pct",
            "to_core_pct", "to_dormant_pct", "avg_orders_90d_t1",
            "avg_spend_90d_t1"])
        frames["loyalty_by_segment"] = empty.copy()
        frames["loyalty_pooled"] = empty.copy()
        frames["meta"]["anchored_tables"] = f"missing: {exc}"

    # Its own try for the same reason as the two below: this table is newer
    # than the others, so a warehouse that predates it should still draw every
    # other tab rather than fail on the first read.
    try:
        frames["sales_trend"] = sales_trend(cat, cfg, exclude_outliers)
        frames["meta"]["sales_trend_table"] = "present"
    except NoSuchTableError as exc:
        frames["sales_trend"] = pd.DataFrame(columns=[
            "sales_date_local", "restaurant_id", "item_category", "hour_local",
            "daypart", "day_of_week", "dow_number", "is_weekend", "is_holiday",
            "holiday_name", "week_start", "month_start", "sales_date_utc",
            "orders", "line_items", "units", "total_revenue",
            "outlier_orders", "outlier_line_items", "outlier_units",
            "outlier_revenue", "week_complete",
            "month_complete"])
        frames["meta"]["sales_trend_table"] = f"missing: {exc}"

    # Its own try, like the one above: this is the newest table in the
    # warehouse, so a clone that predates it must still draw the other four
    # tabs and the two sections of this one that do not use it.
    try:
        frames["join_profile"] = join_profile(cat, cfg, exclude_outliers)
        frames["meta"]["join_profile_table"] = "present"
    except NoSuchTableError as exc:
        frames["join_profile"] = pd.DataFrame(columns=[
            "coverage_days", "period_index", "period_start_day",
            "period_end_day", "window_days", "customers", "orders",
            "line_items", "units", "total_revenue", "active_days",
            "outlier_customers", "outlier_orders", "outlier_line_items",
            "outlier_units", "outlier_revenue", "outlier_active_days",
            "median_order_value", "orders_per_customer",
            "visits_per_customer", "orders_per_visit", "avg_order_value",
            "revenue_per_customer"])
        frames["meta"]["join_profile_table"] = f"missing: {exc}"

    # A separate try, not folded into the one above: a missing CLV table must
    # not report the anchored tables as missing too, and vice versa. They are
    # published by different runs and either can be absent on its own.
    try:
        frames.update(gold_clv(
            cat, cfg,
            OUTLIERS_EXCLUDED if exclude_outliers else OUTLIERS_INCLUDED))
        frames["meta"]["clv_tables"] = "present"
    # RuntimeError as well as NoSuchTableError: `gold_clv` raises it when the
    # tables exist but predate the `outliers` grain, which is a stale-warehouse
    # problem with the same correct handling -- report it, draw the rest.
    except (NoSuchTableError, RuntimeError) as exc:
        frames["clv_profile"] = pd.DataFrame(columns=[
            "clv_tier", "customers", "pct_of_customers", "total_spend",
            "pct_of_spend", "median_total_spend", "avg_total_spend",
            "spend_floor", "spend_ceiling", "total_orders", "avg_orders",
            "median_orders", "avg_spend_per_order", "median_spend_per_order",
            "one_order_pct", "median_tenure_days", "median_recency_days",
            "pct_of_orders", "avg_units_per_order", "median_units_per_order",
            "volume_outliers", "anchor_date"])
        frames["clv_segments"] = pd.DataFrame(columns=[
            "clv_tier", "segment", "segment_group", "customers",
            "pct_of_tier", "pct_of_segment", "total_spend",
            "pct_of_tier_spend", "median_total_spend", "avg_orders",
            "median_orders", "avg_customer_order_value",
            "volume_outliers", "vip_spend_cut", "anchor_date"])
        frames["meta"]["clv_tables"] = f"missing: {exc}"

    return frames
