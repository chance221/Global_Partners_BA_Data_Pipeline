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

_ORDER_COLS = ("user_id", "order_id", "creation_time_utc", "item_price",
               "item_quantity", "is_loyalty")


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


# --- the one entry point --------------------------------------------------

def datasets(env: str = "prod") -> dict[str, pd.DataFrame]:
    """Every frame the dashboard draws, from one read of the warehouse."""
    cat, cfg = catalog(env)

    orders = per_order(cat, cfg)
    cust = customers(cat, cfg)

    # The anchor the page reports is silver's own last order date, never
    # batch_date -- which is pinned to 2026-10-16 in the job arguments and
    # would claim data from the future.
    anchor = max(orders.order_date)

    # gold was built at ITS anchor. If silver has moved on since the last Glue
    # run the two disagree, and every segment on the page is stale by that
    # many days. Surfaced rather than silently reconciled.
    gold_anchor = pd.to_datetime(cust.anchor_date.iloc[0]).date()

    frames: dict[str, pd.DataFrame] = {}
    frames.update(window_frames(cust, orders, anchor))
    frames.update(cadence_frames(cust))
    frames["reorder_cliff"] = reorder_cliff(orders, anchor)

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
    }])

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

    return frames
