"""Customer segmentation dashboard -- the 90-day RFM view, and what to do about it.

Reads the CSVs that `build_data.py` materialises. Nothing here touches Spark or
Iceberg: Streamlit reruns this whole script on every widget interaction, and a
Spark session costs a JVM start each time.

Run:
    ./.venv/Scripts/python.exe -m streamlit run dashboard/app.py

Charting rules this file follows, because breaking them produces a chart that is
confidently wrong rather than obviously broken:

  * No dual axes anywhere. Where two measures have different units they get
    separate panels, never a second y-scale.
  * The basket-size axis is zero-based. Average order value spans $11.91-$15.24
    and frequency spans 1.00-10.52; on a cropped axis the former looks like a
    real spread. The flatness IS the finding, so the axis has to show it.
  * Recency is never plotted as a measured bar. Five of seven active segments
    share the 1-29 day band because `orders_last_30d >= 1` is in their
    definition, so a recency bar restates the segment chain. It is encoded as
    row order and as the cadence ratio instead.
  * The as-of date is anchor_date, the last order in silver. Never batch_date,
    which is pinned to 2026-10-16 in the job arguments and would claim data
    from the future.
  * Colours come from a palette validated with the dataviz skill's checker in
    both light and dark mode (blue/orange: CVD dE 24.7 protan, 33.6 normal,
    both >= 3:1 on their surface).
"""

from __future__ import annotations

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

DATA = Path(__file__).resolve().parent / "data"

st.set_page_config(page_title="Customer Segmentation",
                   page_icon="📊", layout="wide")

# Segments ordered by the recency band their own definitions force, so every
# chart stacks them identically and one segment can be tracked down the page.
SEGMENT_ORDER = ["VIP", "Loyal", "Promising", "Re-engaging", "New",
                 "Slipping", "Lapsing", "Never Returned"]
GROUP_ORDER = ["Core", "Developing", "New Growth", "Churn Risk",
               "Infrequent", "Never Returned"]
PERIOD_ORDER = ["P3 61-90d", "P2 31-60d", "P1 last 30d"]
CLIFF_ORDER = ["same day", "1-3 days", "4-7 days", "8-14 days",
               "15-30 days", "31+ days", "never"]

_LIGHT = {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e",
          "grid": "#e6e5e1", "s1": "#2a78d6", "s2": "#eb6834",
          "critical": "#d03b3b", "good": "#0ca30c", "muted": "#9a9893"}
_DARK = {"surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7",
         "grid": "#33332f", "s1": "#3987e5", "s2": "#d95926",
         "critical": "#e66767", "good": "#1ab81a", "muted": "#6e6d67"}


def palette() -> dict[str, str]:
    """Light or dark steps, picked from the running Streamlit theme.

    Dark mode is a separate validated set of steps, not an automatic flip of
    the light ones -- the same hex that clears 3:1 on #fcfcfb does not
    necessarily clear it on #1a1a19.
    """
    try:
        base = st.get_option("theme.base") or "light"
    except Exception:
        base = "light"
    return _DARK if str(base).lower() == "dark" else _LIGHT


@st.cache_data
def load(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / f"{name}.csv")


def styled(chart: alt.Chart) -> alt.Chart:
    """Recessive axes and grid, no view border, text in ink tokens."""
    p = palette()
    return (chart
            .configure_view(strokeWidth=0, fill=p["surface"])
            .configure_axis(domainColor=p["grid"], tickColor=p["grid"],
                            gridColor=p["grid"], labelColor=p["ink2"],
                            titleColor=p["ink2"], labelFontSize=12,
                            titleFontSize=11, titleFontWeight="normal")
            .configure_legend(labelColor=p["ink2"], titleColor=p["ink2"],
                              labelFontSize=12, titleFontSize=11)
            .configure_title(color=p["ink"], fontSize=15, anchor="start",
                             subtitleColor=p["ink2"], subtitleFontSize=12))


def ranked_bar(frame: pd.DataFrame, value: str, key: str, *, title: str,
               subtitle: str, axis_title: str, label_fmt: str,
               order: list[str] | None = None, height: int = 300,
               color: str | None = None, xmax: float | None = None) -> alt.Chart:
    """One measure across categories: a single hue plus a direct label.

    One hue and not a categorical scale, because the category is already named
    on the axis -- colouring each bar differently would encode nothing and
    would repaint the survivors whenever a filter changes the row set.
    """
    p = palette()
    sort = order if order is not None else (
        frame.sort_values(value, ascending=False)[key].tolist())
    # An explicit domain is how the small-multiple panels stay comparable:
    # three separate charts cannot share a scale the way an Altair facet can.
    scale = alt.Scale(domain=[0, xmax]) if xmax else alt.Scale(nice=True)
    base = alt.Chart(frame).encode(
        y=alt.Y(f"{key}:N", sort=sort, title=None),
        x=alt.X(f"{value}:Q", title=axis_title, scale=scale),
        tooltip=[alt.Tooltip(f"{key}:N", title=key.replace("_", " ")),
                 *[alt.Tooltip(f"{c}:Q", format=",.2f")
                   for c in frame.columns if c != key and
                   pd.api.types.is_numeric_dtype(frame[c])]])
    bars = base.mark_bar(color=color or p["s1"], height=18,
                         cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
    labels = base.mark_text(align="left", dx=6, color=p["ink2"],
                            fontSize=12).encode(
        text=alt.Text(f"{value}:Q", format=label_fmt))
    return (alt.layer(bars, labels)
            .properties(height=height, title=alt.Title(title, subtitle=subtitle)))


def paired_dots(frame: pd.DataFrame, *, key: str, value: str, phase: str,
                order: list[str], phase_order: list[str], title: str,
                subtitle: str, axis_title: str, height: int = 320) -> alt.Chart:
    """Two states of the same category on ONE axis, joined by a rule.

    A dumbbell rather than grouped bars because the quantity of interest is the
    distance between the two values, and a rule shows distance directly. Both
    series are in the legend AND the tooltip, so identity is never carried by
    colour alone.
    """
    p = palette()
    # Reindexed before pivoting: a category with no rows for one phase (a
    # segment nobody moved out of, for instance) would otherwise be missing
    # from the long frame entirely and its rule would silently not draw.
    full = pd.MultiIndex.from_product([order, phase_order], names=[key, phase])
    frame = (frame.set_index([key, phase]).reindex(full, fill_value=0)
             .reset_index())
    wide = frame.pivot(index=key, columns=phase, values=value).reset_index()
    rules = alt.Chart(wide).mark_rule(color=p["muted"], strokeWidth=2).encode(
        y=alt.Y(f"{key}:N", sort=order, title=None),
        x=alt.X(f"{phase_order[0]}:Q", title=axis_title),
        x2=alt.X2(phase_order[1]))
    dots = alt.Chart(frame).mark_point(filled=True, size=130,
                                       stroke=p["surface"],
                                       strokeWidth=2).encode(
        y=alt.Y(f"{key}:N", sort=order, title=None),
        x=alt.X(f"{value}:Q", title=axis_title),
        color=alt.Color(f"{phase}:N",
                        scale=alt.Scale(domain=phase_order,
                                        range=[p["s1"], p["s2"]]),
                        legend=alt.Legend(title=None, orient="top")),
        tooltip=[alt.Tooltip(f"{key}:N"), alt.Tooltip(f"{phase}:N"),
                 alt.Tooltip(f"{value}:Q", format=".1f")])
    return (alt.layer(rules, dots)
            .properties(height=height, title=alt.Title(title, subtitle=subtitle)))


# --- page ----------------------------------------------------------------

meta = load("meta").iloc[0]
p = palette()

st.title("Customer segmentation: the last 90 days")
st.caption(
    f"As of **{meta.anchor_date}** (the last order in silver, not the pipeline "
    f"batch date) · {int(meta.active_customers):,} active customers · "
    f"${meta.spend_90d:,.2f} · {int(meta.orders_90d):,} orders · "
    f"VIP threshold frozen at ${meta.frozen_vip_cut:.2f} for era comparisons")

st.info(
    "**Dormant customers are excluded throughout.** All 16,382 of them have "
    "`orders_last90d = 0` and `spend_last90d = 0` by definition, which is what "
    "`is_active` exists to express. Lifetime figures, where shown, say so.",
    icon=":material/filter_alt:")

tab_value, tab_freq, tab_cadence, tab_reorder, tab_program = st.tabs(
    ["Where the value is", "Frequency, not basket", "Who is overdue",
     "The second order", "Is the program working"])


# --- 1. where the value is -----------------------------------------------

with tab_value:
    grain = st.radio("Grain", ["segment", "segment_group"], horizontal=True,
                     key="grain_value")
    frame = load(f"window_{grain}")
    order = SEGMENT_ORDER if grain == "segment" else GROUP_ORDER
    order = [o for o in order if o in set(frame[grain])]

    left, right = st.columns(2)
    with left:
        st.altair_chart(styled(ranked_bar(
            frame, "revenue", grain,
            title="90-day revenue",
            subtitle="Active customers only. Ordered by value.",
            axis_title="revenue ($)", label_fmt="$,.0f")),
            width="stretch")
    with right:
        st.altair_chart(styled(ranked_bar(
            frame, "rev_per_cust", grain,
            title="Revenue per customer",
            subtitle="The same 90 days, divided by headcount.",
            axis_title="revenue per customer ($)", label_fmt="$,.2f")),
            width="stretch")

    st.markdown(
        f"**VIP is {frame.loc[frame[grain].eq('VIP'), 'rev_share_pct'].sum() if 'VIP' in set(frame[grain]) else 0:.1f}% "
        "of 90-day revenue** on a small headcount, while the large segments "
        "carry their share on volume of people rather than value per person. "
        "The two panels disagree precisely because `revenue = customers x "
        "orders per customer x average order`, and the three factors do not "
        "move together.")

    with st.expander("The three 30-day periods, and why some cells are empty"):
        per = load(f"period_{grain}")
        # Explicit small multiples rather than an Altair facet: each panel
        # keeps its own title, and a shared x domain is set by hand so the
        # three periods stay comparable.
        xmax = float(per.revenue.max()) * 1.18
        cols = st.columns(3)
        for col, period in zip(cols, PERIOD_ORDER):
            sub = per[per.period.eq(period)]
            with col:
                st.altair_chart(styled(ranked_bar(
                    sub, "revenue", grain, order=order,
                    title=period,
                    subtitle=f"${sub.revenue.sum():,.0f} in this period",
                    axis_title="revenue ($)", label_fmt="$,.0f",
                    height=260, xmax=xmax)), width="stretch")
        st.caption(
            f"All three panels share an axis to ${xmax:,.0f} and the same row "
            "order, so a missing bar is visibly a gap rather than a short bar.")
        st.warning(
            "**Read the gaps as arithmetic.** Slipping and Lapsing are absent "
            "from P1 because both require `orders_last_30d = 0` to be in those "
            "segments at all. Re-engaging is absent from P2 and P3 because it "
            "requires both to be zero. New all but vanishes after P1 because "
            "`tenure_days <= 30`. None of those is a trend.",
            icon=":material/warning:")


# --- 2. frequency, not basket --------------------------------------------

with tab_freq:
    frame = load("window_segment")
    st.subheader("All the variance is on the frequency axis")

    span_aov = frame.avg_order.max() / frame.avg_order.min()
    span_freq = frame.orders_per_cust.max() / frame.orders_per_cust.min()
    a, b, c = st.columns(3)
    a.metric("Average order value spread", f"{span_aov:.2f}x",
             f"${frame.avg_order.min():.2f} - ${frame.avg_order.max():.2f}")
    b.metric("Orders per customer spread", f"{span_freq:.1f}x",
             f"{frame.orders_per_cust.min():.2f} - "
             f"{frame.orders_per_cust.max():.2f}")
    c.metric("Median items per order", f"{int(frame.med_units.median())}",
             "every segment, every period", delta_color="off")

    scatter = alt.Chart(frame).mark_point(
        filled=True, stroke=p["surface"], strokeWidth=2).encode(
        # Zero-based on purpose. A 1.28x spread on a cropped axis reads as a
        # large one; the point of this chart is that the spread is small.
        x=alt.X("avg_order:Q", title="average order value ($)",
                scale=alt.Scale(domain=[0, float(frame.avg_order.max()) * 1.15])),
        y=alt.Y("orders_per_cust:Q", title="orders per customer (90 days)",
                scale=alt.Scale(zero=True, nice=True)),
        size=alt.Size("customers:Q", title="customers",
                      scale=alt.Scale(range=[80, 1200]),
                      legend=alt.Legend(orient="right")),
        color=alt.value(p["s1"]),
        tooltip=[alt.Tooltip("segment:N"),
                 alt.Tooltip("customers:Q", format=","),
                 alt.Tooltip("avg_order:Q", title="avg order", format="$.2f"),
                 alt.Tooltip("med_order:Q", title="median order", format="$.2f"),
                 alt.Tooltip("orders_per_cust:Q", title="orders/cust",
                             format=".2f"),
                 alt.Tooltip("rev_per_cust:Q", title="revenue/cust",
                             format="$.2f")])
    labels = alt.Chart(frame).mark_text(
        align="left", dx=12, dy=-10, color=p["ink2"], fontSize=12).encode(
        x=alt.X("avg_order:Q",
                scale=alt.Scale(domain=[0, float(frame.avg_order.max()) * 1.15])),
        y=alt.Y("orders_per_cust:Q"), text="segment:N")
    st.altair_chart(styled(alt.layer(scatter, labels).properties(
        height=430,
        title=alt.Title(
            "Basket size against visit frequency",
            subtitle="Bubble area = customers. The x-axis starts at zero, so "
                     "the narrowness of the basket spread is visible."))),
        width="stretch")

    st.success(
        "**Average order value does not discriminate between segments; "
        "frequency does.** Basket size spans "
        f"{span_aov:.2f}x while visits span {span_freq:.1f}x, and the median "
        "order is a single item in every segment and every period. The "
        "relationship even runs backwards: Lapsing has the highest average "
        "order ($14.67) and the lowest frequency (1.31), while Loyal has the "
        "lowest average order ($11.91) at 4.2x the visits. There is no "
        "upsell story in this data -- an add-on campaign chases a 1.3x spread "
        "and ignores a 10x one.",
        icon=":material/insights:")

    st.altair_chart(styled(ranked_bar(
        frame, "orders_per_cust", "segment",
        title="Orders per customer, 90 days",
        subtitle="The axis that actually separates the segments.",
        axis_title="orders per customer", label_fmt=".2f", height=280)),
        width="stretch")


# --- 3. who is overdue ---------------------------------------------------

with tab_cadence:
    frame = load("cadence_segment")
    scored = frame.dropna(subset=["med_overdue"]).copy()
    no_cadence = frame[frame.med_overdue.isna()]

    st.subheader("Late by their own standard, not by the calendar")
    st.caption(
        "`recency_days / avg_gap_days`, divided per customer and then taken at "
        "the median. A ratio of medians is not the median of the ratio, and "
        "here they differ enough to matter.")

    scored["label"] = (scored.med_recency.astype(int).astype(str) + "d silent / "
                       + scored.med_gap.round(1).astype(str) + "d normal")
    sort = scored.sort_values("med_overdue", ascending=False).segment.tolist()

    base = alt.Chart(scored).encode(
        y=alt.Y("segment:N", sort=sort, title=None),
        x=alt.X("med_overdue:Q", title="recency as a multiple of own cadence"))
    bars = base.mark_bar(height=20, cornerRadiusTopRight=4,
                         cornerRadiusBottomRight=4).encode(
        # The only place a status colour is used: being overdue IS a state,
        # and it ships with the ratio in text so colour is not load-bearing.
        color=alt.condition(alt.datum.med_overdue >= 2,
                            alt.value(p["critical"]), alt.value(p["s1"])),
        tooltip=[alt.Tooltip("segment:N"),
                 alt.Tooltip("customers:Q", format=","),
                 alt.Tooltip("med_recency:Q", title="median days silent"),
                 alt.Tooltip("med_gap:Q", title="median normal gap"),
                 alt.Tooltip("med_overdue:Q", title="overdue", format=".2f"),
                 alt.Tooltip("pct_2x_overdue:Q", title="% at 2x+",
                             format=".1f"),
                 alt.Tooltip("lifetime_revenue:Q", title="lifetime revenue",
                             format="$,.0f")])
    text = base.mark_text(align="left", dx=8, color=p["ink2"],
                          fontSize=12).encode(text="label:N")
    on_time = alt.Chart(pd.DataFrame({"x": [1.0]})).mark_rule(
        color=p["ink2"], strokeDash=[5, 4], strokeWidth=1.5).encode(x="x:Q")
    st.altair_chart(styled(alt.layer(bars, text, on_time).properties(
        height=330,
        title=alt.Title(
            "How overdue each segment is, against its own rhythm",
            subtitle="Dashed line = 1.0x, exactly on schedule. Right of it is "
                     "late; left of it is early."))),
        width="stretch")

    if len(no_cadence):
        row = no_cadence.iloc[0]
        st.warning(
            f"**{row.segment} ({int(row.customers):,} customers) has no bar, "
            "and that is the finding.** A single lifetime order means there is "
            "no gap to be late against, so `avg_gap_days` is NULL. They are "
            f"{int(row.med_recency)} days silent with no rhythm ever "
            "established -- an acquisition failure, not a retention one.",
            icon=":material/help:")

    st.success(
        "**Slipping is the intervention list; Lapsing is not.** Slipping sits "
        "at 3.01x its own cadence with 71.1% of its customers at 2x or worse. "
        "Lapsing's 61-day silence is close to its own 45.8-day rhythm (1.22x) "
        "-- those customers are infrequent by nature and always were, so a "
        "win-back campaign treats normal behaviour as churn.",
        icon=":material/target:")
    st.caption(
        "Honest caveat for the pitch: 33.0% of Lapsing is still at 2x or worse "
        "-- about 361 customers. The median says on-schedule; a third are not. "
        "The defensible framing is Slipping first, then the overdue tail of "
        "Lapsing, rather than ignoring Lapsing.")


# --- 4. the second order -------------------------------------------------

with tab_reorder:
    frame = load("reorder_cliff").sort_values("bucket_order")
    st.subheader("How fast the second order arrives, and what it is worth")
    st.caption(
        "Every customer whose first order is at least 90 days before the "
        "anchor, so nobody is right-censored: a customer who first ordered "
        "three days ago cannot have reordered within seven.")

    never = frame[frame.bucket.eq("never")].iloc[0]
    fast = frame[frame.bucket.isin(["same day", "1-3 days"])]
    a, b, c = st.columns(3)
    a.metric("Never place a second order", f"{never.pct_of_customers:.1f}%",
             f"{int(never.customers):,} customers", delta_color="off")
    b.metric("Reorder within 3 days", f"{fast.pct_of_customers.sum():.1f}%",
             f"{fast.pct_of_revenue.sum():.1f}% of lifetime revenue",
             delta_color="off")
    c.metric("Lifetime value, 1-3 days vs 31+ days",
             f"{frame[frame.bucket.eq('1-3 days')].avg_life_revenue.iloc[0] / frame[frame.bucket.eq('31+ days')].avg_life_revenue.iloc[0]:.1f}x",
             "$232.56 against $87.35", delta_color="off")

    left, right = st.columns(2)
    with left:
        st.altair_chart(styled(ranked_bar(
            frame, "customers", "bucket", order=CLIFF_ORDER,
            title="How many customers",
            subtitle="Time from first order to second.",
            axis_title="customers", label_fmt=",.0f", height=300)),
            width="stretch")
    with right:
        st.altair_chart(styled(ranked_bar(
            frame, "avg_life_revenue", "bucket", order=CLIFF_ORDER,
            title="What each one became",
            subtitle="Average lifetime revenue per customer.",
            axis_title="lifetime revenue ($)", label_fmt="$,.0f",
            height=300, color=p["s2"])),
            width="stretch")

    st.success(
        "**The break is at three days, not seven.** 1-3 days is worth $232.56 "
        "of lifetime revenue; 4-7 days is $168.35 and 8-14 days is $166.55 -- "
        "indistinguishable from each other, so there is no seven-day boundary "
        "in the data to build a trigger on. The decay after day three is "
        "smooth.",
        icon=":material/insights:")
    st.error(
        "**Do not tell marketing that a customer silent past day 7 is lost.** "
        "Of customers still silent at day 7, 39.1% eventually came back. "
        "Writing them off discards a recoverable third. What is true and still "
        "a strong pitch: when a New customer does reorder, 73.7% of the time "
        "it happens inside seven days.",
        icon=":material/dangerous:")
    st.caption(
        "The headline is blunter than any cliff: 48.6% of customers never "
        "place a second order at all. Those 9,141 people are half the customer "
        "base and 9.0% of lifetime revenue, while the 2,593 who reorder within "
        "three days are 13.8% of customers and 37.7% of revenue.")


# --- 5. is the program working -------------------------------------------

with tab_program:
    st.subheader("Two readings of the loyalty program, and they disagree in scale")

    trans = load("transitions")
    # New Growth and Never Returned are structurally unreachable as
    # destinations over 180 days, so they are zero in every row. Showing a
    # column of zeros invites the reading that acquisition stopped.
    trans = trans[~trans.grp_t1.isin(["New Growth", "Never Returned"])]
    era_order = ["Pre-loyalty", "Post-loyalty"]
    seg_order = [s for s in SEGMENT_ORDER + ["Dormant"]
                 if s in set(trans.seg_t0)]

    st.markdown("##### 1. Calendar comparison: two 180-day spans, one each side of launch")
    left, right = st.columns(2)
    for col, grp, label, note in (
            (left, "Core", "Reached Core 180 days later", "higher is better"),
            (right, "Dormant", "Went Dormant 180 days later",
             "LOWER is better -- rightward is worse on this panel")):
        sub = trans[trans.grp_t1.eq(grp)][["era", "seg_t0", "row_pct"]]
        with col:
            st.altair_chart(styled(paired_dots(
                sub, key="seg_t0", value="row_pct", phase="era",
                order=seg_order, phase_order=era_order,
                title=label, subtitle=note,
                axis_title="% of that segment")), width="stretch")

    st.caption(
        "Both spans use the VIP threshold frozen at $49.95. A floating "
        "percentile pins VIP near a fixed share of actives, which would make "
        "this comparison measure nothing. The post span is also measured "
        "through Christmas (-73.5% against an ordinary day) while the pre span "
        "is not, so the post figures are if anything understated.")

    st.warning(
        "**This is not a controlled experiment.** The two spans are different "
        "calendar periods, so everything else that changed in ten months is "
        "inside the delta -- including the ~29 menu reprices of January 2023, "
        "which land between the two windows.",
        icon=":material/warning:")

    st.divider()
    st.markdown("##### 2. Apples to apples: same window, members against non-members")

    pooled = load("loyalty_pooled")
    by_seg = load("loyalty_by_segment")
    cohort_order = ["non-member", "member"]
    seg2 = [s for s in SEGMENT_ORDER + ["Dormant"] if s in set(by_seg.seg_t0)]

    m = pooled.set_index("cohort")
    a, b = st.columns(2)
    a.metric("Pooled: still active after 180 days",
             f"{m.loc['member', 'still_active_pct']:.1f}% member",
             f"vs {m.loc['non-member', 'still_active_pct']:.1f}% non-member")
    worst = (by_seg.pivot(index="seg_t0", columns="cohort",
                          values="still_active_pct")
             .assign(delta=lambda d: d["member"] - d["non-member"]))
    b.metric("Within-segment advantage, range",
             f"+{worst.delta.min():.1f} to +{worst.delta.max():.1f} pp",
             "positive in all 9 segments", delta_color="off")

    st.altair_chart(styled(paired_dots(
        by_seg[["seg_t0", "cohort", "still_active_pct"]],
        key="seg_t0", value="still_active_pct", phase="cohort",
        order=seg2, phase_order=cohort_order,
        title="Still active 180 days later, by membership at the start",
        subtitle="Split within each starting segment, which holds engagement "
                 "level fixed.",
        axis_title="% still active", height=360)), width="stretch")

    st.success(
        "**Members retained better in every one of the nine segments**, by "
        f"+{worst.delta.min():.1f} to +{worst.delta.max():.1f} percentage "
        "points. The biggest effect is on New customers: 23.4% of members were "
        "still active against 10.9% of non-members -- enrollment roughly "
        "doubles New retention. Set against the era-level finding that New "
        "customers fell 48.6%, the implication is specific: the program works "
        "on the new customers it reaches, and the problem is reach and "
        "acquisition, not program design.",
        icon=":material/insights:")

    st.error(
        "**The pooled number is the one to distrust.** 32.7% against 6.1% looks "
        "like a five-fold effect, but members are concentrated in engaged "
        "segments, so most of that gap is composition. Within-segment the "
        "honest range is +4.2 to +12.5pp -- real, consistent, and two to six "
        "times smaller than the pooled figure implies.",
        icon=":material/dangerous:")
    st.caption(
        "Remaining bias, not removed by this design: joining is self-selected, "
        "so members were already more inclined to return. Comparing within the "
        "starting segment controls for engagement level at the start but not "
        "for that propensity, so these deltas are an upper bound rather than a "
        "causal estimate. Cells with a thin non-member count -- Loyal (50), "
        "VIP (69), Slipping (73) -- should not carry weight on their own.")

    with st.expander("Full transition matrix"):
        wide = (trans.pivot_table(index=["era", "seg_t0"], columns="grp_t1",
                                  values="row_pct", aggfunc="first")
                .reindex(era_order, level=0).fillna(0).round(1))
        st.dataframe(wide, width="stretch")
        st.caption(
            "Row percentages. `New Growth` and `Never Returned` are omitted: "
            "New requires tenure <= 30 days, which nobody present at the start "
            "can satisfy 180 days later, and Never Returned requires a single "
            "lifetime order, which would place the customer past 180 days of "
            "recency and so into Dormant first. Both are structurally zero.")
