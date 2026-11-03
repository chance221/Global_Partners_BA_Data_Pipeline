"""Customer segmentation dashboard -- the 90-day RFM view, and what to do about it.

Reads the pipeline's own Iceberg tables live, through `warehouse.py`: PyIceberg
against the Glue catalog, no Spark and no intermediate files. The whole read is
about eight seconds cold and is cached for the session, so there is no build
step to forget and nothing that can go stale behind a daily job.

Every derived frame comes from ONE read of the warehouse, so the segments and
the orders on this page cannot come from either side of a pipeline run.

Run:
    ./.venv/Scripts/python.exe -m streamlit run dashboard/app.py --server.address 127.0.0.1

`--server.address 127.0.0.1` because Streamlit otherwise binds every interface,
which publishes the page to the network it is running on.

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

import os
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

# The repo root, so `python_scripts` (which warehouse.py reads the catalog
# config from) resolves whether or not the project is pip-installed. Streamlit
# puts only this file's own directory on sys.path.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard import warehouse  # noqa: E402

ENV = os.environ.get("GP_DASHBOARD_ENV", "prod")

st.set_page_config(page_title="Customer Segmentation",
                   page_icon="📊", layout="wide")

# Segments ordered by the recency band their own definitions force, so every
# chart stacks them identically and one segment can be tracked down the page.
# The tab LABELS, kept as constants because the page copy cross-references
# them by name. They used to be referenced by position -- "the first tab",
# "the last tab" -- which a reorder silently falsifies, and did. A name
# cannot go stale that way. Defined up here rather than beside `st.tabs`
# because the not-yet-published warnings above the tabs name them too.
TAB_CLV = "Customer Lifetime Value"
TAB_VALUE = "Where the Value Is"
TAB_PROGRAM = "Is the program working"
TAB_TREND = "Sales Trend Monitoring"

SEGMENT_ORDER = ["VIP", "Loyal", "Promising", "Re-engaging", "New",
                 "Slipping", "Lapsing", "Never Returned"]
GROUP_ORDER = ["Core", "Developing", "New Growth", "Churn Risk",
               "Infrequent", "Never Returned"]
PERIOD_ORDER = ["P3 61-90d", "P2 31-60d", "P1 last 30d"]
CLIFF_ORDER = ["same day", "1-3 days", "4-7 days", "8-14 days",
               "15-30 days", "31+ days", "never"]

# The nine segments and the rule each one actually encodes, in the same row
# order every chart below uses. This is the page's legend: the segmentation is
# the only dimension on every tab, so the definitions belong once at the top
# rather than restated per chart.
#
# Transcribed from the CASE in `customer_rfm.py`, which is evaluated top to
# bottom with the first match winning. Anything edited there has to be edited
# here -- there is no test tying the two together, because prose cannot be
# asserted against a CASE expression.
SEGMENT_DEFS = [
    ("VIP", "Core",
     "Two or more orders in the **last 30 days**, at least one more in days "
     "**31-90**, and 90-day spend at or above the VIP cut of **${cut:,.2f}** "
     "-- the 80th percentile of 90-day spend among active customers, "
     "recomputed on every run. All three legs must hold: spend alone is not "
     "enough, and 63 Promising customers clear the spend bar on too few "
     "orders. **The group to protect.**"),
    ("Loyal", "Core",
     "At least one order in the **last 30 days** and **four or more** across "
     "the 90 days. Four is the top fifth of 90-day order count -- the 80th "
     "percentile is 3 -- so this is frequency, not spend. Loyal is where a "
     "frequent customer lands when they miss any single leg of the VIP test."),
    ("Promising", "Developing",
     "Ordered in the **last 30 days** but fewer than four times across the "
     "90 days. Recent and active, with no rhythm established yet."),
    ("Re-engaging", "Infrequent",
     "Ordered in the **last 30 days**, **nothing at all** in days 31-90, and "
     "a gap of **180 days or more** between that order and the one before "
     "it. A win-back from real dormancy, not a quiet quarter."),
    ("New", "New Growth",
     "**First order ever** within the last 30 days. Tenure, not volume: 17 of "
     "these 443 ordered four or more times inside those three weeks and would "
     "read as Loyal on frequency alone, but New is tested first and wins."),
    ("Slipping", "Churn Risk",
     "**Three or more** orders across the 90 days and **none** in the last "
     "30. A rhythm that has stopped, which is exactly what separates it from "
     "Lapsing. **The group to recapture.**"),
    ("Lapsing", "Infrequent",
     "One or two orders in days 31-90, none in the last 30, and **more than "
     "one order** in their lifetime. Infrequent by nature rather than lost -- "
     "the cadence panel settles which."),
    ("Never Returned", "Never Returned",
     "**Exactly one order ever**, placed more than 30 and less than 90 days "
     "ago. Not infrequency but an acquisition that never produced a second "
     "visit, which is why it is split out instead of pooled into Lapsing."),
    ("Dormant", "Dormant",
     "**No orders at all** in the 90 days before the anchor. Four fifths of "
     "the base, and excluded from every 90-day measure here because all of "
     "their window figures are zero by construction."),
]

# `ramp` is the ordinal scale the revenue pie uses: ONE hue, monotone
# lightness, eight steps. Not categorical hues -- the slices are sorted by
# share, so lightness is carrying magnitude, and eight distinguishable
# categorical hues would need a palette this project has never validated.
# Both ramps clear the dataviz ordinal checks against their own surface
# (monotone L, adjacent dL >= 0.06, light end >= 2:1):
#   light  #7cbbf0 at 2.00:1, hue spread 17 degrees
#   dark   #085076 at 2.01:1, hue spread 3 degrees
_LIGHT = {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e",
          "grid": "#e6e5e1", "s1": "#2a78d6", "s2": "#eb6834",
          "critical": "#d03b3b", "good": "#0ca30c", "muted": "#9a9893",
          "ramp": ["#7cbbf0", "#62a0d4", "#4986b8", "#2f6d9e",
                   "#115583", "#003d6a", "#002651", "#000f3a"]}
_DARK = {"surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7",
         "grid": "#33332f", "s1": "#3987e5", "s2": "#d95926",
         "critical": "#e66767", "good": "#1ab81a", "muted": "#6e6d67",
         "ramp": ["#97d6ff", "#83c2ee", "#70aed9", "#5c9ac4",
                  "#4987b0", "#36749c", "#226189", "#085076"]}


def theme_base() -> str:
    """The theme the VIEWER is actually seeing, not the one in config.

    `st.get_option("theme.base")` returns the *configured* theme, which is
    None unless someone has written `.streamlit/config.toml`. So on a dark
    page it reported "light", every chart took the light steps, and the
    titles rendered in #0b0b0b ink on a #1a1a19 background -- legible nowhere.
    The chart area itself was filled light by `configure_view`, which is why
    the plot bodies looked right while the text around them did not.

    `st.context.theme.type` is resolved in the browser and reports what is on
    screen, including the "follow the system setting" case that has no config
    value at all. Streamlit reruns the script when it changes, so the charts
    repaint on a theme switch.
    """
    try:
        resolved = st.context.theme.type
        if resolved:
            return str(resolved).lower()
    except Exception:
        # No script run context (AppTest, a bare import), or an older
        # Streamlit without st.context.theme.
        pass
    try:
        return str(st.get_option("theme.base") or "light").lower()
    except Exception:
        return "light"


def palette() -> dict:
    """Light or dark steps, picked from the theme on screen.

    Dark mode is a separate validated set of steps, not an automatic flip of
    the light ones -- the same hex that clears 3:1 on #fcfcfb does not
    necessarily clear it on #1a1a19.
    """
    return _DARK if theme_base() == "dark" else _LIGHT


def ramp(n: int) -> list[str]:
    """`n` steps of the validated ordinal ramp, MOST PROMINENT FIRST.

    Sliced from the eight-step ramp rather than regenerated, so the two steps
    that were actually checked against the surface -- the lightest and the
    darkest -- are always the two in use. Fewer categories therefore means
    wider gaps between steps, never a narrower span.

    The stored ramp runs light to dark. Which end is *prominent* depends on
    the surface: the dark end stands out on #fcfcfb and recedes on #1a1a19.
    So the light-mode list is reversed and the dark-mode list is not, and a
    caller can hand the first step to its largest value in either theme.
    Without this the biggest slice of the revenue pie came out the palest one
    in light mode, which inverts the whole encoding.
    """
    full = palette()["ramp"]
    steps = (full[:n] if n >= len(full) or n < 2
             else [full[round(i * (len(full) - 1) / (n - 1))]
                   for i in range(n)])
    return steps if theme_base() == "dark" else steps[::-1]


# Steps in the matrix shading below. Six is measured, not a round number: it
# is the step count whose WORST cell contrast is the highest in both themes at
# once -- light 5.38:1 and dark 4.46:1, against 5.05/4.42 for every other
# count from 4 to 9. Dark mode cannot do better than about 4.4 at the ramp's
# mid-tones, where both ink ends are equally far away, so its two crossover
# steps sit a hair under the 4.5:1 AA floor whatever the sampling.
#
# Acceptable HERE and not in a chart: every cell prints its own value, so the
# shade is a second encoding of a number already written in text rather than
# the only way to read it.
_SHADE_STEPS = 6


def _luminance(hex_colour: str) -> float:
    """WCAG relative luminance of a `#rrggbb` string."""
    r, g, b = (int(hex_colour[i:i + 2], 16) / 255 for i in (1, 3, 5))
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
           for c in (r, g, b)]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def _contrast(fg: str, bg: str) -> float:
    """WCAG contrast ratio: 1.0 for identical colours, 21.0 black on white."""
    hi, lo = sorted((_luminance(fg), _luminance(bg)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def _ink_for(bg: str, p: dict) -> str:
    """Whichever of the two ink ends actually reads against `bg`.

    Picked by measured contrast with the cell's own background, not by where
    the cell's value sits in its range. Those are not the same test and the
    difference is visible: this ramp spans most of the lightness axis, so the
    step in the MIDDLE of it is already dark enough to need light ink, and
    flipping at the halfway value put near-black text on a navy cell.
    """
    return max((p["ink"], p["surface"]), key=lambda ink: _contrast(ink, bg))


def shaded(frame: pd.DataFrame, *, axis: int | None, fmt: str = "{:.2f}"):
    """A numeric matrix with every cell shaded by its magnitude.

    A one-hue sequential ramp from `palette()`, with the value printed in
    every cell, so colour is a second encoding of something already written
    in text and never the only way to read a cell. That is the encoding the
    tier-by-segment heatmap carried before this table replaced it; what
    changed is the order of priority, with the exact figures leading and the
    shape following behind them.

    Hand-rolled rather than `Styler.background_gradient`, which needs
    matplotlib. A plotting stack is a steep dependency for a page whose whole
    premise is that it runs with no JVM on a small instance, and it would not
    read `palette()` anyway, so the table would drift from every chart beside
    it on a theme change.

    `axis` is the direction the percentages sum to 100 in -- 1 for row
    percentages, 0 for column percentages -- so a shade only ever compares
    cells that share a denominator. Shading the other way round would rank a
    tier's 58% against a segment's 4% as though they were one measure.

    `axis=None` shades against the largest cell in the WHOLE frame instead,
    which is the right reading when every cell shares one denominator. A
    staffing grid compares Friday 11am against Monday 3pm directly; shading
    it row by row would give every weekday its own private scale and make
    each one's quietest hour look equally dead.

    Scaled from zero to the largest value along that axis, not min-to-max:
    under min-to-max the smallest cell in every line is forced to the palest
    step whatever it holds, which makes an 18% read like an empty cell.

    BOTH colours are set on every cell, the ink by `_ink_for` against that
    cell's own background. Streamlit takes a dataframe's text colour from the
    active theme, so a background set without an ink is unreadable in one of
    the two themes -- and which end of the ramp goes pale swaps between them,
    so neither ink can be hard-coded.
    """
    p = palette()
    # `ramp` hands its first step to the largest value in either theme, so
    # reversing it once here lets the index below grow with the magnitude.
    steps = ramp(_SHADE_STEPS)[::-1]
    inks = [_ink_for(step, p) for step in steps]

    def top_of(values) -> float:
        """Largest value, ignoring NaN, and 0.0 for an all-empty line."""
        return max((float(v) for v in values if pd.notna(v)), default=0.0)

    def css_for(values, top: float) -> list[str]:
        out = []
        for raw in values:
            frac = (0.0 if top <= 0 or pd.isna(raw)
                    else max(float(raw), 0.0) / top)
            i = min(int(frac * len(steps)), len(steps) - 1)
            out.append(f"background-color: {steps[i]}; color: {inks[i]}")
        return out

    if axis is None:
        def css_block(block: pd.DataFrame) -> pd.DataFrame:
            top = top_of(block.to_numpy().ravel())
            return pd.DataFrame(
                [css_for(row, top) for row in block.to_numpy()],
                index=block.index, columns=block.columns)

        return frame.style.apply(css_block, axis=None).format(fmt)

    def css_line(line: pd.Series) -> list[str]:
        return css_for(line, top_of(line))

    return frame.style.apply(css_line, axis=axis).format(fmt)


@st.cache_data(ttl=900, show_spinner="Reading the warehouse...")
def load_all(exclude_outliers: bool) -> dict[str, pd.DataFrame]:
    """One read of the warehouse, cached for fifteen minutes.

    Cached as a whole rather than per frame so a pipeline run cannot land
    between two queries and leave the page quoting segments from one state of
    gold against orders from another. The TTL is what makes the page pick up a
    daily job without anyone restarting it.

    `exclude_outliers` is a cache KEY, so the two populations are two cache
    entries and flipping the toggle back is instant rather than a second trip
    to the catalog. It costs one extra set of frames in memory, which at 20k
    customers is nothing.
    """
    return warehouse.datasets(ENV, exclude_outliers=exclude_outliers)


def load(name: str) -> pd.DataFrame:
    """One frame, from whichever population the sidebar toggle selected.

    Reads the module-level `EXCLUDE_OUTLIERS`, which the sidebar sets before
    any `load` call below it runs. A forward reference rather than a parameter
    on purpose: threading the flag through all forty-odd call sites would make
    it possible to miss one, and a chart reading the unfiltered frames while
    the rest of the page reads the filtered ones is the exact failure this
    toggle has to avoid.
    """
    return load_all(EXCLUDE_OUTLIERS)[name]


def styled(chart: alt.Chart) -> alt.Chart:
    """Recessive axes and grid, no view border, text in ink tokens.

    `background` is set as well as the view fill, so the title and legend sit
    on the same surface the palette's ink was chosen against rather than on
    whatever Streamlit paints behind the chart. Without it the view rect was
    filled and the text around it was not, which is how a dark-mode title
    ended up invisible. It also means that if theme detection is ever wrong
    the chart reads as a mis-toned card rather than as blank space.
    """
    p = palette()
    return (chart
            .configure(background=p["surface"])
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


def trend_line(frame: pd.DataFrame, *, x: str, y: str, title: str,
               subtitle: str, y_title: str, value_fmt: str = "$,.0f",
               height: int = 300) -> alt.Chart:
    """One measure over time, with a crosshair instead of a dot per point.

    A line is the form for change over time, and the granularity control
    below goes down to the day -- over 1,300 points. A marker on every one is
    a smear, and a label on every one is unreadable, so the hover layer is a
    rule that snaps to the nearest x and reveals that point alone.

    Single hue: there is one series, so a legend would name a colour nobody
    needs to distinguish, and the title says what is plotted.
    """
    p = palette()
    hover = alt.selection_point(nearest=True, on="mouseover", fields=[x],
                                empty=False, clear="mouseout")
    base = alt.Chart(frame).encode(
        x=alt.X(f"{x}:T", title=None),
        y=alt.Y(f"{y}:Q", title=y_title, scale=alt.Scale(nice=True)))
    line = base.mark_line(strokeWidth=2, color=p["s1"])
    # The rule and the point share one selection, so the crosshair and the
    # tooltip cannot disagree about which period is under the cursor.
    rule = base.mark_rule(color=p["ink2"]).encode(
        opacity=alt.condition(hover, alt.value(0.35), alt.value(0)))
    points = base.mark_circle(size=80, color=p["s1"]).encode(
        opacity=alt.condition(hover, alt.value(1), alt.value(0)),
        tooltip=[alt.Tooltip(f"{x}:T", title="period"),
                 alt.Tooltip(f"{y}:Q", title=y_title, format=",.2f")],
    ).add_params(hover)
    return (alt.layer(line, rule, points)
            .properties(height=height,
                        title=alt.Title(title, subtitle=subtitle)))


def revenue_pie(frame: pd.DataFrame, key: str, order: list[str], *,
                label_floor: float = 5.0, height: int = 400) -> alt.Chart:
    """Share of the window's revenue as a part-to-whole.

    A pie answers exactly one question well -- what fraction of cashflow this
    segment carries -- and answers "which segment is bigger than which" badly,
    because that is an angle comparison. The ranked bars below it do the
    comparing, so the two are complementary rather than duplicates.

    Three things make it readable rather than decorative:

      * Slices are sorted by share, so the clockwise order, the legend order
        and the ramp's lightness order are all the same reading. Identity is
        never carried by a hue the viewer has to match against a swatch.
      * A 2px surface ring between slices, so adjacent fills never touch.
      * Percentages are labelled outside the arc, but only above
        `label_floor` -- below it the labels overlap each other and the ones
        that matter. The small slices read from the legend and the tooltip.

    The small slices are faded with an `opacity` condition rather than removed
    with `transform_filter`: filtering rows out of a layer re-stacks that
    layer, which would silently move every remaining label off its own slice.
    """
    p = palette()
    base = alt.Chart(frame).encode(
        theta=alt.Theta("share_pct:Q", stack=True),
        order=alt.Order("share_pct:Q", sort="descending"),
        color=alt.Color(f"{key}:N", sort=order,
                        scale=alt.Scale(domain=order, range=ramp(len(order))),
                        legend=alt.Legend(title=None, orient="right")),
        tooltip=[alt.Tooltip(f"{key}:N", title=key.replace("_", " ")),
                 alt.Tooltip("share_pct:Q", title="share of revenue",
                             format=".2f"),
                 alt.Tooltip("revenue:Q", title="revenue", format="$,.0f"),
                 alt.Tooltip("customers:Q", title="customers", format=","),
                 alt.Tooltip("rev_per_cust:Q", title="revenue/cust",
                             format="$,.2f")])
    arcs = base.mark_arc(outerRadius=118, stroke=p["surface"], strokeWidth=2)
    labels = base.mark_text(radius=140, fontSize=12).encode(
        text=alt.Text("label:N"),
        # Text wears an ink token, never the slice colour it sits beside.
        color=alt.value(p["ink2"]),
        opacity=alt.condition(alt.datum.share_pct >= label_floor,
                              alt.value(1.0), alt.value(0.0)))
    hidden = int((frame.share_pct < label_floor).sum())
    tail = (f" {hidden} slice{'s' if hidden != 1 else ''} under "
            f"{label_floor:g}% {'are' if hidden != 1 else 'is'} labelled in "
            f"the legend only." if hidden else "")
    return (alt.layer(arcs, labels).properties(
        height=height,
        title=alt.Title(
            "Share of 90-day revenue",
            subtitle=f"${float(frame.revenue.sum()):,.0f} in the window, "
                     f"active customers only.{tail}")))


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

# The toggle sits in the sidebar, not in a tab, because it changes the data
# every tab is drawn from. In the main column it would have to live somewhere,
# and wherever that was would imply it only governed the charts beside it;
# worse, it has to be read before the title caption quotes a total, so putting
# it inline would put a control above the page title. The sidebar renders
# independently of the main column's order, which makes it the only placement
# that is both always visible and correctly scoped.
with st.sidebar:
    st.markdown("### Population")
    EXCLUDE_OUTLIERS = st.toggle(
        "Exclude the 3 volume accounts",
        value=False,
        help="Three accounts place 1.4-2.4 orders every day for over a year "
             "each -- corporate or aggregator traffic rather than people. "
             "Turning this on removes them from every aggregate on the page. "
             "They are never removed from gold.customer_lifetime_value, so "
             "the exclusion stays measurable.")

meta = load("meta").iloc[0]
p = palette()

# Named and quoted whether or not the toggle is on. An exclusion the reader
# cannot see the size of is worse than no exclusion -- and with it off, this
# is the only place the page admits the accounts are in the numbers.
_outliers = load("volume_outliers")
if str(meta.outlier_flag) != "present":
    st.warning(
        f"**The volume-outlier flag is not available**, so the sidebar toggle "
        f"does nothing. `gold.customer_lifetime_value` has no "
        f"`is_volume_outlier` column -- it predates it. Re-run:  `python -m "
        f"python_scripts.silver_to_gold.silver_to_gold --env prod --full "
        f"--tables customer_lifetime_value clv_tier_profile "
        f"clv_tier_segments`  ({meta.outlier_flag})",
        icon=":material/flag:")
elif bool(meta.outliers_excluded):
    st.success(
        f"**{int(meta.outlier_accounts)} volume accounts excluded** from every "
        f"figure below: {int(meta.outlier_orders):,} lifetime orders and "
        f"${meta.outlier_revenue:,.2f}. The trailing-90-day numbers barely "
        f"move (-0.04% of revenue) because two of the three are Slipping and "
        f"the third has been Dormant for 974 days; the lifetime numbers move "
        f"-3.66%. Every tier boundary is unchanged, which is why the "
        f"concentration finding does not depend on this choice.",
        icon=":material/filter_alt_off:")

st.title("Global Partners Sales Analysis")
st.caption(
    f"Due to the stale nature of the demo data an anchor was established using the "
    f"last order date of **{meta.anchor_date}** -- and not the pipeline batch date. \n \n"
    f"Because we need to understand recent cudsomter trends as of that anchor a 90 "
    f"day look-back period was created for in-depth analysis of recent trends at that time"
    f"The total numberof lifetime customers was {int(meta.all_customers):,}of which " 
    f" {int(meta.active_customers):,} were considered active due to purchase activity "
    f"during this time period for analysis."
    )
    

# Silver's last order against the anchor gold was built at. A daily job that
# loads new orders moves the first and not the second until it runs, and every
# segment on this page is stale by the difference. Shown rather than reconciled,
# because reconciling it silently is how a dashboard reports last week.
if int(meta.stale_days) > 0:
    st.error(
        f"**Gold is {int(meta.stale_days)} days behind silver.** The segments "
        f"here were computed at **{meta.gold_anchor_date}** but silver now "
        f"holds orders through **{meta.anchor_date}**. Re-run the "
        f"`gp_silver_to_gold` job; until then every segment, threshold and "
        f"share on this page is as of the older date.",
        icon=":material/sync_problem:")

if str(meta.anchored_tables) != "present":
    st.warning(
        f"**The loyalty and transition tables are not published yet**, so "
        f"the **{TAB_PROGRAM}** tab is empty. Everything else on this page "
        f"reads gold and silver "
        "directly and is unaffected. Run:  `python -m "
        "python_scripts.silver_to_gold.silver_to_gold --env prod --full "
        "--tables customer_segment_transitions loyalty_cohort_outcomes`",
        icon=":material/table_rows:")

# The definitions reference. Collapsed so it does not push the charts below the
# fold, but placed above the tabs rather than inside one because the
# segmentation is the only dimension every tab shares.
with st.expander("**Segment definitions** -- what each of the nine segments "
                 "means, how it is calculated, and how many customers are in "
                 "it"):
    sizes = load("segment_sizes").set_index("segment")
    rows = ["| Segment | Group | Customers | % of base | Defined as |",
            "|---|---|---:|---:|---|"]
    for name, group, text in SEGMENT_DEFS:
        rows.append(
            f"| **{name}** | {group} | {int(sizes.customers.get(name, 0)):,} "
            f"| {float(sizes.pct_of_base.get(name, 0.0)):.1f}% "
            f"| {text.format(cut=meta.live_vip_cut)} |")
    st.markdown("\n".join(rows))

    st.caption(
        "Counts are read live from `gold.customer_rfm`, so a definition and "
        "its headcount cannot drift apart after a pipeline run. All windows "
        "are measured back from the anchor date above, never from today.")
    st.info(
        "**First match wins.** The rules are evaluated in the order Dormant, "
        "New, Re-engaging, VIP, Loyal, Promising, Slipping, Never Returned, "
        "Lapsing, and a customer takes the first rule they satisfy rather "
        "than the best-sounding one. The order is load-bearing in one place: "
        "**New outranks Loyal and Promising**, so a customer three weeks old "
        "with six orders is New. Read it as a statement about tenure, not "
        "about value.",
        icon=":material/low_priority:")

# Beside the segment definitions because it is the same kind of thing: a
# page-wide decision about who is in the numbers, which every tab inherits.
if not _outliers.empty:
    with st.expander(
            f"**The {len(_outliers)} volume accounts** -- who they are, why "
            f"they are flagged, and what excluding them does"):
        rows = ["| Account | Lifetime orders | Orders/day | Lifetime spend "
                "| Avg order | Active span | 90-day segment |",
                "|---|---:|---:|---:|---:|---:|---|"]
        for r in _outliers.itertuples():
            span = (pd.to_datetime(r.last_order_date)
                    - pd.to_datetime(r.first_order_date)).days + 1
            rows.append(
                f"| `{str(r.user_id)[:8]}...` | {int(r.total_orders):,} "
                f"| **{float(r.orders_per_active_day):.2f}** "
                f"| ${float(r.total_lifetime_revenue):,.2f} "
                f"| ${float(r.avg_order_value):.2f} | {span:,} d "
                f"| {r.segment} |")
        st.markdown("\n".join(rows))

        st.caption(
            "Flagged by **lifetime order count**, not by a percentile -- the "
            "threshold itself lives in `clv_history.OUTLIER_ORDER_FLOOR` and "
            "is not repeated here, so the page cannot drift from the rule it "
            "is describing. The ranked counts run 2124, 1803, 1000, then 799, "
            "so any floor between 800 and 1000 isolates exactly these three. "
            "`orders/day` is shown as the evidence, not as the rule: it is "
            "meaningless for a one-order customer, whose span is a single day "
            "and whose rate is therefore exactly 1.00.")
        st.warning(
            "**A percentile rule would be wrong here.** A p99 cut on lifetime "
            "spend flags 202 customers holding 26.7% of revenue, 64 of them "
            "current VIPs -- real best customers excluded for being good. A "
            "1.5x IQR cut flags 2,464 customers holding **67.0%** of revenue: "
            "lifetime spend is severely right-skewed, so IQR treats the whole "
            "healthy right tail as anomalous. Neither is outlier detection on "
            "this distribution.",
            icon=":material/warning:")
        st.info(
            f"Excluding them removes {int(meta.outlier_orders):,} orders and "
            f"${meta.outlier_revenue:,.2f} of lifetime spend (-3.66% of "
            f"revenue, -4.15% of orders) and moves the Top tier's share of "
            f"spend from 76.59% to 75.70%. It leaves every tier boundary "
            f"exactly where it was -- the Top tier still starts at $75.94 and "
            f"the Bottom tier still ends at $9.88 -- and the median lifetime "
            f"spend across the base stays $23.78. (The Top tier's own median "
            f"shifts a cent, $158.24 to $158.23, which is `percentile_approx` "
            f"over three fewer rows.) **The concentration finding survives the "
            f"exclusion** -- which is the reason the toggle is worth having: "
            f"it lets the reader confirm that rather than take it on trust.",
            icon=":material/function:")

st.info(
    f"**Dormant customers are excluded throughout.** All "
    f"{int(meta.all_customers) - int(meta.active_customers):,} of them have "
    f"`orders_last90d = 0` and `spend_last90d = 0` by definition, which is "
    f"what `is_active` exists to express. Lifetime figures, where shown, say "
    f"so.",
    icon=":material/filter_alt:")

# Lifetime value leads. It defines the tiers the 90-day segments are read
# against, and it is where "revenue is repeat orders, not bigger baskets" is
# established over all of history rather than inferred from a single 90-day
# window -- so the value tab now reads as a recent slice of a pattern already
# proved, rather than the pattern being guessed at and confirmed later.
#
# Display order is THIS list, not the order of the `with` blocks below. Those
# stay where they are, which is what makes reordering the page a one-line
# change instead of moving six hundred lines past each other.
# Sales trend sits last because it is the one tab with no customer on it:
# hours, weekdays, stores and menu categories. It answers an operational
# question rather than a retention one, so it reads as a coda rather than as
# an interruption to the three that build on each other.
tab_clv, tab_value, tab_program, tab_trend = st.tabs(
    [TAB_CLV, TAB_VALUE, TAB_PROGRAM, TAB_TREND])


# --- 1. where the value is -----------------------------------------------
#
# Retention is one decision, so the three panels that inform it sit on one
# tab: what separates the segments (frequency, not basket), where the money
# actually is, and who is overdue against their own rhythm. Splitting them
# across tabs made the reader hold three charts in their head to reach a
# conclusion the page could have shown them.

with tab_value:
    frame = load("window_segment")

    # --- what separates the segments -------------------------------------
    st.subheader("All the variance is on the frequency axis")

    span_aov = frame.avg_order.max() / frame.avg_order.min()
    span_freq = frame.orders_per_cust.max() / frame.orders_per_cust.min()
    a, b, c = st.columns(3)
    a.metric("Average order value spread", f"{span_aov:.2f}x",
             f"${frame.avg_order.min():.2f} - ${frame.avg_order.max():.2f}",
             delta_color="off")
    b.metric("Orders per customer spread", f"{span_freq:.1f}x",
             f"{frame.orders_per_cust.min():.2f} - "
             f"{frame.orders_per_cust.max():.2f}", delta_color="off")
    c.metric("Median items per order", f"{int(frame.med_units.median())}",
             "every segment, every period", delta_color="off")

    wide_col, narrow_col = st.columns([3, 2])

    with wide_col:
        scatter = alt.Chart(frame).mark_point(
            filled=True, stroke=p["surface"], strokeWidth=2).encode(
            # Zero-based on purpose. A 1.28x spread on a cropped axis reads as
            # a large one; the point of this chart is that the spread is small.
            x=alt.X("avg_order:Q", title="average order value ($)",
                    scale=alt.Scale(
                        domain=[0, float(frame.avg_order.max()) * 1.15])),
            y=alt.Y("orders_per_cust:Q",
                    title="orders per customer (90 days)",
                    scale=alt.Scale(zero=True, nice=True)),
            size=alt.Size("customers:Q", title="customers",
                          scale=alt.Scale(range=[80, 1200]),
                          legend=alt.Legend(orient="bottom")),
            color=alt.value(p["s1"]),
            tooltip=[alt.Tooltip("segment:N"),
                     alt.Tooltip("customers:Q", format=","),
                     alt.Tooltip("avg_order:Q", title="avg order",
                                 format="$.2f"),
                     alt.Tooltip("med_order:Q", title="median order",
                                 format="$.2f"),
                     alt.Tooltip("orders_per_cust:Q", title="orders/cust",
                                 format=".2f"),
                     alt.Tooltip("rev_per_cust:Q", title="revenue/cust",
                                 format="$.2f")])
        labels = alt.Chart(frame).mark_text(
            align="left", dx=12, dy=-10, color=p["ink2"], fontSize=12).encode(
            x=alt.X("avg_order:Q",
                    scale=alt.Scale(
                        domain=[0, float(frame.avg_order.max()) * 1.15])),
            y=alt.Y("orders_per_cust:Q"), text="segment:N")
        st.altair_chart(styled(alt.layer(scatter, labels).properties(
            height=400,
            title=alt.Title(
                "Basket size against visit frequency",
                subtitle="Bubble area = customers. The x-axis starts at zero, "
                         "so the narrowness of the basket spread is visible."))),
            width="stretch")

    with narrow_col:
        st.altair_chart(styled(ranked_bar(
            frame, "orders_per_cust", "segment",
            title="Orders per customer, 90 days",
            subtitle="The axis that actually separates the segments.",
            axis_title="orders per customer", label_fmt=".2f", height=400)),
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

    st.divider()

    # --- where the revenue sits ------------------------------------------
    st.subheader("Which of those visits the revenue comes from")

    grain = st.radio("Grain", ["segment", "segment_group"], horizontal=True,
                     key="grain_value",
                     help="Applies to the two revenue panels and the period "
                          "breakdown below them. The charts above and below "
                          "are segment-level only.")
    gframe = load(f"window_{grain}")
    order = SEGMENT_ORDER if grain == "segment" else GROUP_ORDER
    order = [o for o in order if o in set(gframe[grain])]

    # Share of the window's revenue, which is the part-to-whole question the
    # ranked bars below cannot answer: a bar says VIP is the tallest, a pie
    # says VIP is nearly two fifths of the money.
    #
    # Recomputed from `revenue` rather than reusing the published
    # `rev_share_pct`, which is rounded per row in gold and sums to 100.01.
    # A tenth of a point is invisible on a bar and is exactly the kind of
    # thing a reader checks on a pie.
    share = gframe[[grain, "revenue", "customers"]].copy()
    share["share_pct"] = 100.0 * share.revenue / share.revenue.sum()
    share["rev_per_cust"] = (share.revenue / share.customers).round(2)
    share = (share.sort_values("share_pct", ascending=False)
             .reset_index(drop=True))
    share["label"] = share.share_pct.map("{:.1f}%".format)
    slice_order = share[grain].tolist()

    pie_col, note_col = st.columns([3, 2])
    with pie_col:
        st.altair_chart(styled(revenue_pie(share, grain, slice_order)),
                        width="stretch")
    with note_col:
        top2 = share.head(2)
        head_pct = 100.0 * top2.customers.sum() / share.customers.sum()
        st.markdown(
            f"**{top2[grain].iloc[0]} and {top2[grain].iloc[1]} together carry "
            f"{top2.share_pct.sum():.1f}% of the window's revenue**, on "
            f"{int(top2.customers.sum()):,} of "
            f"{int(share.customers.sum()):,} active customers "
            f"({head_pct:.1f}% of them).")
        st.markdown(
            f"The slice order is not a ranking of customer quality. "
            f"{top2[grain].iloc[0]} earns its share on "
            f"${top2.rev_per_cust.iloc[0]:,.2f} per customer and "
            f"{top2[grain].iloc[1]} on "
            f"${top2.rev_per_cust.iloc[1]:,.2f} -- the same revenue reached "
            f"two opposite ways, which is what the two panels below separate.")
        st.caption(
            "Share of revenue answers *where the cash is now*. It is a "
            "90-day snapshot and says nothing about whether a segment is "
            "growing, so read it against the cadence panel further down "
            "before acting on it.")

    left, right = st.columns(2)
    with left:
        st.altair_chart(styled(ranked_bar(
            gframe, "revenue", grain,
            title="90-day revenue",
            subtitle="Active customers only. Ordered by value.",
            axis_title="revenue ($)", label_fmt="$,.0f")),
            width="stretch")
    with right:
        st.altair_chart(styled(ranked_bar(
            gframe, "rev_per_cust", grain,
            title="Revenue per customer",
            subtitle="The same 90 days, divided by headcount.",
            axis_title="revenue per customer ($)", label_fmt="$,.2f")),
            width="stretch")

    st.markdown(
        f"**VIP is {gframe.loc[gframe[grain].eq('VIP'), 'rev_share_pct'].sum() if 'VIP' in set(gframe[grain]) else 0:.1f}% "
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


# --- 1c. who is overdue, on the same tab --------------------------------
#
# Re-entering the tab container instead of nesting this inside the block
# above: Streamlit appends to a DeltaGenerator every time it is used, so this
# lands under the revenue panels while staying its own readable section.

with tab_value:
    cadence = load("cadence_segment")
    scored = cadence.dropna(subset=["med_overdue"]).copy()
    no_cadence = cadence[cadence.med_overdue.isna()]

    st.divider()

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


# --- 2. customer lifetime value ------------------------------------------
#
# Lifetime spend cut into fifths by `gold.clv_tier_profile`, crossed against
# the 90-day segment by `gold.clv_tier_segments`. Both read straight off the
# catalog: the tier is a percentile over the whole customer base, so
# recomputing it here would be a second definition of "top fifth", and
# `median_spend_per_order` could not be rebuilt from the other frames at all.

with tab_clv:
    prof = load("clv_profile")

    if prof.empty:
        st.warning(
            "**The lifetime-value tables are not published yet.** Run:  "
            "`python -m python_scripts.silver_to_gold.silver_to_gold --env "
            "prod --full --tables clv_tier_profile clv_tier_segments`",
            icon=":material/table_rows:")
    else:
        tiers = prof.clv_tier.tolist()
        # The table is a three-way partition by construction. Indexed
        # positionally rather than by label so a degenerate batch degrades to
        # a repeated row instead of raising KeyError on the page.
        top, low = prof.iloc[0], prof.iloc[-1]
        mid = prof.iloc[1] if len(prof) > 2 else top

        st.subheader("Lifetime spend, cut into fifths")
        st.caption(
            f"Every customer ranked by **lifetime** revenue and split into "
            f"**Top** (the highest-spending fifth), **Mid** (the middle "
            f"60%) and **Bottom** (the lowest fifth). Lifetime, not 90-day, so "
            f"Dormant customers are included throughout this tab -- spend "
            f"does not go to zero when activity does. "
            f"As of {meta.anchor_date}.")

        a, b, c = st.columns(3)
        a.metric(f"{top.clv_tier} tier share of all lifetime spend",
                 f"{top.pct_of_spend:.1f}%",
                 f"on {top.pct_of_customers:.1f}% of customers",
                 delta_color="off")
        b.metric("Median lifetime spend, Top vs Low",
                 f"{top.median_total_spend / max(low.median_total_spend, 0.01):,.0f}x",
                 f"${top.median_total_spend:,.2f} against "
                 f"${low.median_total_spend:,.2f}", delta_color="off")
        c.metric(f"Average order value, {top.clv_tier} vs {mid.clv_tier}",
                 f"${top.avg_spend_per_order:,.2f}",
                 f"against ${mid.avg_spend_per_order:,.2f} -- the same to "
                 f"the cent", delta_color="off")

        # The requested measures, transposed: six measures across three
        # categories reads as a profile down the page and as noise across it.
        st.markdown("##### The measures, tier by tier")
        rows = [
            ("Customers", "customers", "{:,.0f}"),
            ("Share of all customers", "pct_of_customers", "{:.2f}%"),
            ("Lifetime spend band", None, None),
            ("Total spend", "total_spend", "${:,.2f}"),
            ("Share of all spend", "pct_of_spend", "{:.2f}%"),
            ("Median total spend", "median_total_spend", "${:,.2f}"),
            ("Average total spend", "avg_total_spend", "${:,.2f}"),
            ("Average spend per order", "avg_spend_per_order", "${:,.2f}"),
            ("Median spend per order", "median_spend_per_order", "${:,.2f}"),
            ("Average total orders", "avg_orders", "{:,.2f}"),
            ("Median total orders", "median_orders", "{:,.0f}"),
            ("Customers with one order only", "one_order_pct", "{:.2f}%"),
            ("Median tenure (days)", "median_tenure_days", "{:,.0f}"),
            ("Median days since last order", "median_recency_days",
             "{:,.0f}"),
        ]
        table = {}
        for label, col, fmt in rows:
            if col is None:
                table[label] = [
                    f"${r.spend_floor:,.2f} - ${r.spend_ceiling:,.2f}"
                    for r in prof.itertuples()]
            else:
                table[label] = [fmt.format(v) for v in prof[col]]
        st.dataframe(pd.DataFrame(table, index=tiers).T, width="stretch")
        st.caption(
            "**Spend per order is order grain** -- tier revenue divided by "
            "tier orders, and the median over every order placed by the "
            "tier. It is not the mean of each customer's own average, which "
            "would weight a one-order customer the same as a 200-order one. "
            "`customer_lifetime_value.avg_order_value` holds that other "
            "reading per customer.")

        # Two measures, one unit, so one axis and a rule showing the gap.
        left, right = st.columns(2)
        with left:
            conc = pd.concat([
                prof[["clv_tier", "pct_of_customers"]]
                .rename(columns={"pct_of_customers": "pct"})
                .assign(measure="% of customers"),
                prof[["clv_tier", "pct_of_spend"]]
                .rename(columns={"pct_of_spend": "pct"})
                .assign(measure="% of lifetime spend")])
            st.altair_chart(styled(paired_dots(
                conc, key="clv_tier", value="pct", phase="measure",
                order=tiers,
                phase_order=["% of customers", "% of lifetime spend"],
                title="Headcount against money",
                subtitle="The distance between the dots IS the "
                         "concentration.",
                axis_title="percent", height=240)), width="stretch")
        with right:
            per_order_cmp = pd.concat([
                prof[["clv_tier", "median_spend_per_order"]]
                .rename(columns={"median_spend_per_order": "value"})
                .assign(measure="median order"),
                prof[["clv_tier", "avg_spend_per_order"]]
                .rename(columns={"avg_spend_per_order": "value"})
                .assign(measure="average order")])
            st.altair_chart(styled(paired_dots(
                per_order_cmp, key="clv_tier", value="value",
                phase="measure", order=tiers,
                phase_order=["median order", "average order"],
                title="Spend per order, median against average",
                subtitle="A wide gap is skew: a few large orders pulling "
                         "the mean.",
                axis_title="dollars per order", height=240)),
                width="stretch")

        # The two tiers' average order is $13.84 to the cent on the full base,
        # which is a striking way to put the finding -- but it stops being
        # literally true the moment the volume accounts are excluded ($13.95
        # against $13.84), and a page that asserts an equality the table beside
        # it contradicts is worse than one that states two numbers. So the
        # claim is checked against the data it is about rather than written in.
        _basket_gap = abs(float(top.avg_spend_per_order)
                          - float(mid.avg_spend_per_order))
        if _basket_gap < 0.005:
            _basket = (f"the average order is "
                       f"${top.avg_spend_per_order:,.2f} in **both** tiers, "
                       f"the same to the cent")
        else:
            _basket = (f"the average order is ${top.avg_spend_per_order:,.2f} "
                       f"against ${mid.avg_spend_per_order:,.2f}, a gap of "
                       f"{100 * _basket_gap / max(float(mid.avg_spend_per_order), 0.01):,.1f}% "
                       f"against a {top.avg_orders / max(mid.avg_orders, 0.01):,.0f}x "
                       f"gap in order count")

        st.success(
            f"**Lifetime value is order count. Basket size barely moves.** "
            f"{top.clv_tier} averages {top.avg_orders:,.2f} lifetime orders "
            f"against {mid.clv_tier}'s {mid.avg_orders:,.2f}, a "
            f"{top.avg_orders / max(mid.avg_orders, 0.01):,.1f}x spread -- "
            f"while {_basket}. {top.clv_tier}'s median order is actually the "
            f"smaller of the two, ${top.median_spend_per_order:,.2f} against "
            f"${mid.median_spend_per_order:,.2f}, so what the top tier has "
            f"is a longer right tail of large orders on top of far more of "
            f"them. The 90-day window reaches the same conclusion under "
            f"**{TAB_VALUE}**; this is that conclusion over all of history.",
            icon=":material/insights:")
        st.caption(
            f"One figure above is definitional, not behavioural: the "
            f"{low.clv_tier} tier's ${low.median_spend_per_order:,.2f} median "
            f"order is bounded by the tier's own ceiling. A customer whose "
            f"whole lifetime spend is at most ${low.spend_ceiling:,.2f} and "
            f"who ordered once ({low.one_order_pct:.1f}% of the tier did) "
            f"cannot have placed a bigger order than that. So "
            f"{top.clv_tier}-against-{mid.clv_tier} is the comparison that "
            f"answers the basket question; "
            f"{top.clv_tier}-against-{low.clv_tier} partly restates the cut.")

        st.warning(
            f"**The tiers are not 20/60/20, and that is correct.** They "
            f"realise {'/'.join(f'{v:.2f}' for v in prof.pct_of_customers)}. "
            f"Customers who spent an identical amount must land in the same "
            f"tier, so `percent_rank` keeps ties together -- and 475 "
            f"customers have lifetime spend of exactly $9.89, sharing a "
            f"percentile of 0.796 just inside Mid. Forcing exact fifths with "
            f"`ntile(5)` would split those 475 at an arbitrary point. The "
            f"realised shares are published as a column rather than assumed.",
            icon=":material/balance:")

        st.info(
            f"**Age does not explain the tiers, which is worth checking "
            f"before trusting any of this.** Lifetime spend accumulates with "
            f"observation time, so tenure should rank with it. It does not: "
            f"median tenure is "
            f"{', '.join(f'{r.clv_tier} {r.median_tenure_days:,.0f}d' for r in prof.itertuples())}"
            f" -- {low.clv_tier} is **older** than {mid.clv_tier}. These are "
            f"not customers who have not had time to spend. The "
            f"{low.clv_tier} tier has had {low.median_tenure_days:,.0f} days "
            f"at the median and placed {low.median_orders:,.0f} order. "
            f"Median days since the last order runs "
            f"{', '.join(f'{r.clv_tier} {r.median_recency_days:,.0f}d' for r in prof.itertuples())}, "
            f"which does rank with the tier -- so the tiers track whether a "
            f"customer is still ordering, not how long they have been "
            f"around.",
            icon=":material/schedule:")

        # --- what puts a customer in a tier ------------------------------
        segs = load("clv_segments")
        if not segs.empty:
            st.divider()
            st.markdown("##### What each tier is made of")

            view = st.radio(
                "Read the matrix", ["% of tier", "% of segment"],
                horizontal=True, key="clv_matrix_view",
                help="'% of tier' reads across a row and sums to 100 within "
                     "a tier: what the Top tier is composed of. '% of "
                     "segment' reads down a column and sums to 100 within a "
                     "segment: what share of VIPs have reached the Top "
                     "tier.")
            col = ("pct_of_tier" if view == "% of tier"
                   else "pct_of_segment")
            seg_cols = [s for s in SEGMENT_ORDER + ["Dormant"]
                        if s in set(segs.segment)]
            # The matrix IS this section's evidence, so it leads the section
            # and it leads as numbers. The two callouts below quote single
            # cells to the decimal -- 58.7% of the Top tier, 49.1% of its
            # spend -- and a colour ramp can only gesture at a value a
            # reader then has to hunt for in a table anyway. At three tiers
            # by eight segments the grid is small enough to read directly.
            wide = (segs.pivot_table(index="clv_tier", columns="segment",
                                     values=col, aggfunc="first")
                    .reindex(index=tiers, columns=seg_cols).fillna(0))

            # Shaded along whichever direction sums to 100, so the colour
            # and the percentage agree on what is being compared.
            st.dataframe(shaded(wide, axis=1 if col == "pct_of_tier" else 0),
                         width="stretch")
            st.caption(
                f"`{col}`. Rows are lifetime spend tiers, columns are the "
                f"current 90-day segment. "
                + ("Row percentages: each tier's customers split across "
                   "their current 90-day segment, so each row sums to 100 "
                   "and each row is shaded against its own largest cell."
                   if col == "pct_of_tier"
                   else "Column percentages: each segment split across "
                        "lifetime spend tiers, so each column sums to 100 "
                        "and each column is shaded against its own largest "
                        "cell."))
            dormant_top = (segs[segs.clv_tier.eq(tiers[0])
                                & segs.segment.eq("Dormant")])
            core_all = segs[segs.segment.isin(["VIP", "Loyal"])]
            core = core_all[core_all.clv_tier.eq(tiers[0])]
            # Recombined from counts, not averaged from pct_of_segment: the
            # mean of two percentages over different base sizes is not a
            # percentage of anything.
            core_reach = (100.0 * core.customers.sum()
                          / max(int(core_all.customers.sum()), 1))
            st.error(
                f"**{dormant_top.pct_of_tier.sum():.1f}% of the Top tier is "
                f"currently Dormant** -- "
                f"{int(dormant_top.customers.sum()):,} customers who are "
                f"among the highest lifetime spenders in the book and have "
                f"not ordered in 90 days, holding "
                f"${dormant_top.total_spend.sum():,.0f} of lifetime spend "
                f"between them. That cell is the reason this table crosses a "
                f"lifetime measure against a recent one: neither the 90-day "
                f"segments nor the lifetime tiers can show it alone, and it "
                f"is a larger recoverable sum than anything under "
                f"**{TAB_VALUE}**.",
                icon=":material/priority_high:")
            st.markdown(
                f"Read the other way: VIP and Loyal together are "
                f"{core.pct_of_tier.sum():.1f}% of the Top tier by headcount "
                f"but {core.pct_of_tier_spend.sum():.1f}% of its spend, and "
                f"{core_reach:.1f}% of all Core customers have "
                f"reached the Top tier. The segments and the "
                f"tiers agree at the extremes and disagree in the middle, "
                f"which is where the money that can still be moved sits.")

    st.divider()

# --- 2b. the second order, still on the lifetime-value tab ---------------

with tab_clv:
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


# --- 3. is the program working -------------------------------------------

with tab_program:
    st.subheader("Two readings of the loyalty program, and they disagree in scale")

    trans = load("transitions")
    if trans.empty:
        st.info(
            "Nothing to show until `customer_segment_transitions` and "
            "`loyalty_cohort_outcomes` are published. See the banner at the "
            "top of the page for the command.",
            icon=":material/hourglass_empty:")
        st.stop()

    # Said plainly rather than left to be assumed. These two tables arrive
    # pre-aggregated at segment grain and carry no outlier variant, so there
    # is nothing for the sidebar filter to act on -- a published percentage
    # cannot be un-aggregated. Extending the variant to them is a gold change,
    # not a dashboard one.
    if bool(meta.outliers_excluded):
        st.warning(
            f"**This tab still includes the {int(meta.outlier_accounts)} "
            f"volume accounts**, unlike the rest of the page. The era and "
            f"cohort tables are published pre-aggregated at segment grain, so "
            f"they carry no outlier variant for the sidebar toggle to select. "
            f"The effect is small -- the three accounts are 0.04% of "
            f"trailing-90-day revenue -- but it is a real inconsistency "
            f"between this tab and the others while the toggle is on.",
            icon=":material/filter_alt:")

    # New Growth and Never Returned are structurally unreachable as
    # destinations over 180 days, so they are zero in every row. Showing a
    # column of zeros invites the reading that acquisition stopped.
    trans = trans[~trans.grp_t1.isin(["New Growth", "Never Returned"])]
    # Pre-launch is anchored to the program's first flagged order and never
    # moves; Recent and Prior roll with the data. The paired panels compare the
    # two fixed-against-latest ends, and Prior is the control that keeps the
    # comparison honest -- see the caveat below them.
    era_order = ["Pre-launch", "Recent"]
    all_eras = [e for e in ["Pre-launch", "Prior", "Recent"]
                if e in set(trans.era)]
    seg_order = [s for s in SEGMENT_ORDER + ["Dormant"]
                 if s in set(trans.seg_t0)]

    st.markdown("##### 1. Calendar comparison: two 180-day spans, one each side of launch")
    left, right = st.columns(2)
    for col, grp, label, note in (
            (left, "Core", "Reached Core 180 days later", "higher is better"),
            (right, "Dormant", "Went Dormant 180 days later",
             "LOWER is better -- rightward is worse on this panel")):
        sub = trans[trans.grp_t1.eq(grp)
                    & trans.era.isin(era_order)][["era", "seg_t0", "row_pct"]]
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

    # A third window, entirely inside the post-launch period, as a control. If
    # progression to Core were improving with the program, the latest window
    # should be the best one. It is not, and showing that is the difference
    # between a dashboard and a sales deck.
    if "Prior" in all_eras:
        core = (trans[trans.grp_t1.eq("Core")]
                .pivot_table(index="seg_t0", columns="era", values="row_pct",
                             aggfunc="first")
                .reindex(columns=all_eras))
        if core.notna().all(axis=None):
            peak = core.idxmax(axis=1)
            n_prior = int((peak == "Prior").sum())
            st.error(
                f"**Progression to Core is not improving monotonically, and "
                f"the middle window is the best one.** Of "
                f"{len(core)} starting segments, **{n_prior} peak in the "
                f"*Prior* window** (Feb-Aug 2023) rather than the latest one "
                f"-- Slipping reaches Core 7.0% of the time before launch, "
                f"13.7% in Prior, then falls back to 9.0% in Recent. So the "
                f"pre/post gap above is a difference between two points, not "
                f"a trend, and the honest claim is that the program coincided "
                f"with a lift that has since partly reversed.",
                icon=":material/dangerous:")
            st.dataframe(core.reindex(seg_order).round(1), width="stretch")
            st.caption(
                "Share of each starting segment that reached Core 180 days "
                "later, by window. `Pre-launch` is anchored to the program's "
                "first flagged order and never moves; `Prior` and `Recent` "
                "roll with the data.")

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
                .reindex(all_eras, level=0).fillna(0).round(1))
        st.dataframe(wide, width="stretch")
        st.caption(
            "Row percentages. `New Growth` and `Never Returned` are omitted: "
            "New requires tenure <= 30 days, which nobody present at the start "
            "can satisfy 180 days later, and Never Returned requires a single "
            "lifetime order, which would place the customer past 180 days of "
            "recency and so into Dormant first. Both are structurally zero.")


# --- 4. sales trend monitoring -------------------------------------------
#
# One decision: when and where to put people and product. Nothing on this tab
# is a customer -- it is hours, weekdays, stores and menu categories -- which
# is why it is a tab of its own rather than a section inside one of the
# others.
#
# Reads `gold.sales_trend_grid`, the only gold table carrying a clock, and the
# only one on local time.

with tab_trend:
    grid = load("sales_trend")

    if grid.empty:
        st.warning(
            f"**The sales-trend table is not published yet.** Run:  `python "
            f"-m python_scripts.silver_to_gold.silver_to_gold --env prod "
            f"--full --tables sales_trend_grid`  ({meta.sales_trend_table})",
            icon=":material/table_rows:")
    else:
        st.subheader("When the demand lands, and where")
        st.caption(
            f"One row per store, per menu category, per hour of the **local** "
            f"day. Every clock time on this tab is America/New_York: the "
            f"pipeline stores UTC and `sales_trend_grid` is the one table "
            f"that converts, because read in UTC the 11:00 revenue peak "
            f"lands at 16:00 and every staffing conclusion drawn from it "
            f"would be five hours out. Through {meta.anchor_date}.")

        _dev_cells = int(grid.is_development.sum())
        _dev_rev = float(grid[grid.is_development].total_revenue.sum())
        keep_dev = st.toggle(
            "Put the development store back in", value=False,
            key="trend_dev",
            help="One restaurant id carries the entire 'Alltown Fresh - "
                 "DEVELOPMENT' app. It is test traffic whose hours are test "
                 "hours, and it is left out of this tab by default.")
        if not keep_dev:
            grid = grid[~grid.is_development].copy()

        if _dev_cells and not keep_dev:
            st.info(
                f"**The development store is excluded here, and only here.** "
                f"{_dev_cells:,} cells, ${_dev_rev:,.2f}, 0.41% of revenue. "
                f"It matters more than its size: because the test traffic is "
                f"one entire store rather than rows scattered through real "
                f"ones, it would appear below as the single location of 28 "
                f"that trades overnight -- it peaks at 01:00 and takes most "
                f"of its revenue before 06:00. The rows are still in "
                f"`gold.sales_trend_grid` and in every other table in the "
                f"warehouse, so this tab sits ${_dev_rev:,.2f} below the "
                f"rest of the page on purpose rather than by accident.",
                icon=":material/science:")

        if bool(meta.outliers_excluded):
            st.caption(
                f"The sidebar exclusion **is** applied to this tab. It works "
                f"by subtraction rather than by filtering: every measure in "
                f"`sales_trend_grid` is additive and each cell publishes its "
                f"own flagged-account portion, so the excluded reading is "
                f"`measure - outlier_measure` at any rollup. There is no "
                f"customer column on this table to filter on.")

        _rev = float(grid.total_revenue.sum())
        _orders = int(grid.orders.sum())
        by_hour = (grid.groupby("hour_local", as_index=False)
                   .agg(revenue=("total_revenue", "sum"),
                        orders=("orders", "sum")))
        _peak = by_hour.loc[by_hour.revenue.idxmax()]

        def _per_day(weekend: bool) -> float:
            sub = grid[grid.is_weekend.fillna(False).astype(bool) == weekend]
            return (float(sub.total_revenue.sum())
                    / max(sub.sales_date_local.nunique(), 1))

        _weekday, _weekend = _per_day(False), _per_day(True)
        _evening = 100.0 * float(
            grid[grid.hour_local >= 18].total_revenue.sum()) / max(_rev, 0.01)

        a, b, c, d = st.columns(4)
        a.metric("Revenue on this tab", f"${_rev:,.0f}",
                 f"{_orders:,} orders", delta_color="off")
        b.metric("Busiest hour", f"{int(_peak.hour_local):02d}:00",
                 f"{100 * _peak.revenue / _rev:.1f}% of revenue",
                 delta_color="off")
        c.metric("Weekday average", f"${_weekday:,.0f}/day",
                 f"{100 * (_weekday - _weekend) / max(_weekend, 0.01):+.0f}% "
                 f"vs weekend")
        d.metric("After 18:00", f"{_evening:.1f}%", "of all revenue",
                 delta_color="off")

        st.divider()

        # --- the trend itself --------------------------------------------
        st.markdown("##### Revenue over time")

        gran = st.radio(
            "Granularity", ["Month", "Week", "Day"], horizontal=True,
            key="trend_gran",
            help="Month and Week drop their incomplete first and last "
                 "periods. Day does not need to: a day is a whole day.")
        _period_col, _complete_col = {
            "Month": ("month_start", "month_complete"),
            "Week": ("week_start", "week_complete"),
            "Day": ("sales_date_local", None),
        }[gran]

        series = (grid.groupby(_period_col, as_index=False)
                  .agg(revenue=("total_revenue", "sum"),
                       orders=("orders", "sum")))
        _dropped: list[str] = []
        if _complete_col:
            _ok = grid.groupby(_period_col)[_complete_col].max()
            _dropped = [str(k) for k, good in _ok.items() if not good]
            series = series[series[_period_col].isin(
                [k for k, good in _ok.items() if good])]
        series[_period_col] = pd.to_datetime(series[_period_col])

        st.altair_chart(styled(trend_line(
            series, x=_period_col, y="revenue",
            title=f"Revenue by {gran.lower()}",
            subtitle=("Local calendar periods. Hover for the figure; the "
                      "line is one series, so there is no legend to read."),
            y_title="revenue")), width="stretch")

        if _dropped:
            st.caption(
                f"**{len(_dropped)} incomplete {gran.lower()}"
                f"{'s' if len(_dropped) > 1 else ''} dropped** rather than "
                f"plotted: {', '.join(_dropped)}. Silver ends "
                f"{meta.anchor_date} and begins mid-period, so the first and "
                f"last buckets hold fewer days than the rest. Drawn "
                f"unmarked they read as a collapse in demand, and the "
                f"collapse is the calendar.")

        # The slope is worth stating rather than leaving to the eye, because
        # the eye will read the dropped partial period as part of the shape.
        if len(series) >= 6:
            _first3 = float(series.sort_values(_period_col).revenue.head(3).mean())
            _last3 = float(series.sort_values(_period_col).revenue.tail(3).mean())
            _slope = 100.0 * (_last3 - _first3) / max(_first3, 0.01)
            st.markdown(
                f"Across complete {gran.lower()}s the first three average "
                f"${_first3:,.0f} and the last three ${_last3:,.0f}, "
                f"**{_slope:+.1f}%**. Read it as a trend only at Month: at "
                f"Day and Week the series is dominated by the weekday cycle "
                f"below, not by anything moving over the period.")

        st.divider()

        # --- hour of day --------------------------------------------------
        st.markdown("##### The shape of a day")

        by_hour["hour_label"] = by_hour.hour_local.map(lambda h: f"{h:02d}")
        _hour_order = [f"{h:02d}" for h in range(24)]
        # Selective labels, not one per bar: three labels name the peak, and
        # twenty-four would be a second axis made of text.
        _top3 = set(by_hour.nlargest(3, "revenue").hour_local)
        by_hour["peak_label"] = [
            f"${r.revenue:,.0f}" if r.hour_local in _top3 else ""
            for r in by_hour.itertuples()]

        _p = palette()
        _hour_base = alt.Chart(by_hour).encode(
            x=alt.X("hour_label:N", sort=_hour_order,
                    title="hour of the local day"),
            y=alt.Y("revenue:Q", title="revenue"))
        st.altair_chart(styled(alt.layer(
            _hour_base.mark_bar(color=_p["s1"], cornerRadiusTopLeft=4,
                                cornerRadiusTopRight=4,
                                stroke=_p["surface"], strokeWidth=2)
            .encode(tooltip=[
                alt.Tooltip("hour_label:N", title="hour"),
                alt.Tooltip("revenue:Q", title="revenue", format="$,.2f"),
                alt.Tooltip("orders:Q", title="orders", format=",.0f")]),
            _hour_base.mark_text(dy=-8, color=_p["ink2"], fontSize=11)
            .encode(text=alt.Text("peak_label:N")),
        ).properties(height=300, title=alt.Title(
            "Revenue by hour of the local day",
            subtitle="One hue: the hour is already named on the axis."))),
            width="stretch")

        _dayparts = [d for d in warehouse.DAYPART_ORDER
                     if d in set(grid.daypart)]
        _dp = (grid.groupby("daypart").total_revenue.sum()
               .reindex(_dayparts).fillna(0.0))
        _dp_pct = 100.0 * _dp / max(float(_dp.sum()), 0.01)
        st.markdown(
            "Grouped into the buckets a shift is actually planned in: "
            + ", ".join(f"**{name}** {_dp_pct[name]:.1f}%"
                        for name in _dayparts)
            + ". The two busiest clock hours together are "
            + f"{100 * float(by_hour.nlargest(2, 'revenue').revenue.sum()) / max(_rev, 0.01):.1f}%"
            + " of everything this tab counts.")

        st.divider()

        # --- the staffing grid -------------------------------------------
        st.markdown("##### Which hours of which days")
        st.caption(
            "Revenue per **open day**, not total revenue, so a Monday is "
            "comparable with a Saturday despite there being more Mondays in "
            "the data. Shaded against the largest cell in the whole grid "
            "rather than within each row: the question is where the busiest "
            "hour of the week is, and per-row shading would give every "
            "weekday its own private scale.")

        _cells = (grid.groupby(["dow_number", "day_of_week", "hour_local"],
                               as_index=False)
                  .agg(revenue=("total_revenue", "sum")))
        _open = (grid.groupby(["dow_number", "day_of_week"], as_index=False)
                 .agg(days=("sales_date_local", "nunique")))
        _cells = _cells.merge(_open, on=["dow_number", "day_of_week"])
        _cells["per_day"] = _cells.revenue / _cells.days.clip(lower=1)

        _dow_labels = (_open.sort_values("dow_number").day_of_week.tolist())
        _week_grid = (_cells.pivot_table(index="day_of_week",
                                         columns="hour_local",
                                         values="per_day", aggfunc="first")
                      .reindex(index=_dow_labels,
                               columns=list(range(24))).fillna(0.0))
        _week_grid.columns = [f"{h:02d}" for h in _week_grid.columns]
        st.dataframe(shaded(_week_grid, axis=None, fmt="${:,.0f}"),
                     width="stretch")

        _hot = _cells.loc[_cells.per_day.idxmax()]
        # What that day would look like with no peak at all, as the thing the
        # peak is worth comparing against: a daily total tells a planner
        # nothing about which shift needs the people.
        _hot_day_total = float(
            _cells[_cells.day_of_week.eq(_hot.day_of_week)].per_day.sum())
        _flat_hour = _hot_day_total / 24.0
        st.success(
            f"**The week peaks on {_hot.day_of_week} at "
            f"{int(_hot.hour_local):02d}:00, at ${_hot.per_day:,.2f} per "
            f"open day.** Weekdays run ${_weekday:,.0f} a day against "
            f"${_weekend:,.0f} at weekends, so the busiest shift of the week "
            f"is a weekday midday one. This is the answer a daily total "
            f"cannot give: spread evenly across 24 hours that same "
            f"{_hot.day_of_week} would be ${_flat_hour:,.2f} an hour, and "
            f"the peak hour is "
            f"{_hot.per_day / max(_flat_hour, 0.01):,.1f}x that. Demand is "
            f"concentrated enough that staffing to the average misses it "
            f"twice over -- short at the peak, idle overnight.",
            icon=":material/schedule:")

        st.divider()

        # --- what to prep, and when --------------------------------------
        st.markdown("##### What sells, and when it sells")

        _by_cat = (grid.groupby("item_category").total_revenue.sum()
                   .sort_values(ascending=False))
        # Eight plus Other, matching the project's existing cut: the category
        # tables use a 1%-of-all-time-sales threshold, which keeps eight
        # categories, and eight is also the ceiling on distinguishable
        # categorical steps. The tail is grouped rather than dropped so each
        # column still sums to 100.
        _keep = list(_by_cat.head(8).index)
        _tail = [c for c in _by_cat.index if c not in _keep]
        _mix = grid.assign(category=grid.item_category.where(
            grid.item_category.isin(_keep), "Other"))
        _mix = (_mix.groupby(["category", "daypart"], as_index=False)
                .agg(revenue=("total_revenue", "sum")))
        _mix["pct"] = 100.0 * _mix.revenue / _mix.groupby(
            "daypart").revenue.transform("sum")
        _cat_order = _keep + (["Other"] if _tail else [])
        _cat_grid = (_mix.pivot_table(index="category", columns="daypart",
                                      values="pct", aggfunc="first")
                     .reindex(index=_cat_order, columns=_dayparts).fillna(0.0))
        st.dataframe(shaded(_cat_grid, axis=0, fmt="{:.1f}"),
                     width="stretch")
        st.caption(
            f"Percent of each daypart's revenue, so every **column** sums to "
            f"100 and the shading runs down the columns for that reason. The "
            f"{len(_tail)} categories below the top eight are grouped as "
            f"`Other` ({_by_cat.tail(len(_tail)).sum() / max(float(_by_cat.sum()), 0.01) * 100:.1f}% "
            f"of revenue) rather than dropped, so the columns still total.")

        if {"Morning", "Midday"} <= set(_cat_grid.columns):
            _am = _cat_grid["Morning"].idxmax()
            _noon = _cat_grid["Midday"].idxmax()
            if _am != _noon:
                st.info(
                    f"**The menu rotates through the day, so one prep plan "
                    f"will not fit it.** `{_am}` is "
                    f"{_cat_grid.loc[_am, 'Morning']:.0f}% of Morning revenue "
                    f"and `{_noon}` is {_cat_grid.loc[_noon, 'Midday']:.0f}% "
                    f"of Midday. Read down a column, not across a row: the "
                    f"columns are shares of very different totals, and "
                    f"Midday alone is "
                    f"{_dp_pct.get('Midday', 0):.0f}% of the day.",
                    icon=":material/restaurant:")

        st.divider()

        # --- by location --------------------------------------------------
        st.markdown("##### Which stores, and whether they agree")

        _stores = (grid.groupby("restaurant_id", as_index=False)
                   .agg(revenue=("total_revenue", "sum"),
                        orders=("orders", "sum"),
                        open_days=("sales_date_local", "nunique"),
                        first_sale=("sales_date_local", "min"),
                        last_sale=("sales_date_local", "max")))
        _store_hour = (grid.groupby(["restaurant_id", "hour_local"],
                                    as_index=False)
                       .agg(revenue=("total_revenue", "sum"))
                       .sort_values("revenue", ascending=False)
                       .drop_duplicates("restaurant_id"))
        _stores = _stores.merge(
            _store_hour[["restaurant_id", "hour_local"]].rename(
                columns={"hour_local": "peak_hour"}),
            on="restaurant_id", how="left")
        _stores["per_open_day"] = _stores.revenue / _stores.open_days.clip(lower=1)
        _stores["pct_of_revenue"] = 100.0 * _stores.revenue / max(_rev, 0.01)
        _stores = _stores.sort_values("revenue", ascending=False)

        _peak_spread = _stores.peak_hour.value_counts().sort_index()
        st.error(
            f"**The {len(_stores)} stores do not share a peak hour, so a "
            f"chain-wide shift pattern is wrong for most of them.** Their "
            f"individual peaks fall across "
            f"{len(_peak_spread)} different hours: "
            + ", ".join(f"{int(h):02d}:00 ({int(n)})"
                        for h, n in _peak_spread.items())
            + f". Only {int(_peak_spread.max())} stores share the most "
            f"common one. The top store is "
            f"{_stores.pct_of_revenue.iloc[0]:.1f}% of revenue on this tab "
            f"and the top five are "
            f"{_stores.pct_of_revenue.head(5).sum():.1f}%, so a single "
            f"rollup hour is largely the biggest store's hour.",
            icon=":material/storefront:")

        st.dataframe(
            _stores[["restaurant_id", "revenue", "per_open_day", "orders",
                     "peak_hour", "open_days", "first_sale", "last_sale",
                     "pct_of_revenue"]]
            .rename(columns={"per_open_day": "revenue_per_open_day"}),
            width="stretch", hide_index=True,
            column_config={
                "revenue": st.column_config.NumberColumn(format="$%.2f"),
                "revenue_per_open_day": st.column_config.NumberColumn(
                    format="$%.2f"),
                "pct_of_revenue": st.column_config.NumberColumn(
                    format="%.2f%%"),
            })
        st.caption(
            f"`revenue_per_open_day` is the comparable column, not `revenue`: "
            f"{int((_stores.first_sale > _stores.first_sale.min()).sum())} of "
            f"{len(_stores)} stores have a first sale later than the "
            f"earliest, and "
            f"{int((_stores.open_days < 30).sum())} traded on fewer than 30 "
            f"days in total, so a lifetime total ranks length of service as "
            f"much as it ranks demand. `peak_hour` is that store's own "
            f"highest-revenue hour and is thin for the smallest stores -- at "
            f"a few hundred dollars across a handful of days it is close to "
            f"noise, which is the other reason the callout above counts "
            f"stores rather than weighting them.")
