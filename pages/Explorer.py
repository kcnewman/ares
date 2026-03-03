"""
pages/Explorer.py — Market Explorer
Inline filter form → summary metrics → tabbed output (Overview / Segments / Listings).
"""

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils import (
    AMENITY_LABELS,
    BAR_COLOR,
    BAR_DIM,
    CHART_CFG,
    GRID_COLOR,
    PAGE_HOME,
    PAGE_PREDICTOR,
    PLOTLY_LAYOUT,
    RED,
    inject_styles,
    load_market_data,
    metric_bar_html,
    page_note,
    section_heading,
)

st.set_page_config(
    page_title="ARES · Market Explorer",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="collapsed",
)
inject_styles()

# ── Nav bar ───────────────────────────────────────────────────────────────────
nc1, nc2 = st.columns(2, gap="small")
with nc1:
    if st.button("← Home", key="ex_home", use_container_width=True):
        st.switch_page(PAGE_HOME)
with nc2:
    if st.button("Predictor →", key="ex_pred", use_container_width=True):
        st.switch_page(PAGE_PREDICTOR)

st.markdown("## Market Explorer")
st.markdown(
    "<p style='color:var(--t2);font-size:0.9rem;margin-top:-0.4rem;margin-bottom:0;'>"
    "Filter the dataset and explore rental price distributions across Greater Accra."
    "</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

# ── Load data ─────────────────────────────────────────────────────────────────
df_full = load_market_data()

if df_full is None:
    st.error(
        "Market data unavailable. "
        "Set the `DATA_PATH` environment variable to point to `preprocessed_train.csv`."
    )
    st.stop()

# ── Derive filter option lists ────────────────────────────────────────────────
ALL = "All"

loc_opts = sorted(df_full["loc"].dropna().unique().tolist())
type_opts = sorted(df_full["house_type"].dropna().unique().tolist())
cond_opts = sorted(df_full["condition"].dropna().unique().tolist())
furn_opts = sorted(df_full["furnishing"].dropna().unique().tolist())

price_min_global = int(df_full["price"].min())
price_max_global = int(
    df_full["price"].quantile(0.995)
)  # clip extreme outliers for slider
beds_max = int(df_full["bedrooms"].max())
baths_max = int(df_full["bathrooms"].max())

amenity_options = [(k, v) for k, v in AMENITY_LABELS.items() if k in df_full.columns]

# ── Filter form ───────────────────────────────────────────────────────────────
st.markdown('<div class="filter-panel">', unsafe_allow_html=True)
section_heading("Filters")

with st.form("explorer_filters", border=False):
    r1c1, r1c2, r1c3, r1c4 = st.columns(4, gap="small")
    with r1c1:
        f_loc = st.selectbox("Location", [ALL] + loc_opts)
    with r1c2:
        f_type = st.selectbox("Property Type", [ALL] + type_opts)
    with r1c3:
        f_cond = st.selectbox("Condition", [ALL] + cond_opts)
    with r1c4:
        f_furn = st.selectbox("Furnishing", [ALL] + furn_opts)

    r2c1, r2c2, r2c3 = st.columns(3, gap="small")
    with r2c1:
        f_beds = st.slider("Bedrooms", 0, beds_max, (0, beds_max))
    with r2c2:
        f_baths = st.slider("Bathrooms", 0, baths_max, (0, baths_max))
    with r2c3:
        f_price = st.slider(
            "Price Range (₵/mo)",
            price_min_global,
            price_max_global,
            (price_min_global, price_max_global),
        )

    f_amenities = st.multiselect(
        "Required Amenities",
        options=[k for k, _ in amenity_options],
        format_func=lambda k: AMENITY_LABELS.get(k, k),
        placeholder="Select amenities to require…",
    )

    apply_col, _ = st.columns([1, 4])
    with apply_col:
        apply = st.form_submit_button("Apply Filters", use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

# ── Apply filtering ───────────────────────────────────────────────────────────
df = df_full.copy()

if f_loc != ALL:
    df = df[df["loc"] == f_loc]
if f_type != ALL:
    df = df[df["house_type"] == f_type]
if f_cond != ALL:
    df = df[df["condition"] == f_cond]
if f_furn != ALL:
    df = df[df["furnishing"] == f_furn]

df = df[(df["bedrooms"] >= f_beds[0]) & (df["bedrooms"] <= f_beds[1])]
df = df[(df["bathrooms"] >= f_baths[0]) & (df["bathrooms"] <= f_baths[1])]
df = df[(df["price"] >= f_price[0]) & (df["price"] <= f_price[1])]

for am in f_amenities:
    if am in df.columns:
        df = df[df[am] == 1]

n_filtered = len(df)

# ── Summary metric bar ────────────────────────────────────────────────────────
if n_filtered == 0:
    st.warning("No listings match the current filters. Try relaxing your criteria.")
    st.stop()

q25 = df["price"].quantile(0.25)
q50 = df["price"].median()
q75 = df["price"].quantile(0.75)
iqr = q75 - q25
std = df["price"].std()

st.markdown(
    metric_bar_html(
        [
            ("Listings", f"{n_filtered:,}"),
            ("Median Rent", f"₵{q50:,.0f}"),
            ("IQR", f"₵{q25:,.0f} – ₵{q75:,.0f}"),
            ("Std. Dev.", f"₵{std:,.0f}"),
        ]
    ),
    unsafe_allow_html=True,
)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_ov, tab_seg, tab_list = st.tabs(["Overview", "Segments", "Listings"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 · OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
with tab_ov:
    # Price distribution with outlier fences
    section_heading("Price Distribution")
    page_note(f"{n_filtered:,} listings · IQR fences shown as dashed lines.")

    # Clip to 99th pct for readability
    p99 = df["price"].quantile(0.99)
    hist_df = df[df["price"] <= p99]

    fence_lo = max(q25 - 1.5 * iqr, df["price"].min())
    fence_hi = min(q75 + 1.5 * iqr, p99)

    fig1 = px.histogram(
        hist_df,
        x="price",
        nbins=35,
        labels={"price": "Monthly Rent (₵)"},
        color_discrete_sequence=[BAR_COLOR],
    )
    fig1.add_vline(
        x=fence_lo,
        line_dash="dash",
        line_color=BAR_DIM,
        line_width=1.5,
        annotation_text="Lower fence",
        annotation_font_size=10,
        annotation_font_color=BAR_DIM,
        annotation_position="top left",
    )
    fig1.add_vline(
        x=fence_hi,
        line_dash="dash",
        line_color=BAR_DIM,
        line_width=1.5,
        annotation_text="Upper fence",
        annotation_font_size=10,
        annotation_font_color=BAR_DIM,
        annotation_position="top right",
    )
    fig1.add_vline(
        x=q50,
        line_dash="dot",
        line_color=RED,
        line_width=1.5,
        annotation_text="Median",
        annotation_font_size=10,
        annotation_font_color=RED,
        annotation_position="top right",
    )
    fig1.update_layout(
        **PLOTLY_LAYOUT,
        xaxis=dict(
            tickprefix="₵", showgrid=False, title="Monthly Rent (₵)", title_font_size=11
        ),
        yaxis=dict(
            showgrid=True, gridcolor=GRID_COLOR, title="Listings", title_font_size=11
        ),
        bargap=0.05,
    )
    st.plotly_chart(fig1, use_container_width=True, config=CHART_CFG)

    # Top locations by listing volume
    section_heading("Listing Volume by Location")
    page_note("Top 15 locations within filtered results.")

    loc_vol = (
        df.groupby("loc")["price"]
        .agg(count="count", median="median")
        .reset_index()
        .nlargest(15, "count")
        .sort_values("count")
    )

    fig2 = go.Figure(
        go.Bar(
            x=loc_vol["count"],
            y=loc_vol["loc"].str.title(),
            orientation="h",
            marker_color=BAR_COLOR,
            hovertemplate="<b>%{y}</b><br>Listings: %{x:,}<extra></extra>",
        )
    )
    fig2.update_layout(
        **PLOTLY_LAYOUT,
        height=380,
        xaxis=dict(
            showgrid=True, gridcolor=GRID_COLOR, title="Listings", title_font_size=11
        ),
        yaxis=dict(showgrid=False, title=None, tickfont=dict(size=11)),
    )
    st.plotly_chart(fig2, use_container_width=True, config=CHART_CFG)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 · SEGMENTS
# ─────────────────────────────────────────────────────────────────────────────
with tab_seg:
    # Median vs Mean by locality (top 15 by count)
    section_heading("Median vs. Mean Rent by Location")
    page_note(
        "Top 15 locations. Gap between median and mean signals skew from luxury listings."
    )

    loc_stats = (
        df.groupby("loc")["price"]
        .agg(count="count", median="median", mean="mean")
        .reset_index()
        .query("count >= 5")
        .nlargest(15, "count")
        .sort_values("median")
    )

    fig3 = go.Figure()
    fig3.add_trace(
        go.Bar(
            name="Median",
            x=loc_stats["median"],
            y=loc_stats["loc"].str.title(),
            orientation="h",
            marker_color=BAR_COLOR,
            hovertemplate="<b>%{y}</b><br>Median: ₵%{x:,.0f}<extra></extra>",
        )
    )
    fig3.add_trace(
        go.Bar(
            name="Mean",
            x=loc_stats["mean"],
            y=loc_stats["loc"].str.title(),
            orientation="h",
            marker_color=BAR_DIM,
            hovertemplate="<b>%{y}</b><br>Mean: ₵%{x:,.0f}<extra></extra>",
        )
    )
    fig3.update_layout(
        **PLOTLY_LAYOUT,
        height=400,
        barmode="group",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.0,
            xanchor="right",
            x=1,
            font_size=11,
        ),
        xaxis=dict(tickprefix="₵", showgrid=True, gridcolor=GRID_COLOR, title=None),
        yaxis=dict(showgrid=False, title=None, tickfont=dict(size=11)),
    )
    st.plotly_chart(fig3, use_container_width=True, config=CHART_CFG)

    # Segment bars: property type / furnishing / condition
    sc1, sc2, sc3 = st.columns(3, gap="medium")

    def _compact_bar(group_col: str, title: str, col):
        seg = (
            df.groupby(group_col)["price"]
            .agg(count="count", median="median")
            .reset_index()
            .query("count >= 3")
            .sort_values("median")
        )
        fig = go.Figure(
            go.Bar(
                x=seg["median"],
                y=seg[group_col].str.title(),
                orientation="h",
                marker_color=BAR_COLOR,
                hovertemplate="<b>%{y}</b><br>Median: ₵%{x:,.0f}<extra></extra>",
            )
        )
        fig.update_layout(
            **PLOTLY_LAYOUT,
            height=max(200, len(seg) * 30 + 50),
            margin=dict(l=0, r=0, t=36, b=0),
            title=dict(text=title, font_size=11, x=0, xanchor="left"),
            xaxis=dict(
                tickprefix="₵",
                showgrid=True,
                gridcolor=GRID_COLOR,
                title=None,
                tickfont_size=10,
            ),
            yaxis=dict(showgrid=False, title=None, tickfont_size=10),
        )
        with col:
            st.plotly_chart(fig, use_container_width=True, config=CHART_CFG)

    _compact_bar("house_type", "By Property Type", sc1)
    _compact_bar("furnishing", "By Furnishing", sc2)
    _compact_bar("condition", "By Condition", sc3)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 · LISTINGS
# ─────────────────────────────────────────────────────────────────────────────
with tab_list:
    section_heading("Filtered Listings")
    page_note(
        f"Showing up to 500 of {n_filtered:,} filtered listings, sorted by price."
    )

    display_cols = {
        "loc": "Location",
        "house_type": "Type",
        "bedrooms": "Beds",
        "bathrooms": "Baths",
        "condition": "Condition",
        "furnishing": "Furnishing",
        "price": "Rent (₵/mo)",
    }

    disp = (
        df[list(display_cols.keys())]
        .rename(columns=display_cols)
        .sort_values("Rent (₵/mo)", ascending=False)
        .head(500)
    )
    for col_str in ["Location", "Type", "Condition", "Furnishing"]:
        if col_str in disp.columns:
            disp[col_str] = disp[col_str].str.title()
    disp["Rent (₵/mo)"] = disp["Rent (₵/mo)"].map("₵{:,.0f}".format)

    st.dataframe(disp.reset_index(drop=True), use_container_width=True, hide_index=True)
