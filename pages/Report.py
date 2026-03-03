"""
pages/Report.py — Prediction Report
Centered result card + property chips + report metadata + tabbed market analysis.
Tabs: Insights (segment context) | Comparables (ranked table).
"""

from datetime import datetime

import plotly.express as px
import streamlit as st

from utils import (
    AMENITY_LABELS,
    BAR_COLOR,
    CHART_CFG,
    GRID_COLOR,
    PAGE_EXPLORER,
    PAGE_PREDICTOR,
    PLOTLY_LAYOUT,
    RED,
    chip_grid_html,
    compute_segment,
    confidence_tier,
    inject_styles,
    insight_box_html,
    load_market_data,
    page_note,
    result_card_html,
    scroll_to_top,
    section_heading,
)

st.set_page_config(
    page_title="ARES · Valuation Report",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="collapsed",
)
inject_styles()

# ── Scroll to top on fresh navigation ────────────────────────────────────────
_uid = st.session_state.get("scroll_uid", 0)
if st.session_state.get("_report_uid") != _uid:
    scroll_to_top(_uid)
    st.session_state["_report_uid"] = _uid

# ── Guard ─────────────────────────────────────────────────────────────────────
result = st.session_state.get("prediction_result")
inputs = st.session_state.get("form_inputs", {})

if not result:
    st.warning("No valuation data found. Run a prediction first.")
    if st.button("← Go to Predictor"):
        st.switch_page(PAGE_PREDICTOR)
    st.stop()

# ── Unpack result + inputs ────────────────────────────────────────────────────
est_price = result.get("estimated_price", 0)
low_b = result.get("lower_band", 0)
high_b = result.get("upper_band", 0)
vol = result.get("market_volatility_idx", 0)

loc = inputs.get("location", "—")
prop_type = inputs.get("property_type", "—")
condition = inputs.get("condition", "—")
furnishing = inputs.get("furnishing", "—")
bedrooms = inputs.get("bedrooms", 0)
bathrooms = inputs.get("bathrooms", 0)
amenities = inputs.get("amenities", {})
gen_at = inputs.get("generated_at", datetime.now().strftime("%d %b %Y · %H:%M"))

# ── Load market data + compute segment ───────────────────────────────────────
df = load_market_data()
seg_df, seg_label, seg_median_val = None, None, None

if df is not None:
    seg_df, seg_label = compute_segment(df, loc, prop_type, furnishing)
    seg_n = len(seg_df)
    seg_median_val = seg_df["price"].median()
    seg_q25 = seg_df["price"].quantile(0.25)
    seg_q75 = seg_df["price"].quantile(0.75)
    confidence = confidence_tier(seg_n)

# ── Nav bar ───────────────────────────────────────────────────────────────────
nc1, nc2 = st.columns(2, gap="small")
with nc1:
    if st.button("← Predictor", key="rpt_pred", use_container_width=True):
        st.switch_page(PAGE_PREDICTOR)
with nc2:
    if st.button("Explorer →", key="rpt_expl", use_container_width=True):
        st.switch_page(PAGE_EXPLORER)

# ── Page title ────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='margin-bottom:0.15rem;'>Valuation Report</h1>"
    f"<p style='color:var(--t2);font-size:0.9rem;margin-top:0;'>"
    f"{prop_type.title()} &middot; {loc.title()}</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

# ── Result card ───────────────────────────────────────────────────────────────
st.markdown(
    result_card_html(est_price, low_b, high_b, vol, seg_median=seg_median_val),
    unsafe_allow_html=True,
)

# ── Report metadata ───────────────────────────────────────────────────────────
n_market = f"{len(df):,}" if df is not None else "—"
st.markdown(
    f'<div class="report-meta">'
    f"Generated {gen_at} &nbsp;·&nbsp; Market snapshot: {n_market} listings"
    f"</div>",
    unsafe_allow_html=True,
)

# ── Property chips ────────────────────────────────────────────────────────────
chips = [
    ("Location", loc.title()),
    ("Property Type", prop_type.title()),
    ("Condition", condition.title()),
    ("Furnishing", furnishing.title()),
    ("Bedrooms", str(bedrooms)),
    ("Bathrooms", str(bathrooms)),
]
if amenities:
    am_names = ", ".join(
        AMENITY_LABELS.get(k, k.replace("_", " ").title()) for k in amenities
    )
    chips.append(("Amenities", am_names))

st.markdown(chip_grid_html(chips), unsafe_allow_html=True)
st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
section_heading("Market Analysis")
tab_ins, tab_comp = st.tabs(["Insights", "Comparables"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 · INSIGHTS
# ─────────────────────────────────────────────────────────────────────────────
with tab_ins:
    if df is None:
        st.info("Market data unavailable. Set the `DATA_PATH` env variable.")
    else:
        # Segment context box
        st.markdown(
            insight_box_html(
                seg_label,
                seg_n,
                seg_median_val,
                seg_q25,
                seg_q75,
                est_price,
                confidence,
            ),
            unsafe_allow_html=True,
        )

        # Price distribution: segment vs estimate
        page_note(
            f"Distribution of {seg_n:,} listings in segment '{seg_label}'. "
            "Red dashed line marks your estimate."
        )

        p99 = seg_df["price"].quantile(0.99)
        plot_df = seg_df[seg_df["price"] <= p99]

        fig = px.histogram(
            plot_df,
            x="price",
            nbins=30,
            labels={"price": "Monthly Rent (₵)"},
            color_discrete_sequence=[BAR_COLOR],
        )
        fig.add_vline(
            x=est_price,
            line_color=RED,
            line_width=2,
            line_dash="dash",
            annotation_text="Your estimate",
            annotation_position="top right",
            annotation_font_color=RED,
            annotation_font_size=10,
        )
        fig.add_vrect(
            x0=low_b,
            x1=high_b,
            fillcolor=RED,
            opacity=0.06,
            layer="below",
            line_width=0,
        )
        fig.update_layout(
            **PLOTLY_LAYOUT,
            xaxis=dict(
                tickprefix="₵",
                showgrid=False,
                title="Monthly Rent (₵)",
                title_font_size=11,
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor=GRID_COLOR,
                title="Listings",
                title_font_size=11,
            ),
            bargap=0.05,
        )
        st.plotly_chart(fig, use_container_width=True, config=CHART_CFG)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 · COMPARABLES
# ─────────────────────────────────────────────────────────────────────────────
with tab_comp:
    if df is None:
        st.info("Market data unavailable. Set the `DATA_PATH` env variable.")
    else:
        # Filter: loc + type, fallback to loc only
        comp_df = df[(df["loc"] == loc) & (df["house_type"] == prop_type)].copy()
        scope_note = f"{prop_type.title()} in {loc.title()}"

        if len(comp_df) < 5:
            comp_df = df[df["loc"] == loc].copy()
            scope_note = f"all types in {loc.title()} (fewer than 5 exact matches)"

        if len(comp_df) == 0:
            st.info("No comparable listings found for this location.")
        else:
            # Rank by proximity to estimate
            comp_df["_delta"] = (comp_df["price"] - est_price).abs()
            comp_df = comp_df.nsmallest(25, "_delta").copy()
            comp_df = comp_df.sort_values("_delta")

            page_note(f"Top 25 listings closest to your estimate · {scope_note}.")

            display_cols = {
                "house_type": "Type",
                "bedrooms": "Beds",
                "bathrooms": "Baths",
                "condition": "Condition",
                "furnishing": "Furnishing",
                "price": "Rent (₵/mo)",
            }
            disp = (
                comp_df[list(display_cols.keys())].rename(columns=display_cols).copy()
            )
            for col in ["Type", "Condition", "Furnishing"]:
                disp[col] = disp[col].str.title()
            disp["Rent (₵/mo)"] = comp_df["price"].map("₵{:,.0f}".format)

            st.dataframe(
                disp.reset_index(drop=True),
                use_container_width=True,
                hide_index=True,
            )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:var(--t3);font-size:0.72rem;'>"
    "ARES uses an ML model trained on historical Accra rental listings. "
    "Estimates are indicative only."
    "</p>",
    unsafe_allow_html=True,
)
