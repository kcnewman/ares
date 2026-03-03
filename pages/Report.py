"""
pages/Report.py — Prediction Report (Page 2 of 2)

Renders the centred result card, property summary chips, and
market-context tabs (distribution, comparable listings, amenity impact).
"""

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils import (
    AMENITY_LABELS,
    chip_grid_html,
    inject_styles,
    load_market_data,
    result_card_html,
    scroll_to_top,
)

st.set_page_config(
    page_title="ARES · Valuation Report",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="collapsed",
)
inject_styles()

_PLOTLY_LAYOUT: dict = dict(
    font_family="Manrope, sans-serif",
    font_color="#52525b",
    plot_bgcolor="#ffffff",
    paper_bgcolor="#ffffff",
    margin=dict(l=0, r=0, t=24, b=0),
    showlegend=False,
    hoverlabel=dict(
        bgcolor="#18181b",
        font_color="#ffffff",
        font_family="Manrope, sans-serif",
        font_size=12,
    ),
)

_uid_key = "report_scroll_uid"
_uid = st.session_state.get("scroll_uid", 0)
if st.session_state.get(_uid_key) != _uid:
    scroll_to_top(_uid)
    st.session_state[_uid_key] = _uid

result = st.session_state.get("prediction_result")
inputs = st.session_state.get("form_inputs", {})

if not result:
    st.warning("No valuation data found. Please run a valuation first.")
    if st.button("← Go to Estimator"):
        st.switch_page("app.py")
    st.stop()

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

back_col, _ = st.columns([1, 4])
with back_col:
    if st.button("← New Valuation"):
        st.switch_page("app.py")

st.markdown(
    "<h1 style='margin-bottom:0.15rem;'>Valuation Report</h1>"
    f"<p style='color:var(--text-2);font-size:0.9rem;margin-top:0;'>"
    f"{prop_type.title()} &middot; {loc.title()}</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

st.markdown(result_card_html(est_price, low_b, high_b, vol), unsafe_allow_html=True)

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
        AMENITY_LABELS.get(k) or k.replace("_", " ").title() for k in amenities
    )
    chips.append(("Amenities", am_names))

st.markdown(chip_grid_html(chips), unsafe_allow_html=True)

st.markdown("---")

st.markdown(
    "<div class='section-heading'><span class='eyebrow'>Market Context</span></div>",
    unsafe_allow_html=True,
)

tab_dist, tab_type, tab_comps = st.tabs(
    [
        "Price Distribution",
        "By Property Type",
        "Comparable Listings",
    ]
)

df = load_market_data()

if df is None:
    for tab in [tab_dist, tab_type, tab_comps]:
        with tab:
            st.info(
                "Market data unavailable. "
                "Set the `DATA_PATH` env variable to `preprocessed_train.csv`."
            )
else:
    with tab_dist:
        loc_df = df[df["loc"] == loc] if loc in df["loc"].values else df
        scope = f"in {loc.title()}" if loc in df["loc"].values else "(all locations)"
        n = len(loc_df)

        st.markdown(
            f"<p style='color:var(--text-2);font-size:0.85rem;'>"
            f"Rental price distribution {scope} &mdash; {n:,} listings</p>",
            unsafe_allow_html=True,
        )

        p99 = loc_df["price"].quantile(0.99)
        plot_df = loc_df[loc_df["price"] <= p99]

        fig = px.histogram(
            plot_df,
            x="price",
            nbins=40,
            labels={"price": "Monthly Rent (₵)"},
            color_discrete_sequence=["#18181b"],
        )

        fig.add_vline(
            x=est_price,
            line_color="#dc2626",
            line_width=2,
            line_dash="dash",
            annotation_text="Your estimate",
            annotation_position="top right",
            annotation_font_color="#dc2626",
            annotation_font_size=11,
        )
        fig.add_vrect(
            x0=low_b,
            x1=high_b,
            fillcolor="#dc2626",
            opacity=0.07,
            layer="below",
            line_width=0,
        )

        fig.update_layout(
            **_PLOTLY_LAYOUT,
            xaxis=dict(
                tickprefix="₵",
                showgrid=False,
                title="Monthly Rent (₵)",
                title_font_size=11,
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor="#f4f4f5",
                title="# Listings",
                title_font_size=11,
            ),
            bargap=0.06,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        q25, q50, q75 = (
            loc_df["price"].quantile(0.25),
            loc_df["price"].median(),
            loc_df["price"].quantile(0.75),
        )
        stat_chips = [
            ("25th Pct.", f"₵{q25:,.0f}"),
            ("Median", f"₵{q50:,.0f}"),
            ("75th Pct.", f"₵{q75:,.0f}"),
            ("Your Estimate", f"₵{est_price:,.0f}"),
        ]
        st.markdown(chip_grid_html(stat_chips), unsafe_allow_html=True)

    with tab_type:
        type_stats = (
            df.groupby("house_type")["price"]
            .agg(median="median", count="count")
            .reset_index()
            .query("count >= 10")
            .sort_values("median", ascending=True)
        )

        colors = [
            "#18181b" if t == prop_type else "#d4d4d8" for t in type_stats["house_type"]
        ]

        fig2 = go.Figure(
            go.Bar(
                x=type_stats["median"],
                y=type_stats["house_type"].str.title(),
                orientation="h",
                marker_color=colors,
                hovertemplate="<b>%{y}</b><br>Median: ₵%{x:,.0f}<extra></extra>",
            )
        )
        fig2.update_layout(
            **_PLOTLY_LAYOUT,
            height=420,
            xaxis=dict(
                tickprefix="₵",
                showgrid=True,
                gridcolor="#f4f4f5",
                title=None,
            ),
            yaxis=dict(showgrid=False, title=None),
        )
        st.markdown(
            "<p style='color:var(--text-2);font-size:0.85rem;'>"
            "Median monthly rent by property type (min. 10 listings). "
            "Your selected type is highlighted.</p>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            fig2, use_container_width=True, config={"displayModeBar": False}
        )

    with tab_comps:
        comp_df = df[(df["loc"] == loc) & (df["house_type"] == prop_type)].copy()

        if len(comp_df) < 5:
            comp_df = df[df["loc"] == loc].copy()
            scope_note = f"(all types in {loc.title()} — fewer than 5 exact matches)"
        else:
            scope_note = f"({prop_type.title()} in {loc.title()})"

        comp_df["_delta"] = (comp_df["price"] - est_price).abs()
        comp_df = comp_df.nsmallest(20, "_delta").copy()

        display_cols = [
            "house_type",
            "bedrooms",
            "bathrooms",
            "condition",
            "furnishing",
            "price",
        ]
        rename_map = {
            "house_type": "Type",
            "bedrooms": "Beds",
            "bathrooms": "Baths",
            "condition": "Condition",
            "furnishing": "Furnishing",
            "price": "Rent (₵/mo)",
        }

        if len(comp_df) == 0:
            st.info("No comparable listings found for this location.")
        else:
            st.markdown(
                f"<p style='color:var(--text-2);font-size:0.85rem;'>"
                f"Listings closest to your estimate {scope_note}.</p>",
                unsafe_allow_html=True,
            )
            disp = (
                comp_df[display_cols]
                .rename(columns=rename_map)
                .assign(**{"Rent (₵/mo)": comp_df["price"].map("₵{:,.0f}".format)})
            )
            disp["Type"] = disp["Type"].str.title()
            disp["Condition"] = disp["Condition"].str.title()
            disp["Furnishing"] = disp["Furnishing"].str.title()

            st.dataframe(
                disp.reset_index(drop=True),
                use_container_width=True,
                hide_index=True,
            )

st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:var(--text-3);font-size:0.75rem;'>"
    "ARES uses a machine learning model trained on historical Accra rental listings. "
    "Estimates are indicative and should be validated against current market conditions."
    "</p>",
    unsafe_allow_html=True,
)
