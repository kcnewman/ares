from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from ui.utils import (
    AMENITY_LABELS,
    BAR_COLOR,
    CHART_CFG,
    GRID_COLOR,
    PAGE_EXPLORER,
    PAGE_PREDICTOR,
    PLOTLY_LAYOUT,
    RED,
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


@dataclass(frozen=True)
class PredictionContext:
    estimated_price: float
    lower_band: float
    upper_band: float
    volatility_pct: float
    volatility_tier: str
    location: str
    property_type: str
    condition: str
    furnishing: str
    bedrooms: int
    bathrooms: int
    amenities: dict[str, Any]
    generated_at: str
    summary: str = ""
    key_factors: list[dict[str, str]] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    confidence: str = "Moderate"


@dataclass(frozen=True)
class SegmentSummary:
    data: pd.DataFrame
    label: str
    listing_count: int
    median: float
    q25: float
    q75: float
    confidence: str


def configure_page() -> None:
    st.set_page_config(
        page_title="ARES \u00b7 Valuation Report",
        page_icon="\U0001f4ca",
        layout="centered",
        initial_sidebar_state="collapsed",
    )
    inject_styles()


def maybe_scroll_to_top() -> None:
    uid = st.session_state.get("scroll_uid", 0)
    if st.session_state.get("_report_uid") != uid:
        scroll_to_top(uid)
        st.session_state["_report_uid"] = uid


def get_required_session_data() -> tuple[dict[str, Any], dict[str, Any]]:
    result = st.session_state.get("prediction_result")
    inputs = st.session_state.get("form_inputs", {})
    if result:
        return result, inputs

    st.warning("No valuation data found. Run a prediction first.")
    if st.button("\u2190 Go to Predictor"):
        st.switch_page(PAGE_PREDICTOR)
    st.stop()


def build_prediction_context(
    result: dict[str, Any],
    inputs: dict[str, Any],
) -> PredictionContext:
    volatility_pct = float(result.get("market_volatility_pct", 0))
    volatility_tier = str(result.get("market_volatility_tier", "Moderate")).title()
    key_factors_raw = result.get("key_factors", [])
    key_factors = (
        [
            {
                "factor": str(kf.get("factor", "")),
                "impact": str(kf.get("impact", "")),
                "direction": str(kf.get("direction", "neutral")),
            }
            for kf in key_factors_raw
            if isinstance(kf, dict)
        ]
        if isinstance(key_factors_raw, list)
        else []
    )

    return PredictionContext(
        estimated_price=float(result.get("estimated_price", 0)),
        lower_band=float(result.get("lower_band", 0)),
        upper_band=float(result.get("upper_band", 0)),
        volatility_pct=volatility_pct,
        volatility_tier=volatility_tier,
        location=str(inputs.get("location", "\u2014")),
        property_type=str(inputs.get("property_type", "\u2014")),
        condition=str(inputs.get("condition", "\u2014")),
        furnishing=str(inputs.get("furnishing", "\u2014")),
        bedrooms=int(inputs.get("bedrooms", 0)),
        bathrooms=int(inputs.get("bathrooms", 0)),
        amenities=dict(inputs.get("amenities", {})),
        generated_at=str(
            inputs.get("generated_at", datetime.now().strftime("%d %b %Y \u00b7 %H:%M"))
        ),
        summary=str(result.get("summary", "")),
        key_factors=key_factors,
        risks=list(result.get("risks", [])),
        confidence=str(result.get("confidence", "Moderate")),
    )


def build_segment_summary(
    market_data: pd.DataFrame | None,
    context: PredictionContext,
) -> SegmentSummary | None:
    if market_data is None:
        return None

    segment_df, segment_label = compute_segment(
        market_data,
        context.location,
        context.property_type,
        context.furnishing,
    )
    listing_count = len(segment_df)
    return SegmentSummary(
        data=segment_df,
        label=segment_label,
        listing_count=listing_count,
        median=segment_df["price"].median(),
        q25=segment_df["price"].quantile(0.25),
        q75=segment_df["price"].quantile(0.75),
        confidence=confidence_tier(listing_count),
    )


def build_comparables_dataset(
    market_data: pd.DataFrame,
    context: PredictionContext,
) -> tuple[pd.DataFrame, str]:
    comparables = market_data[
        (market_data["loc"] == context.location)
        & (market_data["house_type"] == context.property_type)
    ].copy()
    scope_note = f"{context.property_type.title()} in {context.location.title()}"

    if len(comparables) < 5:
        comparables = market_data[market_data["loc"] == context.location].copy()
        scope_note = (
            f"all types in {context.location.title()} (fewer than 5 exact matches)"
        )

    return comparables, scope_note


def render_navigation() -> None:
    left_col, right_col = st.columns(2, gap="small")
    with left_col:
        if st.button("\u2190 Predictor", key="rpt_pred", width="stretch"):
            st.switch_page(PAGE_PREDICTOR)
    with right_col:
        if st.button("Explorer \u2192", key="rpt_expl", width="stretch"):
            st.switch_page(PAGE_EXPLORER)


def render_header(context: PredictionContext) -> None:
    st.markdown(
        "<h1 style='margin-bottom:0.15rem;'>Valuation Report</h1>"
        f"<p style='color:var(--t2);font-size:0.9rem;margin-top:0;'>"
        f"{context.property_type.title()} &middot; {context.location.title()}</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")


def render_result_and_metadata(
    context: PredictionContext,
    market_data: pd.DataFrame | None,
    segment: SegmentSummary | None,
) -> None:
    segment_median = segment.median if segment is not None else None
    st.markdown(
        result_card_html(
            context.estimated_price,
            context.lower_band,
            context.upper_band,
            context.volatility_pct,
            context.volatility_tier,
            seg_median=segment_median,
        ),
        unsafe_allow_html=True,
    )

    chips = [
        ("Location", context.location.title()),
        ("Property Type", context.property_type.title()),
        ("Condition", context.condition.title()),
        ("Furnishing", context.furnishing.title()),
        ("Bedrooms", str(context.bedrooms)),
        ("Bathrooms", str(context.bathrooms)),
    ]
    if context.amenities:
        amenity_names = ", ".join(
            AMENITY_LABELS.get(key, key.replace("_", " ").title())
            for key in context.amenities
        )
        chips.append(("Amenities", amenity_names))

    chip_style = (
        "background:var(--bg);border:1px solid var(--bd);"
        "border-radius:4px;padding:0.15rem 0.55rem;"
        "font-size:0.75rem;white-space:nowrap;"
    )
    st.markdown(
        '<div style="display:flex;flex-wrap:wrap;gap:0.35rem;'
        'justify-content:center;margin:-0.5rem 0 1rem;">'
        + "".join(
            f'<span style="{chip_style}">'
            f'<span style="color:var(--t3);font-weight:600;">{lbl}</span>'
            f" {val}</span>"
            for lbl, val in chips
        )
        + "</div>",
        unsafe_allow_html=True,
    )

    market_count = f"{len(market_data):,}" if market_data is not None else "\u2014"
    st.markdown(
        '<div class="report-meta">'
        f"Generated {context.generated_at} &nbsp;\u00b7&nbsp; Market snapshot: {market_count} listings"
        "</div>",
        unsafe_allow_html=True,
    )


def render_insights_tab(
    market_data: pd.DataFrame | None,
    context: PredictionContext,
    segment: SegmentSummary | None,
) -> None:
    if market_data is None or segment is None:
        st.info("Market data unavailable. Set the `DATA_PATH` env variable.")
        return

    st.markdown(
        insight_box_html(
            segment.label,
            segment.listing_count,
            segment.median,
            segment.q25,
            segment.q75,
            context.estimated_price,
            segment.confidence,
        ),
        unsafe_allow_html=True,
    )
    page_note(
        f"Distribution of {segment.listing_count:,} listings in segment '{segment.label}'. "
        "Red dashed line marks your estimate."
    )

    p99 = segment.data["price"].quantile(0.99)
    plot_data = segment.data[segment.data["price"] <= p99]

    figure = px.histogram(
        plot_data,
        x="price",
        nbins=30,
        labels={"price": "Monthly Rent (\u20b5)"},
        color_discrete_sequence=[BAR_COLOR],
    )
    figure.add_vline(
        x=context.estimated_price,
        line_color=RED,
        line_width=2,
        line_dash="dash",
        annotation_text="Your estimate",
        annotation_position="top right",
        annotation_font_color=RED,
        annotation_font_size=10,
    )
    figure.add_vrect(
        x0=context.lower_band,
        x1=context.upper_band,
        fillcolor=RED,
        opacity=0.06,
        layer="below",
        line_width=0,
    )
    figure.update_layout(
        **PLOTLY_LAYOUT,
        xaxis=dict(
            tickprefix="\u20b5",
            showgrid=False,
            title="Monthly Rent (\u20b5)",
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
    st.plotly_chart(figure, width="stretch", config=CHART_CFG)
    st.markdown('<div style="height:1.1rem;"></div>', unsafe_allow_html=True)
    render_comparables_table(market_data, context)


def render_comparables_table(
    market_data: pd.DataFrame | None,
    context: PredictionContext,
) -> None:
    if market_data is None:
        st.info("Market data unavailable. Set the `DATA_PATH` env variable.")
        return

    comparables, scope_note = build_comparables_dataset(market_data, context)

    if comparables.empty:
        st.info("No comparable listings found for this location.")
        return

    section_heading("Comparable Listings")
    comparables["_delta"] = (comparables["price"] - context.estimated_price).abs()
    comparables = comparables.nsmallest(25, "_delta").sort_values("_delta")
    page_note(
        "Top 25 listings closest to your estimate. "
        f"Scope: {scope_note}. Use Listing URL to open each listing."
    )

    display_columns = {
        "house_type": "Type",
        "bedrooms": "Beds",
        "bathrooms": "Baths",
        "condition": "Condition",
        "furnishing": "Furnishing",
        "price": "Rent (\u20b5/mo)",
    }
    if "url" in comparables.columns:
        display_columns["url"] = "Listing URL"
    display_frame = comparables[list(display_columns)].rename(columns=display_columns)
    for column_name in ["Type", "Condition", "Furnishing"]:
        display_frame[column_name] = display_frame[column_name].str.title()
    display_frame["Rent (\u20b5/mo)"] = comparables["price"].map("\u20b5{:,.0f}".format)
    if "Listing URL" in display_frame.columns:
        display_frame["Listing URL"] = (
            display_frame["Listing URL"].fillna("").astype(str).str.strip()
        )

    display_frame = display_frame.reset_index(drop=True)
    if "Listing URL" in display_frame.columns:
        st.dataframe(
            display_frame,
            width="stretch",
            hide_index=True,
            column_config={
                "Listing URL": st.column_config.LinkColumn(
                    "Listing URL",
                    help="Open source listing in a new tab.",
                    display_text="Open listing",
                )
            },
        )
    else:
        st.dataframe(
            display_frame,
            width="stretch",
            hide_index=True,
        )


def render_explanation(context: PredictionContext) -> None:
    if not context.summary and not context.key_factors and not context.risks:
        return

    section_heading("AI Explanation")

    if context.summary:
        st.markdown(
            f"<p style='color:var(--t1);font-size:0.95rem;line-height:1.6;margin-bottom:1rem;'>{context.summary}</p>",
            unsafe_allow_html=True,
        )

    if context.key_factors:
        for kf in context.key_factors:
            direction = kf.get("direction", "neutral")
            icon = {"up": "\u2191", "down": "\u2193", "neutral": "\u2192"}.get(
                direction, "\u2192"
            )
            st.markdown(
                f"<div style='display:flex;align-items:baseline;gap:0.5rem;padding:0.3rem 0;'>"
                f"<span style='font-size:0.85rem;'>{icon}</span>"
                f"<span style='font-weight:600;font-size:0.88rem;'>{kf.get('factor', '')}</span>"
                f"<span style='color:var(--t3);font-size:0.82rem;'>{kf.get('impact', '')}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    if context.risks:
        st.markdown(
            "<p style='font-size:0.82rem;font-weight:700;color:var(--amber);margin:0.75rem 0 0.25rem;'>"
            "\u26a0 Risks</p>",
            unsafe_allow_html=True,
        )
        for risk in context.risks:
            st.markdown(
                f"<p style='font-size:0.85rem;color:var(--t2);margin:0.1rem 0;'>{risk}</p>",
                unsafe_allow_html=True,
            )

    if context.confidence:
        c_color = {
            "High": "var(--green)",
            "Moderate": "var(--amber)",
            "Low": "var(--red)",
        }.get(context.confidence, "inherit")
        st.markdown(
            f"<p style='font-size:0.82rem;margin:0.75rem 0 0;'>"
            f"Confidence: <span style='font-weight:700;color:{c_color};'>{context.confidence}</span></p>",
            unsafe_allow_html=True,
        )

    st.markdown("---")


def main() -> None:
    configure_page()
    maybe_scroll_to_top()
    result, inputs = get_required_session_data()
    context = build_prediction_context(result, inputs)
    market_data = load_market_data()
    segment = build_segment_summary(market_data, context)

    render_navigation()
    render_header(context)
    render_result_and_metadata(context, market_data, segment)

    render_explanation(context)

    section_heading("Market Analysis")
    render_insights_tab(market_data, context, segment)

    st.markdown("---")


main()
st.markdown(
    "<p style='text-align:center;color:var(--t3);font-size:0.72rem;'>"
    "ARES uses an ML model trained on historical Accra rental listings. "
    "Estimates are indicative only."
    "</p>",
    unsafe_allow_html=True,
)
