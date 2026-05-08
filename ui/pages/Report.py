import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx
import pandas as pd
import plotly.express as px
import streamlit as st
from utils import (
    AMENITY_LABELS,
    BACKEND_URL,
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

LLM_SERVICE_URL = os.getenv("LLM_SERVICE_URL", "http://localhost:8001")


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
        page_title="ARES · Valuation Report",
        page_icon="📊",
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
    if st.button("← Go to Predictor"):
        st.switch_page(PAGE_PREDICTOR)
    st.stop()


def build_prediction_context(
    result: dict[str, Any],
    inputs: dict[str, Any],
) -> PredictionContext:
    return PredictionContext(
        estimated_price=result.get("estimated_price", 0),
        lower_band=result.get("lower_band", 0),
        upper_band=result.get("upper_band", 0),
        volatility_pct=result.get("market_volatility_pct", 0),
        volatility_tier=str(result.get("market_volatility_tier", "Moderate")).title(),
        location=inputs.get("location", "—"),
        property_type=inputs.get("property_type", "—"),
        condition=inputs.get("condition", "—"),
        furnishing=inputs.get("furnishing", "—"),
        bedrooms=inputs.get("bedrooms", 0),
        bathrooms=inputs.get("bathrooms", 0),
        amenities=inputs.get("amenities", {}),
        generated_at=str(
            inputs.get(
                "generated_at", datetime.now(timezone.utc).strftime("%d %b %Y · %H:%M")
            )
        ),
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
        if st.button("← Predictor", key="rpt_pred", width="stretch"):
            st.switch_page(PAGE_PREDICTOR)
    with right_col:
        if st.button("Explorer →", key="rpt_expl", width="stretch"):
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

    market_count = f"{len(market_data):,}" if market_data is not None else "—"
    st.markdown(
        '<div class="report-meta">'
        f"Generated {context.generated_at} &nbsp;·&nbsp; Market snapshot: {market_count} listings"
        "</div>",
        unsafe_allow_html=True,
    )


def render_property_chips(context: PredictionContext) -> None:
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

    st.markdown(chip_grid_html(chips), unsafe_allow_html=True)
    st.markdown("---")


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
        labels={"price": "Monthly Rent (₵)"},
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
        **PLOTLY_LAYOUT,  # type: ignore[arg-type]
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
        "price": "Rent (₵/mo)",
    }
    if "url" in comparables.columns:
        display_columns["url"] = "Listing URL"
    display_frame = comparables[list(display_columns)].rename(columns=display_columns)
    for column_name in ["Type", "Condition", "Furnishing"]:
        display_frame[column_name] = display_frame[column_name].str.title()
    display_frame["Rent (₵/mo)"] = comparables["price"].map("₵{:,.0f}".format)
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


def _fetch_explanation(context: PredictionContext) -> dict | None:
    try:
        payload = {
            "house_type": context.property_type,
            "condition": context.condition,
            "furnishing": context.furnishing,
            "loc": context.location,
            "bathrooms": context.bathrooms,
            "bedrooms": context.bedrooms,
            "24_hour_electricity": context.amenities.get("24_hour_electricity", 0),
            "air_conditioning": context.amenities.get("air_conditioning", 0),
            "apartment": context.amenities.get("apartment", 0),
            "balcony": context.amenities.get("balcony", 0),
            "chandelier": context.amenities.get("chandelier", 0),
            "dining_area": context.amenities.get("dining_area", 0),
            "dishwasher": context.amenities.get("dishwasher", 0),
            "hot_water": context.amenities.get("hot_water", 0),
            "kitchen_cabinets": context.amenities.get("kitchen_cabinets", 0),
            "kitchen_shelf": context.amenities.get("kitchen_shelf", 0),
            "microwave": context.amenities.get("microwave", 0),
            "pop_ceiling": context.amenities.get("pop_ceiling", 0),
            "pre_paid_meter": context.amenities.get("pre_paid_meter", 0),
            "refrigerator": context.amenities.get("refrigerator", 0),
            "tv": context.amenities.get("tv", 0),
            "tiled_floor": context.amenities.get("tiled_floor", 0),
            "wardrobe": context.amenities.get("wardrobe", 0),
            "wi_fi": context.amenities.get("wi_fi", 0),
        }
        resp = httpx.post(f"{BACKEND_URL}/explain", json=payload, timeout=15)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def render_ai_explanation(explanation: dict | None) -> None:
    if explanation is None:
        return

    st.markdown("---")
    section_heading("AI Market Analysis")

    conf = explanation.get("confidence", "")
    conf_color = {"High": "var(--green)", "Moderate": "var(--amber)", "Low": "var(--red)"}.get(conf, "var(--t2)")
    conf_html = f'<span style="color:{conf_color};font-weight:700;">{conf}</span>'

    st.markdown(
        f"<div style='background:var(--surface);border:1px solid var(--bd);"
        f"border-radius:var(--rl);padding:1.25rem 1.5rem;margin-bottom:1rem;'>"
        f"<p style='font-size:0.9rem;line-height:1.6;color:var(--t2);margin:0 0 1rem;'>"
        f"{explanation.get('summary', '')}</p>",
        unsafe_allow_html=True,
    )

    key_factors = explanation.get("key_factors", [])
    if key_factors:
        bullets = []
        for f in key_factors:
            icon = {"up": "▲", "down": "▼", "neutral": "◆"}.get(f.get("direction", "neutral"), "◆")
            color = {"up": "var(--red)", "down": "var(--green)", "neutral": "var(--t3)"}.get(f.get("direction", "neutral"), "var(--t3)")
            bullets.append(
                f"<li style='margin-bottom:0.35rem;font-size:0.88rem;'>"
                f"<span style='color:{color};margin-right:0.4rem;'>{icon}</span>"
                f"<strong>{f.get('factor', '')}</strong>: {f.get('impact', '')}</li>"
            )
        st.markdown(
            f"<div style='margin-bottom:0.75rem;'><div class='eyebrow' style='margin-bottom:0.4rem;'>"
            f"Key Factors</div><ul style='margin:0;padding-left:0;list-style:none;'>{''.join(bullets)}</ul></div>",
            unsafe_allow_html=True,
        )

    risks = explanation.get("risks", [])
    if risks:
        risk_items = "".join(f"<li style='font-size:0.85rem;margin-bottom:0.2rem;'>{r}</li>" for r in risks)
        st.markdown(
            f"<div style='background:#fef2f2;border:1px solid #fecaca;border-radius:var(--r);"
            f"padding:0.75rem 1rem;margin-bottom:0.75rem;'>"
            f"<div class='eyebrow' style='color:var(--red);margin-bottom:0.3rem;'>Risks & Uncertainties</div>"
            f"<ul style='margin:0;padding-left:1.2rem;color:var(--t2);'>{risk_items}</ul></div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        f"<div style='display:flex;justify-content:flex-end;gap:0.5rem;align-items:center;'>"
        f"<span style='font-size:0.72rem;color:var(--t3);'>Confidence: {conf_html}</span></div>"
        f"</div>",
        unsafe_allow_html=True,
    )


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
    render_property_chips(context)

    explanation = _fetch_explanation(context)
    render_ai_explanation(explanation)

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
