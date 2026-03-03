from dataclasses import dataclass

import streamlit as st

from utils import (
    PAGE_EXPLORER,
    PAGE_PREDICTOR,
    PROJECT_GITHUB_URL,
    check_api,
    inject_styles,
    load_market_data,
    stat_row_html,
    workflow_card_html,
)


@dataclass(frozen=True)
class HomeSnapshot:
    api_online: bool
    total_listings: str
    locations_covered: str


def configure_page() -> None:
    st.set_page_config(
        page_title="ARES · Accra Rent Estimator",
        page_icon="🏠",
        layout="centered",
        initial_sidebar_state="collapsed",
    )
    inject_styles()


def build_snapshot() -> HomeSnapshot:
    api_online = check_api()
    market_data = load_market_data()
    return HomeSnapshot(
        api_online=api_online,
        total_listings=f"{len(market_data):,}" if market_data is not None else "—",
        locations_covered=f"{market_data['loc'].nunique()}"
        if market_data is not None
        else "—",
    )


def render_header() -> None:
    st.markdown("## ARES")
    st.markdown(
        "<p style='color:var(--t2);font-size:0.9rem;margin-top:-0.4rem;margin-bottom:0;'>"
        "Accra Rental Estimation System &mdash; Greater Accra Region, Ghana"
        "</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")


def render_snapshot_metrics(snapshot: HomeSnapshot) -> None:
    api_label = "Online" if snapshot.api_online else "Offline"
    api_class = "online" if snapshot.api_online else "offline"
    st.markdown(
        stat_row_html(
            [
                ("Prediction API", api_label, api_class),
                ("Total Listings", snapshot.total_listings, ""),
                ("Total Localities", snapshot.locations_covered, ""),
                ("Market", "Greater Accra", ""),
            ]
        ),
        unsafe_allow_html=True,
    )


def render_ctas(api_online: bool) -> None:
    left_col, right_col = st.columns(2, gap="medium")
    with left_col:
        if st.button("Explore Market →", width="stretch"):
            st.switch_page(PAGE_EXPLORER)

    with right_col:
        if st.button("Run Predictor →", width="stretch"):
            if not api_online:
                st.error("API is offline. Start the FastAPI service first.")
            else:
                st.switch_page(PAGE_PREDICTOR)


def render_footer() -> None:
    st.markdown("---")
    st.markdown(
        "<p style='color:var(--t3);font-size:0.72rem;text-align:center;'>"
        "ARES uses an ML model trained on historical Jiji.com.gh listings. "
        "Estimates are indicative and should be validated against current market conditions."
        "</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='color:var(--t3);font-size:0.72rem;text-align:center;margin-top:0.25rem;'>"
        f"Project repository: <a href='{PROJECT_GITHUB_URL}' target='_blank'>"
        "github.com/kcnewman/ares</a></p>",
        unsafe_allow_html=True,
    )


def main() -> None:
    configure_page()
    snapshot = build_snapshot()
    render_header()
    render_snapshot_metrics(snapshot)
    st.markdown(workflow_card_html(), unsafe_allow_html=True)
    render_ctas(snapshot.api_online)
    render_footer()


main()
