import streamlit as st

from ui.utils import (
    PAGE_EXPLORER,
    PAGE_PREDICTOR,
    check_api,
    inject_styles,
    load_market_data,
    stat_row_html,
    workflow_card_html,
)


def configure_page() -> None:
    st.set_page_config(
        page_title="ARES \u00b7 Accra Rent Estimator",
        page_icon="\U0001f3e0",
        layout="centered",
        initial_sidebar_state="collapsed",
    )
    inject_styles()


def render_header() -> None:
    st.markdown("## ARES")
    st.markdown(
        "<p style='color:var(--t2);font-size:0.9rem;margin-top:-0.4rem;margin-bottom:0;'>"
        "Accra Rental Estimation System &mdash; Greater Accra Region, Ghana"
        "</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")


def render_snapshot() -> None:
    api_online = check_api()
    market_data = load_market_data()
    total_listings = f"{len(market_data):,}" if market_data is not None else "\u2014"
    locations = (
        f"{market_data['loc'].nunique()}" if market_data is not None else "\u2014"
    )

    api_label = "Online" if api_online else "Offline"
    api_class = "online" if api_online else "offline"
    st.markdown(
        stat_row_html(
            [
                ("Prediction API", api_label, api_class),
                ("Total Listings", total_listings, ""),
                ("Localities", locations, ""),
                ("Market", "Greater Accra", ""),
            ]
        ),
        unsafe_allow_html=True,
    )


def render_ctas() -> None:
    api_online = check_api()
    left_col, right_col = st.columns(2, gap="medium")
    with left_col:
        if st.button("Explore Market \u2192", width="stretch"):
            st.switch_page(PAGE_EXPLORER)
    with right_col:
        if st.button("Run Predictor \u2192", width="stretch"):
            if not api_online:
                st.error("API is offline. Start the FastAPI service first.")
            else:
                st.switch_page(PAGE_PREDICTOR)


def render_footer() -> None:
    st.markdown("---")
    st.markdown(
        "<p style='color:var(--t3);font-size:0.72rem;text-align:center;'>"
        "ARES uses an ML model trained on historical rental listings. "
        "Estimates are indicative and should be validated against current market conditions."
        "</p>",
        unsafe_allow_html=True,
    )


def main() -> None:
    configure_page()
    render_header()
    render_snapshot()
    st.markdown(workflow_card_html(), unsafe_allow_html=True)
    render_ctas()
    render_footer()


main()
