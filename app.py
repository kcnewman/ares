"""
app.py — Home Page
Displays API status, market snapshot metrics, workflow overview, and CTAs.
"""

import streamlit as st

from utils import (
    PAGE_EXPLORER, PAGE_PREDICTOR,
    inject_styles, stat_row_html, workflow_card_html,
    check_api, load_market_data,
)

st.set_page_config(
    page_title="ARES · Accra Rent Estimator",
    page_icon="🏠",
    layout="centered",
    initial_sidebar_state="collapsed",
)
inject_styles()

# ── Data + API status ─────────────────────────────────────────────────────────
api_ok = check_api()
df     = load_market_data()

total_listings = f"{len(df):,}" if df is not None else "—"
n_locations    = f"{df['loc'].nunique()}" if df is not None else "—"
api_label      = "Online" if api_ok else "Offline"
api_cls        = "online" if api_ok else "offline"

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## ARES")
st.markdown(
    "<p style='color:var(--t2);font-size:0.9rem;margin-top:-0.4rem;margin-bottom:0;'>"
    "Accra Rental Estimation System &mdash; Greater Accra Region, Ghana"
    "</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

# ── Snapshot metrics ──────────────────────────────────────────────────────────
st.markdown(stat_row_html([
    ("Prediction API",     api_label,       api_cls),
    ("Listings in Dataset", total_listings, ""),
    ("Locations Covered",  n_locations,     ""),
    ("Market",             "Greater Accra", ""),
]), unsafe_allow_html=True)

# ── Workflow card ─────────────────────────────────────────────────────────────
st.markdown(workflow_card_html(), unsafe_allow_html=True)

# ── CTAs ──────────────────────────────────────────────────────────────────────
c1, c2 = st.columns(2, gap="medium")

with c1:
    if st.button("Explore Market →", use_container_width=True):
        st.switch_page(PAGE_EXPLORER)

with c2:
    if st.button("Run Predictor →", use_container_width=True):
        if not api_ok:
            st.error("API is offline. Start the FastAPI service first.")
        else:
            st.switch_page(PAGE_PREDICTOR)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='color:var(--t3);font-size:0.72rem;text-align:center;'>"
    "ARES uses an ML model trained on historical Jiji.com.gh listings. "
    "Estimates are indicative and should be validated against current market conditions."
    "</p>",
    unsafe_allow_html=True,
)
