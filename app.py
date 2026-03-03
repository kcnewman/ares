"""
app.py — Valuation Input Form (Page 1 of 2)

Collects property details, calls the ARES prediction API,
stores the result in session_state, then navigates to the Report page.
"""

import streamlit as st
import requests

from utils import (
    BACKEND_URL,
    AMENITY_LABELS,
    inject_styles,
    scroll_to_top,
    check_api,
    load_schema,
)

# ── Page config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="ARES · Accra Rent Estimator",
    page_icon="🏠",
    layout="centered",
    initial_sidebar_state="collapsed",
)
inject_styles()

# ── Session state defaults ──────────────────────────────────────────────────
if "scroll_uid" not in st.session_state:
    st.session_state.scroll_uid = 0
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "form_inputs" not in st.session_state:
    st.session_state.form_inputs = {}

# ── Header ──────────────────────────────────────────────────────────────────
st.markdown("<div id='top'></div>", unsafe_allow_html=True)

col_title, col_badge = st.columns([4, 1])
with col_title:
    st.markdown("## ARES")
    st.markdown(
        "<p style='color:var(--text-2);font-size:0.9rem;margin-top:-0.5rem;'>"
        "Accra Rental Estimation System &mdash; Greater Accra Region"
        "</p>",
        unsafe_allow_html=True,
    )
with col_badge:
    st.markdown("<br>", unsafe_allow_html=True)

st.markdown("---")

# ── API / Schema gate ────────────────────────────────────────────────────────
api_online = check_api()
schema = load_schema()

if not api_online:
    st.error("⚠️  Backend API is offline. Start the FastAPI service to use the estimator.")
    st.stop()

if not schema:
    st.error("⚠️  Schema file missing at `artifacts/cache/schema.json`. Check your setup.")
    st.stop()

# ── Form ─────────────────────────────────────────────────────────────────────
with st.form("valuation_form", border=False):

    # Section 1 — Property specs
    st.markdown(
        "<div class='section-heading'>"
        "<span class='eyebrow'>Property Details</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2, gap="medium")

    with c1:
        loc_options = sorted(schema["mappings"]["location_class"].keys())
        loc = st.selectbox("Location", options=loc_options)

        prop_options = list(schema["mappings"]["property_density"].keys())
        prop_type = st.selectbox("Property Type", options=prop_options)

        cond_options = list(schema["mappings"]["condition_transform"].keys())
        condition = st.selectbox("Condition", options=cond_options)

    with c2:
        furn_options = list(schema["mappings"]["furnishing_transform"].keys())
        furnishing = st.selectbox("Furnishing", options=furn_options)

        bedrooms = st.number_input("Bedrooms", min_value=0, value=1, step=1)
        bathrooms = st.number_input("Bathrooms", min_value=0, value=1, step=1)

    st.markdown("---")

    # Section 2 — Amenities
    st.markdown(
        "<div class='section-heading'>"
        "<span class='eyebrow'>Amenities</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    lux_list = schema["lists"]["amenities"]["luxury"]
    std_list = schema["lists"]["amenities"]["standard"]
    all_amenities: list[str] = lux_list + std_list

    amenity_inputs: dict[str, bool] = {}
    am_cols = st.columns(3, gap="small")
    for i, am in enumerate(all_amenities):
        label = AMENITY_LABELS.get(am, am.replace("_", " ").title())
        with am_cols[i % 3]:
            amenity_inputs[am] = st.checkbox(label, key=f"am_{am}")

    st.markdown("<br>", unsafe_allow_html=True)

    # Submit
    submitted = st.form_submit_button(
        "Generate Valuation →", use_container_width=True
    )

# ── Submission handler ────────────────────────────────────────────────────────
if submitted:
    if bedrooms == 0 and bathrooms == 0:
        st.error("Provide at least one bedroom or bathroom.")
        st.stop()

    # Build payload
    payload: dict = {
        "house_type": prop_type,
        "condition": condition,
        "furnishing": furnishing,
        "loc": loc,
        "bathrooms": int(bathrooms),
        "bedrooms": int(bedrooms),
    }
    for am, checked in amenity_inputs.items():
        payload[am] = 1 if checked else 0

    with st.spinner("Running valuation model…"):
        try:
            resp = requests.post(f"{BACKEND_URL}/predict", json=payload, timeout=10)
            resp.raise_for_status()
            result = resp.json()
        except requests.HTTPError as e:
            st.error(f"API returned an error: {e.response.status_code}")
            st.stop()
        except Exception as e:
            st.error(f"Could not reach the backend: {e}")
            st.stop()

    # Persist result + inputs for the Report page
    st.session_state.prediction_result = result
    st.session_state.form_inputs = {
        "location": loc,
        "property_type": prop_type,
        "condition": condition,
        "furnishing": furnishing,
        "bedrooms": int(bedrooms),
        "bathrooms": int(bathrooms),
        "amenities": {am: v for am, v in amenity_inputs.items() if v},
    }
    st.session_state.scroll_uid += 1

    # Navigate to Report page
    st.switch_page("pages/Report.py")
