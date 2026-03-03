"""
pages/Predictor.py — Prediction Form
Property input form → POST /predict → navigate to Report page.
Includes nav buttons for Explorer and last Report (if available).
"""

import requests
import streamlit as st

from utils import (
    AMENITY_LABELS,
    BACKEND_URL,
    PAGE_EXPLORER,
    PAGE_REPORT,
    check_api,
    inject_styles,
    load_schema,
    section_heading,
)

st.set_page_config(
    page_title="ARES · Predictor",
    page_icon="🏠",
    layout="centered",
    initial_sidebar_state="collapsed",
)
inject_styles()

# ── Session state defaults ────────────────────────────────────────────────────
for key, default in [
    ("scroll_uid", 0),
    ("prediction_result", None),
    ("form_inputs", {}),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Nav bar ───────────────────────────────────────────────────────────────────
nc1, nc2 = st.columns(2, gap="small")
with nc1:
    if st.button("← Explorer", key="pred_expl", use_container_width=True):
        st.switch_page(PAGE_EXPLORER)
with nc2:
    has_report = st.session_state.prediction_result is not None
    if st.button(
        "Last Report →",
        key="pred_report",
        use_container_width=True,
        disabled=not has_report,
    ):
        st.switch_page(PAGE_REPORT)
    st.caption(
        "Run a valuation to enable this."
        if not has_report
        else "Opens your most recent valuation."
    )

st.markdown("## Predictor")
st.markdown(
    "<p style='color:var(--t2);font-size:0.9rem;margin-top:-0.4rem;margin-bottom:0;'>"
    "Enter property details to run the ML valuation model."
    "</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

# ── API / Schema guard ────────────────────────────────────────────────────────
api_ok = check_api()
schema = load_schema()

if not api_ok:
    st.error(
        "⚠️ Backend API is offline. Start the FastAPI service to use the predictor."
    )
    st.stop()

if not schema:
    st.error("⚠️ Schema file missing at the configured SCHEMA_PATH. Check your setup.")
    st.stop()

# ── Form ──────────────────────────────────────────────────────────────────────
with st.form("valuation_form", border=False):
    section_heading("Property Details")

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
    section_heading("Amenities")

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
    submitted = st.form_submit_button("Generate Valuation →", use_container_width=True)

# ── Submit handler ────────────────────────────────────────────────────────────
if submitted:
    if bedrooms == 0 and bathrooms == 0:
        st.error("Provide at least one bedroom or bathroom.")
        st.stop()

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
            st.error(f"API error {e.response.status_code}. Check backend logs.")
            st.stop()
        except Exception as e:
            st.error(f"Could not reach the backend: {e}")
            st.stop()

    st.session_state.prediction_result = result
    st.session_state.form_inputs = {
        "location": loc,
        "property_type": prop_type,
        "condition": condition,
        "furnishing": furnishing,
        "bedrooms": int(bedrooms),
        "bathrooms": int(bathrooms),
        "amenities": {am: v for am, v in amenity_inputs.items() if v},
        "generated_at": __import__("datetime")
        .datetime.now()
        .strftime("%d %b %Y · %H:%M"),
    }
    st.session_state.scroll_uid += 1

    st.switch_page(PAGE_REPORT)
