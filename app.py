import streamlit as st
import streamlit.components.v1 as components
import requests
import json
import os


BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")


# --- Helper Functions ---
def check_api():
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=2)
        return response.status_code == 200
    except Exception:
        return False


def load_schema():
    try:
        with open("artifacts/cache/schema.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


# --- Config ---
st.set_page_config(page_title="Accra Rent Estimator", layout="centered")

if "scroll_count" not in st.session_state:
    st.session_state.scroll_count = 0

schema = load_schema()

# --- CSS ---
st.markdown(
    """
    <style>
    label { font-weight: 500 !important; margin-bottom: 0.5rem !important; }
    div.stButton > button {
        width: 100% !important; height: 3.5rem !important;
        background-color: #09090b !important; color: white !important;
        border-radius: 6px !important; font-size: 1.1rem !important;
        font-weight: 600 !important; transition: all 0.2s ease !important;
    }
    div.stButton > button:hover { background-color: #000000 !important; transform: translateY(-1px); }
    .res-card { border: 1px solid #e4e4e7; padding: 1.5rem; border-radius: 8px; text-align: center; margin-bottom: 2rem; }
    [data-testid="stCheckbox"] { margin-bottom: -15px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div id='top'></div>", unsafe_allow_html=True)
st.title("Accra Rent Valuation System")
st.caption(
    "A machine learning powered rent valuation with market uncertainty bands for the Greater Accraaa Region of Ghana."
)

result_placeholder = st.empty()

# --- Main Logic Gate ---
api_online = check_api()

if not api_online:
    st.error("Backend API is offline. Please restart the service to use the valuator.")
elif not schema:
    st.error("Schema file missing. Please check artifacts/cache/schema.json")
else:
    with st.form("valuation_form", border=False):
        st.markdown("### Core Specifications")
        c1, c2 = st.columns(2)

        with c1:
            loc_options = sorted(list(schema["mappings"]["location_class"].keys()))
            loc = st.selectbox("Location", options=loc_options)
            prop_options = list(schema["mappings"]["property_density"].keys())
            prop_type = st.selectbox("Property Type", options=prop_options)
            cond_options = list(schema["mappings"]["condition_transform"].keys())
            condition = st.selectbox("Condition", options=cond_options)

        with c2:
            furn_options = list(schema["mappings"]["furnishing_transform"].keys())
            furnishing = st.selectbox("Furnishing", options=furn_options)
            bedrooms = st.number_input("Bedrooms", min_value=0, value=1)
            bathrooms = st.number_input("Bathrooms", min_value=0, value=1)

        st.markdown("---")
        st.markdown("### Amenities")
        lux_list = schema["lists"]["amenities"]["luxury"]
        std_list = schema["lists"]["amenities"]["standard"]
        all_amenities = lux_list + std_list

        am_cols = st.columns(4)
        amenity_inputs = {}  # Store inputs here instead of session_state
        for i, am in enumerate(all_amenities):
            with am_cols[i % 4]:
                label = am.replace("_", " ").title()
                amenity_inputs[am] = st.checkbox(label)

        st.markdown("<br>", unsafe_allow_html=True)
        submit_btn = st.form_submit_button(
            "Predict Valuation", use_container_width=True
        )

    if submit_btn:
        if bedrooms == 0 and bathrooms == 0:
            st.error("Please provide at least one bedroom or bathroom.")
        else:
            # Construct Payload
            payload = {
                "house_type": prop_type,
                "condition": condition,
                "furnishing": furnishing,
                "loc": loc,
                "bathrooms": bathrooms,
                "bedrooms": bedrooms,
            }
            for am, val in amenity_inputs.items():
                payload[am] = 1 if val else 0

            st.session_state.scroll_count += 1
            components.html(
                f"<script>//{st.session_state.scroll_count}\nwindow.parent.document.getElementById('top').scrollIntoView({{behavior: 'smooth'}});</script>",
                height=0,
            )

            try:
                response = requests.post(f"{BACKEND_URL}/predict", json=payload)
                response.raise_for_status()
                result = response.json()

                est_price = result.get("estimated_price", 0)
                low_b = result.get("lower_band", 0)
                high_b = result.get("upper_band", 0)
                vol = result.get("market_volatility_idx", 0)

                vol_color = "#22c55e" if vol < 0.10 else "#f51b0b"

                result_placeholder.markdown(
                    f"""
                    <div class="res-card">
                        <p style="opacity: 0.7; text-transform: uppercase; font-size: 0.8rem; letter-spacing: 0.1rem;">Estimated Market Value</p>
                        <h1 style="font-size: 2.5rem; margin: 0;">₵{est_price:,.2f}</h1>
                        <div style="display: flex; justify-content: center; gap: 40px; margin-top: 15px; border-top: 1px solid #8883; padding-top: 15px;">
                            <div><small style="opacity: 0.6;">Range</small><br><strong>₵{low_b:,.0f} - ₵{high_b:,.0f}</strong></div>
                            <div><small style="opacity: 0.6;">Market Volatility</small><br><strong style="color: {vol_color};">{vol * 100:,.2f}%</strong></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            except Exception as e:
                st.error(f"API Error: {e}")
