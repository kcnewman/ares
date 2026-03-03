from datetime import datetime
from typing import Any

import requests
import streamlit as st
from requests import RequestException

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


def configure_page() -> None:
    st.set_page_config(
        page_title="ARES · Predictor",
        page_icon="🏠",
        layout="centered",
        initial_sidebar_state="collapsed",
    )
    inject_styles()


def init_session_state() -> None:
    st.session_state.setdefault("scroll_uid", 0)
    st.session_state.setdefault("prediction_result", None)
    st.session_state.setdefault("form_inputs", {})


def render_navigation() -> None:
    left_col, right_col = st.columns(2, gap="small")
    with left_col:
        if st.button("← Explorer", key="pred_expl", use_container_width=True):
            st.switch_page(PAGE_EXPLORER)
    with right_col:
        has_report = st.session_state.get("prediction_result") is not None
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


def render_intro() -> None:
    st.markdown("## Predictor")
    st.markdown(
        "<p style='color:var(--t2);font-size:0.9rem;margin-top:-0.4rem;margin-bottom:0;'>"
        "Enter property details to run the ML valuation model."
        "</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")


def load_runtime_schema() -> dict[str, Any]:
    if not check_api():
        st.error(
            "⚠️ Backend API is offline. Start the FastAPI service to use the predictor."
        )
        st.stop()

    schema = load_schema()
    if schema is None:
        st.error(
            "⚠️ Schema file missing at the configured SCHEMA_PATH. Check your setup."
        )
        st.stop()

    return schema


def render_form(
    schema: dict[str, Any],
) -> tuple[bool, dict[str, str | int], dict[str, bool]]:
    with st.form("valuation_form", border=False):
        section_heading("Property Details")

        left_col, right_col = st.columns(2, gap="medium")
        with left_col:
            location_options = sorted(schema["mappings"]["location_class"])
            location = st.selectbox("Location", options=location_options)

            property_options = list(schema["mappings"]["property_density"])
            property_type = st.selectbox("Property Type", options=property_options)

            condition_options = list(schema["mappings"]["condition_transform"])
            condition = st.selectbox("Condition", options=condition_options)

        with right_col:
            furnishing_options = list(schema["mappings"]["furnishing_transform"])
            furnishing = st.selectbox("Furnishing", options=furnishing_options)
            bedrooms = int(st.number_input("Bedrooms", min_value=0, value=1, step=1))
            bathrooms = int(st.number_input("Bathrooms", min_value=0, value=1, step=1))

        st.markdown("---")
        section_heading("Amenities")

        luxury_amenities = schema["lists"]["amenities"]["luxury"]
        standard_amenities = schema["lists"]["amenities"]["standard"]
        all_amenities = [*luxury_amenities, *standard_amenities]

        amenity_inputs: dict[str, bool] = {}
        amenity_columns = st.columns(3, gap="small")
        for index, amenity in enumerate(all_amenities):
            label = AMENITY_LABELS.get(amenity, amenity.replace("_", " ").title())
            with amenity_columns[index % 3]:
                amenity_inputs[amenity] = st.checkbox(label, key=f"am_{amenity}")

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button(
            "Generate Valuation →", use_container_width=True
        )

    form_data: dict[str, str | int] = {
        "location": location,
        "property_type": property_type,
        "condition": condition,
        "furnishing": furnishing,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
    }
    return submitted, form_data, amenity_inputs


def build_payload(
    form_data: dict[str, str | int],
    amenity_inputs: dict[str, bool],
) -> dict[str, int | str]:
    payload: dict[str, int | str] = {
        "house_type": str(form_data["property_type"]),
        "condition": str(form_data["condition"]),
        "furnishing": str(form_data["furnishing"]),
        "loc": str(form_data["location"]),
        "bathrooms": int(form_data["bathrooms"]),
        "bedrooms": int(form_data["bedrooms"]),
    }
    payload.update({key: int(value) for key, value in amenity_inputs.items()})
    return payload


def call_predict_api(payload: dict[str, int | str]) -> dict[str, Any]:
    with st.spinner("Running valuation model…"):
        try:
            response = requests.post(f"{BACKEND_URL}/predict", json=payload, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as exc:
            status_code = exc.response.status_code if exc.response else "unknown"
            st.error(f"API error {status_code}. Check backend logs.")
            st.stop()
        except RequestException as exc:
            st.error(f"Could not reach the backend: {exc}")
            st.stop()

    raise RuntimeError("Prediction flow stopped before returning a response.")


def store_result(
    result: dict[str, Any],
    form_data: dict[str, str | int],
    amenity_inputs: dict[str, bool],
) -> None:
    st.session_state.prediction_result = result
    st.session_state.form_inputs = {
        "location": form_data["location"],
        "property_type": form_data["property_type"],
        "condition": form_data["condition"],
        "furnishing": form_data["furnishing"],
        "bedrooms": int(form_data["bedrooms"]),
        "bathrooms": int(form_data["bathrooms"]),
        "amenities": {key: value for key, value in amenity_inputs.items() if value},
        "generated_at": datetime.now().strftime("%d %b %Y · %H:%M"),
    }
    st.session_state.scroll_uid += 1


def main() -> None:
    configure_page()
    init_session_state()
    render_navigation()
    render_intro()

    schema = load_runtime_schema()
    submitted, form_data, amenity_inputs = render_form(schema)

    if not submitted:
        return

    if int(form_data["bedrooms"]) == 0 and int(form_data["bathrooms"]) == 0:
        st.error("Provide at least one bedroom or bathroom.")
        st.stop()

    payload = build_payload(form_data, amenity_inputs)
    result = call_predict_api(payload)
    store_result(result, form_data, amenity_inputs)
    st.switch_page(PAGE_REPORT)


main()
