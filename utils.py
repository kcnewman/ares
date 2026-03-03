"""
utils.py — Shared helpers, CSS injection, and data utilities for ARES.
Import at the top of every page before any other st.* calls.
"""

import os
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

# ---------------------------------------------------------------------------
# Runtime config
# ---------------------------------------------------------------------------
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
DATA_PATH = os.getenv("DATA_PATH", "artifacts/data/preprocessed_train.csv")

# Amenity display labels (preserves key used for API payload)
AMENITY_LABELS: dict[str, str] = {
    "24_hour_electricity": "24h Electricity",
    "air_conditioning": "Air Conditioning",
    "balcony": "Balcony",
    "chandelier": "Chandelier",
    "dining_area": "Dining Area",
    "dishwasher": "Dishwasher",
    "hot_water": "Hot Water",
    "kitchen_cabinets": "Kitchen Cabinets",
    "kitchen_shelf": "Kitchen Shelf",
    "microwave": "Microwave",
    "pop_ceiling": "POP Ceiling",
    "pre_paid_meter": "Prepaid Meter",
    "refrigerator": "Refrigerator",
    "tv": "TV",
    "tiled_floor": "Tiled Floor",
    "wardrobe": "Wardrobe",
    "wi_fi": "Wi-Fi",
    "apartment": "Apartment",
}

# ---------------------------------------------------------------------------
# Typography + Design tokens
# ---------------------------------------------------------------------------
_FONT_LINK = (
    "<link rel='preconnect' href='https://fonts.googleapis.com'>"
    "<link rel='preconnect' href='https://fonts.gstatic.com' crossorigin>"
    "<link href='https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;0,600;1,400"
    "&family=Manrope:wght@400;500;600;700&display=swap' rel='stylesheet'>"
)

_CSS = """
<style>
/* ── Design tokens ─────────────────────────────── */
:root {
  --bg:         #f8f8f7;
  --surface:    #ffffff;
  --border:     #e4e4e7;
  --border-2:   #d4d4d8;
  --text-1:     #18181b;
  --text-2:     #52525b;
  --text-3:     #a1a1aa;
  --accent:     #18181b;
  --green:      #16a34a;
  --red:        #dc2626;
  --amber:      #d97706;
  --radius:     8px;
  --radius-lg:  12px;
  --shadow-sm:  0 1px 2px rgba(0,0,0,.06);
  --shadow:     0 1px 3px rgba(0,0,0,.08), 0 1px 2px rgba(0,0,0,.04);
  --shadow-md:  0 4px 12px rgba(0,0,0,.07), 0 2px 4px rgba(0,0,0,.04);
}

/* ── Base ──────────────────────────────────────── */
html, body, [data-testid="stApp"] {
  font-family: 'Manrope', sans-serif !important;
  background-color: var(--bg) !important;
  color: var(--text-1) !important;
}
* { box-sizing: border-box; }

/* ── Hide chrome ───────────────────────────────── */
#MainMenu, footer,
[data-testid="stDecoration"],
[data-testid="stSidebarNavItems"] { display: none !important; }

/* ── Sidebar ───────────────────────────────────── */
[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border) !important;
}

/* ── Block container ───────────────────────────── */
.main .block-container {
  padding-top: 2.5rem !important;
  padding-bottom: 4rem !important;
  max-width: 820px !important;
}

/* ── Typography ────────────────────────────────── */
h1, h2, h3 {
  font-family: 'Lora', serif !important;
  font-weight: 600 !important;
  color: var(--text-1) !important;
  line-height: 1.25 !important;
}
h1 { font-size: 1.9rem !important; letter-spacing: -0.02em; }
h2 { font-size: 1.35rem !important; }
h3 { font-size: 1.05rem !important; }

.eyebrow {
  font-size: 0.68rem;
  font-weight: 700;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--text-3);
}

/* ── Inputs ────────────────────────────────────── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stNumberInput"] > div > div > input {
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  font-family: 'Manrope', sans-serif !important;
  font-size: 0.9rem !important;
  background: var(--surface) !important;
  color: var(--text-1) !important;
}
[data-testid="stSelectbox"] > div > div:focus-within,
[data-testid="stNumberInput"] > div > div > input:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px rgba(24,24,27,.08) !important;
}

/* ── Widget labels ─────────────────────────────── */
label[data-testid="stWidgetLabel"] p {
  font-size: 0.75rem !important;
  font-weight: 700 !important;
  color: var(--text-2) !important;
  text-transform: uppercase !important;
  letter-spacing: 0.06em !important;
  margin-bottom: 0.3rem !important;
}

/* ── Checkboxes ────────────────────────────────── */
[data-testid="stCheckbox"] { margin-bottom: -6px !important; }
[data-testid="stCheckbox"] label p {
  font-size: 0.83rem !important;
  font-weight: 500 !important;
  text-transform: none !important;
  letter-spacing: 0 !important;
  color: var(--text-1) !important;
}

/* ── Buttons ───────────────────────────────────── */
div.stButton > button,
div.stFormSubmitButton > button {
  height: 3rem !important;
  background: var(--accent) !important;
  color: #fff !important;
  border: none !important;
  border-radius: var(--radius) !important;
  font-family: 'Manrope', sans-serif !important;
  font-size: 0.82rem !important;
  font-weight: 700 !important;
  letter-spacing: 0.08em !important;
  text-transform: uppercase !important;
  transition: background .15s, transform .1s !important;
  cursor: pointer !important;
  padding: 0 1.5rem !important;
}
div.stButton > button:hover,
div.stFormSubmitButton > button:hover {
  background: #000 !important;
  transform: translateY(-1px) !important;
}
/* Ghost / secondary button override */
div.stButton > button[kind="secondary"] {
  background: transparent !important;
  color: var(--text-2) !important;
  border: 1px solid var(--border-2) !important;
}
div.stButton > button[kind="secondary"]:hover {
  background: var(--bg) !important;
  color: var(--text-1) !important;
  transform: none !important;
}

/* ── Tabs ──────────────────────────────────────── */
[data-testid="stTabs"] [data-testid="stTab"] {
  font-family: 'Manrope', sans-serif !important;
  font-size: 0.78rem !important;
  font-weight: 700 !important;
  letter-spacing: 0.07em !important;
  text-transform: uppercase !important;
  color: var(--text-3) !important;
  padding: 0.6rem 1.1rem !important;
  border: none !important;
  background: transparent !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
  color: var(--text-1) !important;
  border-bottom: 2px solid var(--text-1) !important;
}
[data-testid="stTabs"] > div:first-child {
  border-bottom: 1px solid var(--border) !important;
  margin-bottom: 1.5rem !important;
}

/* ── Divider ───────────────────────────────────── */
hr {
  border: none !important;
  border-top: 1px solid var(--border) !important;
  margin: 1.75rem 0 !important;
}

/* ── Alerts ────────────────────────────────────── */
[data-testid="stAlert"] {
  border-radius: var(--radius) !important;
  font-size: 0.875rem !important;
}

/* ── Info chips grid ───────────────────────────── */
.chip-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
  gap: 0.6rem;
  margin: 1rem 0;
}
.chip {
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 0.65rem 0.85rem;
}
.chip .chip-label {
  font-size: 0.62rem;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--text-3);
  font-weight: 700;
  margin-bottom: 0.2rem;
}
.chip .chip-value {
  font-size: 0.88rem;
  font-weight: 600;
  color: var(--text-1);
}

/* ── Result card ───────────────────────────────── */
.result-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 2.5rem 2rem 2rem;
  text-align: center;
  box-shadow: var(--shadow-md);
  margin-bottom: 2rem;
}
.result-card .price {
  font-family: 'Lora', serif !important;
  font-size: 3.2rem;
  font-weight: 600;
  letter-spacing: -0.02em;
  color: var(--text-1);
  line-height: 1.1;
  margin: 0.4rem 0 1.6rem;
}
.result-card .metric-row {
  display: flex;
  justify-content: center;
  gap: 3rem;
  padding-top: 1.25rem;
  border-top: 1px solid var(--border);
}
.result-card .metric-label {
  font-size: 0.65rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--text-3);
  font-weight: 700;
  margin-bottom: 0.3rem;
}
.result-card .metric-value {
  font-size: 0.95rem;
  font-weight: 700;
  color: var(--text-1);
}

/* ── Section heading ───────────────────────────── */
.section-heading {
  display: flex;
  align-items: center;
  gap: 0.6rem;
  margin: 1.75rem 0 1rem;
}
.section-heading::after {
  content: '';
  flex: 1;
  height: 1px;
  background: var(--border);
}

/* ── Back link ─────────────────────────────────── */
.back-link {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  font-size: 0.78rem;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--text-3);
  text-decoration: none;
  margin-bottom: 1.5rem;
  cursor: pointer;
}
.back-link:hover { color: var(--text-1); }
</style>
"""


def inject_styles() -> None:
    """Inject shared Google Fonts link + global CSS. Call once per page."""
    st.markdown(_FONT_LINK, unsafe_allow_html=True)
    st.markdown(_CSS, unsafe_allow_html=True)


def scroll_to_top(uid: int = 0) -> None:
    """Smoothly scrolls the Streamlit main pane to the top."""
    components.html(
        f"<script>//{uid}\n"
        "window.parent.document.querySelector('.main').scrollTo({top:0,behavior:'smooth'});"
        "</script>",
        height=0,
    )


def result_card_html(
    price: float,
    low: float,
    high: float,
    vol: float,
) -> str:
    """Return the HTML string for the centred result card."""
    vol_color = "var(--green)" if vol < 0.10 else "var(--red)"
    vol_label = "Stable" if vol < 0.10 else "Volatile"
    return f"""
    <div class="result-card">
      <div class="eyebrow">Estimated Market Rent</div>
      <div class="price">₵{price:,.2f}<span style="font-size:1rem;font-weight:400;color:var(--text-3)"> /mo</span></div>
      <div class="metric-row">
        <div>
          <div class="metric-label">Price Range</div>
          <div class="metric-value">₵{low:,.0f} – ₵{high:,.0f}</div>
        </div>
        <div>
          <div class="metric-label">Market Volatility</div>
          <div class="metric-value" style="color:{vol_color};">{vol*100:.2f}% · {vol_label}</div>
        </div>
      </div>
    </div>
    """


def chip_grid_html(chips: list[tuple[str, str]]) -> str:
    """Return HTML for a responsive grid of info chips."""
    inner = "".join(
        f'<div class="chip"><div class="chip-label">{lbl}</div>'
        f'<div class="chip-value">{val}</div></div>'
        for lbl, val in chips
    )
    return f'<div class="chip-grid">{inner}</div>'


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_market_data() -> pd.DataFrame | None:
    """Load training CSV for market context charts. Returns None on failure."""
    try:
        df = pd.read_csv(DATA_PATH)
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        return df.dropna(subset=["price"])
    except Exception:
        return None


def check_api() -> bool:
    import requests as _req
    try:
        r = _req.get(f"{BACKEND_URL}/health", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def load_schema() -> dict | None:
    import json
    try:
        with open("artifacts/cache/schema.json") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
