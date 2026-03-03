from __future__ import annotations

import json
import os

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components

PAGE_HOME = "app.py"
PAGE_EXPLORER = "pages/Explorer.py"
PAGE_PREDICTOR = "pages/Predictor.py"
PAGE_REPORT = "pages/Report.py"

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
DATA_PATH = os.getenv("DATA_PATH", "artifacts/data_processing/preprocessed_train.csv")
SCHEMA_PATH = os.getenv("SCHEMA_PATH", "artifacts/cache/schema.json")

AMENITY_LABELS: dict[str, str] = {
    "24_hour_electricity": "24h Electricity",
    "air_conditioning": "Air Conditioning",
    "apartment": "Apartment",
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
}
AMENITY_COLS = list(AMENITY_LABELS.keys())

PLOTLY_LAYOUT = dict(
    font_family="Manrope, sans-serif",
    font_color="#52525b",
    font_size=11,
    plot_bgcolor="#ffffff",
    paper_bgcolor="#ffffff",
    margin=dict(l=0, r=0, t=28, b=0),
    showlegend=False,
    hoverlabel=dict(
        bgcolor="#18181b",
        font_color="#ffffff",
        font_family="Manrope, sans-serif",
        font_size=11,
    ),
)
GRID_COLOR = "#f0f0ee"
BAR_COLOR = "#18181b"
BAR_DIM = "#d4d4d8"
RED = "#dc2626"
CHART_CFG = {"displayModeBar": False}

_FONT_LINK = (
    "<link rel='preconnect' href='https://fonts.googleapis.com'>"
    "<link rel='preconnect' href='https://fonts.gstatic.com' crossorigin>"
    "<link href='https://fonts.googleapis.com/css2?family=Lora:wght@400;600"
    "&family=Manrope:wght@400;500;600;700&display=swap' rel='stylesheet'>"
)

_CSS = """
<style>
:root {
  --bg:      #f8f8f7;
  --surface: #ffffff;
  --bd:      #e4e4e7;
  --bd2:     #d4d4d8;
  --t1:      #18181b;
  --t2:      #52525b;
  --t3:      #a1a1aa;
  --acc:     #18181b;
  --green:   #16a34a;
  --red:     #dc2626;
  --amber:   #d97706;
  --r:       8px;
  --rl:      12px;
}
html,body,[data-testid="stApp"] {
  font-family:'Manrope',sans-serif !important;
  background:var(--bg) !important;
  color:var(--t1) !important;
}
* { box-sizing:border-box; }

/* hide chrome */
#MainMenu,footer,
[data-testid="stDecoration"],
[data-testid="stSidebarNavItems"],
[data-testid="stSidebarNav"],
[data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"] { display:none !important; }
.main .block-container {
  padding-top:2.25rem !important;
  padding-bottom:4rem !important;
  max-width:860px !important;
}

/* typography */
h1,h2,h3 {
  font-family:'Lora',serif !important;
  font-weight:600 !important;
  color:var(--t1) !important;
  line-height:1.25 !important;
}
h1 { font-size:1.85rem !important; letter-spacing:-0.02em; }
h2 { font-size:1.3rem !important; }
h3 { font-size:1rem !important; }
.eyebrow {
  font-size:0.67rem; font-weight:700;
  letter-spacing:0.13em; text-transform:uppercase; color:var(--t3);
}

/* inputs */
[data-testid="stSelectbox"] > div > div,
[data-testid="stNumberInput"] input {
  border:1px solid var(--bd) !important;
  border-radius:var(--r) !important;
  font-family:'Manrope',sans-serif !important;
  font-size:0.875rem !important;
  background:var(--surface) !important;
  color:var(--t1) !important;
}
[data-testid="stSelectbox"] > div > div:focus-within,
[data-testid="stNumberInput"] input:focus {
  border-color:var(--t2) !important;
  box-shadow:0 0 0 3px rgba(24,24,27,.07) !important;
  outline:none !important;
}
[data-testid="stMultiSelect"] > div > div {
  border:1px solid var(--bd) !important;
  border-radius:var(--r) !important;
  background:var(--surface) !important;
  font-size:0.875rem !important;
}
[data-testid="stMultiSelect"] > div > div:focus-within {
  border-color:var(--t2) !important;
  box-shadow:0 0 0 3px rgba(24,24,27,.07) !important;
}

/* widget labels */
label[data-testid="stWidgetLabel"] p {
  font-size:0.72rem !important;
  font-weight:700 !important;
  color:var(--t2) !important;
  text-transform:uppercase !important;
  letter-spacing:0.06em !important;
  margin-bottom:0.25rem !important;
}

/* checkboxes */
[data-testid="stCheckbox"] { margin-bottom:-4px !important; }
[data-testid="stCheckbox"] label p {
  font-size:0.83rem !important;
  font-weight:500 !important;
  text-transform:none !important;
  letter-spacing:0 !important;
}

/* buttons */
div.stButton > button,
div.stFormSubmitButton > button {
  height:2.75rem !important;
  background:var(--acc) !important;
  color:#fff !important;
  border:none !important;
  border-radius:var(--r) !important;
  font-family:'Manrope',sans-serif !important;
  font-size:0.78rem !important;
  font-weight:700 !important;
  letter-spacing:0.08em !important;
  text-transform:uppercase !important;
  transition:background .12s,transform .1s !important;
  padding:0 1.4rem !important;
  cursor:pointer !important;
}
div.stButton > button:hover,
div.stFormSubmitButton > button:hover {
  background:#000 !important;
  transform:translateY(-1px) !important;
}
div.stButton > button[kind="secondary"] {
  background:transparent !important;
  color:var(--t2) !important;
  border:1px solid var(--bd2) !important;
}
div.stButton > button[kind="secondary"]:hover {
  background:var(--bg) !important;
  color:var(--t1) !important;
  transform:none !important;
}

/* tabs */
[data-testid="stTabs"] [data-testid="stTab"] {
  font-family:'Manrope',sans-serif !important;
  font-size:0.75rem !important;
  font-weight:700 !important;
  letter-spacing:0.07em !important;
  text-transform:uppercase !important;
  color:var(--t3) !important;
  padding:0.55rem 1rem !important;
  border:none !important;
  background:transparent !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
  color:var(--t1) !important;
  border-bottom:2px solid var(--t1) !important;
}
[data-testid="stTabs"] > div:first-child {
  border-bottom:1px solid var(--bd) !important;
  gap:0 !important;
  margin-bottom:1.25rem !important;
}

/* dataframe */
[data-testid="stDataFrame"] {
  border:1px solid var(--bd) !important;
  border-radius:var(--r) !important;
  overflow:hidden;
}

/* alerts */
[data-testid="stAlert"] { border-radius:var(--r) !important; font-size:0.85rem !important; }

/* divider */
hr { border:none !important; border-top:1px solid var(--bd) !important; margin:1.5rem 0 !important; }

/* chip grid */
.chip-grid {
  display:grid;
  grid-template-columns:repeat(auto-fill,minmax(130px,1fr));
  gap:0.5rem; margin:0.75rem 0 1rem;
}
.chip { background:var(--bg); border:1px solid var(--bd); border-radius:var(--r); padding:0.6rem 0.8rem; }
.chip-label { font-size:0.6rem; letter-spacing:0.1em; text-transform:uppercase; color:var(--t3); font-weight:700; margin-bottom:0.18rem; }
.chip-value { font-size:0.85rem; font-weight:600; color:var(--t1); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }

/* home stat row */
.stat-row { display:flex; gap:0.75rem; margin:1rem 0 1.5rem; }
.stat-item { flex:1; background:var(--surface); border:1px solid var(--bd); border-radius:var(--r); padding:0.9rem 1.1rem; }
.stat-item .sl { font-size:0.62rem; letter-spacing:0.1em; text-transform:uppercase; color:var(--t3); font-weight:700; margin-bottom:0.2rem; }
.stat-item .sv { font-size:1.05rem; font-weight:700; color:var(--t1); }
.stat-item .sv.online { color:var(--green); }
.stat-item .sv.offline { color:var(--red); }

/* workflow card */
.workflow-card { border:1px solid var(--bd); border-radius:var(--rl); background:var(--surface); padding:1.25rem 1.5rem; margin:0.75rem 0 1.5rem; }
.workflow-steps { display:flex; }
.workflow-step { flex:1; padding:0.6rem 1rem; border-right:1px solid var(--bd); }
.workflow-step:first-child { padding-left:0; }
.workflow-step:last-child { border-right:none; }
.ws-num { font-size:0.62rem; font-weight:700; letter-spacing:0.1em; text-transform:uppercase; color:var(--t3); margin-bottom:0.25rem; }
.ws-title { font-size:0.88rem; font-weight:700; color:var(--t1); margin-bottom:0.2rem; }
.ws-desc { font-size:0.76rem; color:var(--t2); line-height:1.45; }

/* section heading */
.sh { display:flex; align-items:center; gap:0.6rem; margin:1.5rem 0 1rem; }
.sh::after { content:''; flex:1; height:1px; background:var(--bd); }

/* explorer metric bar */
.metric-bar { display:flex; gap:0.75rem; margin:0.25rem 0 1.25rem; flex-wrap:wrap; }
.mc { flex:1; min-width:100px; background:var(--surface); border:1px solid var(--bd); border-radius:var(--r); padding:0.7rem 1rem; }
.mc-l { font-size:0.6rem; letter-spacing:0.1em; text-transform:uppercase; color:var(--t3); font-weight:700; margin-bottom:0.18rem; }
.mc-v { font-size:0.95rem; font-weight:700; color:var(--t1); }

/* result card */
.result-card {
  background:var(--surface); border:1px solid var(--bd); border-radius:var(--rl);
  padding:2.25rem 2rem 1.75rem; text-align:center;
  box-shadow:0 4px 12px rgba(0,0,0,.07),0 2px 4px rgba(0,0,0,.04); margin-bottom:1.5rem;
}
.result-card .price {
  font-family:'Lora',serif; font-size:2.9rem; font-weight:600;
  letter-spacing:-0.02em; color:var(--t1); line-height:1.1; margin:0.35rem 0 1.5rem;
}
.metric-row { display:flex; justify-content:center; gap:2.5rem; padding-top:1.1rem; border-top:1px solid var(--bd); flex-wrap:wrap; }
.metric-row > div { text-align:center; }
.ml { font-size:0.62rem; letter-spacing:0.12em; text-transform:uppercase; color:var(--t3); font-weight:700; margin-bottom:0.25rem; }
.mv { font-size:0.9rem; font-weight:700; color:var(--t1); }

/* insight box */
.ib { background:var(--bg); border:1px solid var(--bd); border-radius:var(--r); padding:1.1rem 1.4rem; margin-bottom:1rem; }
.ib-title { font-size:0.68rem; font-weight:700; letter-spacing:0.1em; text-transform:uppercase; color:var(--t3); margin-bottom:0.6rem; }
.ig { display:grid; grid-template-columns:repeat(auto-fill,minmax(115px,1fr)); gap:0.6rem 1.25rem; }
.ii-l { font-size:0.6rem; letter-spacing:0.08em; text-transform:uppercase; color:var(--t3); font-weight:700; margin-bottom:0.12rem; }
.ii-v { font-size:0.87rem; font-weight:600; color:var(--t1); }

/* report metadata */
.report-meta { font-size:0.71rem; color:var(--t3); text-align:center; margin:0 0 1.5rem; }

/* filter panel */
.filter-panel { background:var(--surface); border:1px solid var(--bd); border-radius:var(--rl); padding:1.25rem 1.5rem 0.75rem; margin-bottom:1rem; }
</style>
"""


def inject_styles() -> None:
    st.markdown(_FONT_LINK, unsafe_allow_html=True)
    st.markdown(_CSS, unsafe_allow_html=True)


def scroll_to_top(uid: int = 0) -> None:
    components.html(
        f"<script>//{uid}\n"
        "window.parent.document.querySelector('.main')"
        ".scrollTo({top:0,behavior:'smooth'});</script>",
        height=0,
    )


def chip_grid_html(chips: list[tuple[str, str]]) -> str:
    inner = "".join(
        f'<div class="chip"><div class="chip-label">{lbl}</div>'
        f'<div class="chip-value" title="{val}">{val}</div></div>'
        for lbl, val in chips
    )
    return f'<div class="chip-grid">{inner}</div>'


def metric_bar_html(metrics: list[tuple[str, str]]) -> str:
    inner = "".join(
        f'<div class="mc"><div class="mc-l">{lbl}</div>'
        f'<div class="mc-v">{val}</div></div>'
        for lbl, val in metrics
    )
    return f'<div class="metric-bar">{inner}</div>'


def stat_row_html(stats: list[tuple[str, str, str]]) -> str:
    """(label, value, extra_class)"""
    inner = "".join(
        f'<div class="stat-item"><div class="sl">{lbl}</div>'
        f'<div class="sv {cls}">{val}</div></div>'
        for lbl, val, cls in stats
    )
    return f'<div class="stat-row">{inner}</div>'


def workflow_card_html() -> str:
    steps = [
        (
            "01 · Explore",
            "Market Explorer",
            "Browse 19k+ listings with live filters and segment breakdowns.",
        ),
        (
            "02 · Predict",
            "Run Predictor",
            "Input property details and run the ML valuation model.",
        ),
        (
            "03 · Review",
            "Valuation Report",
            "Estimated rent with uncertainty band and market context.",
        ),
    ]
    inner = "".join(
        f'<div class="workflow-step"><div class="ws-num">{n}</div>'
        f'<div class="ws-title">{t}</div><div class="ws-desc">{d}</div></div>'
        for n, t, d in steps
    )
    return (
        '<div class="workflow-card">'
        '<div class="eyebrow" style="margin-bottom:0.6rem;">How it works</div>'
        f'<div class="workflow-steps">{inner}</div></div>'
    )


def result_card_html(
    price: float,
    low: float,
    high: float,
    vol: float,
    seg_median: float | None = None,
) -> str:
    vol_color = "var(--green)" if vol < 0.10 else "var(--red)"
    vol_label = "Stable" if vol < 0.10 else "Volatile"
    metrics = [
        f'<div><div class="ml">Price Range</div>'
        f'<div class="mv">₵{low:,.0f} – ₵{high:,.0f}</div></div>',
        f'<div><div class="ml">Market Volatility</div>'
        f'<div class="mv" style="color:{vol_color};">{vol * 100:.2f}% · {vol_label}</div></div>',
    ]
    if seg_median is not None and seg_median > 0:
        delta = price - seg_median
        delta_pct = delta / seg_median * 100
        sign = "+" if delta >= 0 else ""
        d_color = "var(--red)" if delta >= 0 else "var(--green)"
        metrics.append(
            f'<div><div class="ml">vs. Segment Median</div>'
            f'<div class="mv" style="color:{d_color};">{sign}{delta_pct:.1f}%</div></div>'
        )
    return (
        '<div class="result-card">'
        '<div class="eyebrow">Estimated Market Rent</div>'
        f'<div class="price">₵{price:,.2f}'
        '<span style="font-size:1rem;font-weight:400;color:var(--t3)"> /mo</span></div>'
        f'<div class="metric-row">{"".join(metrics)}</div></div>'
    )


def insight_box_html(
    seg_label: str,
    seg_n: int,
    seg_median: float,
    seg_q25: float,
    seg_q75: float,
    est_price: float,
    confidence: str,
) -> str:
    delta = est_price - seg_median
    delta_pct = delta / seg_median * 100 if seg_median > 0 else 0
    sign = "+" if delta >= 0 else ""
    d_color = "var(--red)" if delta >= 0 else "var(--green)"
    c_color = {
        "High": "var(--green)",
        "Moderate": "var(--amber)",
        "Low": "var(--red)",
    }.get(confidence, "inherit")
    items = [
        ("Baseline Segment", seg_label),
        ("Segment Size", f"{seg_n:,} listings"),
        ("Segment Median", f"₵{seg_median:,.0f}"),
        ("IQR", f"₵{seg_q25:,.0f} – ₵{seg_q75:,.0f}"),
        (
            "Delta vs Median",
            f'<span style="color:{d_color};">{sign}{delta_pct:.1f}%</span>',
        ),
        ("Confidence", f'<span style="color:{c_color};">{confidence}</span>'),
    ]
    inner = "".join(
        f'<div class="insight-item"><div class="ii-l">{lbl}</div>'
        f'<div class="ii-v">{val}</div></div>'
        for lbl, val in items
    )
    return (
        f'<div class="ib"><div class="ib-title">Segment Context</div>'
        f'<div class="ig">{inner}</div></div>'
    )


def section_heading(label: str) -> None:
    st.markdown(
        f"<div class='sh'><span class='eyebrow'>{label}</span></div>",
        unsafe_allow_html=True,
    )


def page_note(text: str) -> None:
    st.markdown(
        f"<p style='color:var(--t2);font-size:0.82rem;margin:-0.25rem 0 0.75rem;'>{text}</p>",
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_market_data() -> pd.DataFrame | None:
    try:
        df = pd.read_csv(DATA_PATH)
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        return df.dropna(subset=["price"])
    except (
        FileNotFoundError,
        OSError,
        KeyError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ):
        return None


def check_api() -> bool:
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=2)
        return response.status_code == 200
    except requests.RequestException:
        return False


def load_schema() -> dict | None:
    try:
        with open(SCHEMA_PATH, encoding="utf-8") as file:
            return json.load(file)
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None


def compute_segment(
    df: pd.DataFrame,
    loc: str,
    prop_type: str,
    furnishing: str,
) -> tuple[pd.DataFrame, str]:
    """
    Tightest segment >= 5 listings.
    Fallback: (loc+type+furn) → (loc+type) → (loc) → full df.
    """
    candidates = [
        (
            df[
                (df["loc"] == loc)
                & (df["house_type"] == prop_type)
                & (df["furnishing"] == furnishing)
            ],
            f"{furnishing.title()} {prop_type.title()} in {loc.title()}",
        ),
        (
            df[(df["loc"] == loc) & (df["house_type"] == prop_type)],
            f"{prop_type.title()} in {loc.title()}",
        ),
        (df[df["loc"] == loc], f"All types in {loc.title()}"),
        (df, "All Greater Accra"),
    ]
    for seg_df, label in candidates:
        if len(seg_df) >= 5:
            return seg_df.copy(), label
    return df.copy(), "All Greater Accra"


def confidence_tier(n: int) -> str:
    if n >= 50:
        return "High"
    if n >= 15:
        return "Moderate"
    return "Low"
