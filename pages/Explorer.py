from dataclasses import dataclass

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils import (
    AMENITY_LABELS,
    BAR_COLOR,
    BAR_DIM,
    CHART_CFG,
    GRID_COLOR,
    PAGE_HOME,
    PAGE_PREDICTOR,
    PLOTLY_LAYOUT,
    RED,
    inject_styles,
    load_market_data,
    metric_bar_html,
    page_note,
    section_heading,
)

ALL_OPTION = "All"


@dataclass(frozen=True)
class FilterOptions:
    locations: list[str]
    property_types: list[str]
    conditions: list[str]
    furnishings: list[str]
    price_min: int
    price_max: int
    beds_max: int
    baths_max: int
    amenities: list[str]


@dataclass(frozen=True)
class FilterState:
    location: str
    property_type: str
    condition: str
    furnishing: str
    beds: tuple[int, int]
    baths: tuple[int, int]
    price: tuple[int, int]
    amenities: list[str]


@dataclass(frozen=True)
class PriceSummary:
    listing_count: int
    q25: float
    q50: float
    q75: float
    iqr: float
    std: float


def configure_page() -> None:
    st.set_page_config(
        page_title="ARES · Market Explorer",
        page_icon="🔍",
        layout="centered",
        initial_sidebar_state="collapsed",
    )
    inject_styles()


def render_navigation() -> None:
    left_col, right_col = st.columns(2, gap="small")
    with left_col:
        if st.button("← Home", key="ex_home", use_container_width=True):
            st.switch_page(PAGE_HOME)
    with right_col:
        if st.button("Predictor →", key="ex_pred", use_container_width=True):
            st.switch_page(PAGE_PREDICTOR)


def render_intro() -> None:
    st.markdown("## Market Explorer")
    st.markdown(
        "<p style='color:var(--t2);font-size:0.9rem;margin-top:-0.4rem;margin-bottom:0;'>"
        "Filter the dataset and explore rental price distributions across Greater Accra."
        "</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")


def load_data() -> pd.DataFrame:
    data = load_market_data()
    if data is None:
        st.error(
            "Market data unavailable. "
            "Set the `DATA_PATH` environment variable to point to `preprocessed_train.csv`."
        )
        st.stop()
    return data


def build_filter_options(df: pd.DataFrame) -> FilterOptions:
    return FilterOptions(
        locations=sorted(df["loc"].dropna().unique().tolist()),
        property_types=sorted(df["house_type"].dropna().unique().tolist()),
        conditions=sorted(df["condition"].dropna().unique().tolist()),
        furnishings=sorted(df["furnishing"].dropna().unique().tolist()),
        price_min=int(df["price"].min()),
        price_max=int(df["price"].quantile(0.995)),
        beds_max=int(df["bedrooms"].max()),
        baths_max=int(df["bathrooms"].max()),
        amenities=[label for label in AMENITY_LABELS if label in df.columns],
    )


def render_filter_form(options: FilterOptions) -> FilterState:
    st.markdown('<div class="filter-panel">', unsafe_allow_html=True)
    section_heading("Filters")
    with st.form("explorer_filters", border=False):
        row_1_col_1, row_1_col_2, row_1_col_3, row_1_col_4 = st.columns(4, gap="small")
        with row_1_col_1:
            selected_location = st.selectbox(
                "Location", [ALL_OPTION, *options.locations]
            )
        with row_1_col_2:
            selected_property_type = st.selectbox(
                "Property Type", [ALL_OPTION, *options.property_types]
            )
        with row_1_col_3:
            selected_condition = st.selectbox(
                "Condition", [ALL_OPTION, *options.conditions]
            )
        with row_1_col_4:
            selected_furnishing = st.selectbox(
                "Furnishing", [ALL_OPTION, *options.furnishings]
            )

        row_2_col_1, row_2_col_2, row_2_col_3 = st.columns(3, gap="small")
        with row_2_col_1:
            selected_beds = st.slider(
                "Bedrooms", 0, options.beds_max, (0, options.beds_max)
            )
        with row_2_col_2:
            selected_baths = st.slider(
                "Bathrooms", 0, options.baths_max, (0, options.baths_max)
            )
        with row_2_col_3:
            selected_price = st.slider(
                "Price Range (₵/mo)",
                options.price_min,
                options.price_max,
                (options.price_min, options.price_max),
            )

        selected_amenities = st.multiselect(
            "Required Amenities",
            options=options.amenities,
            format_func=lambda amenity: AMENITY_LABELS.get(amenity, amenity),
            placeholder="Select amenities to require…",
        )

        apply_col, _ = st.columns([1, 4])
        with apply_col:
            st.form_submit_button("Apply Filters", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)
    return FilterState(
        location=selected_location,
        property_type=selected_property_type,
        condition=selected_condition,
        furnishing=selected_furnishing,
        beds=selected_beds,
        baths=selected_baths,
        price=selected_price,
        amenities=selected_amenities,
    )


def apply_filters(df: pd.DataFrame, filters: FilterState) -> pd.DataFrame:
    filtered = df.copy()

    if filters.location != ALL_OPTION:
        filtered = filtered[filtered["loc"] == filters.location]
    if filters.property_type != ALL_OPTION:
        filtered = filtered[filtered["house_type"] == filters.property_type]
    if filters.condition != ALL_OPTION:
        filtered = filtered[filtered["condition"] == filters.condition]
    if filters.furnishing != ALL_OPTION:
        filtered = filtered[filtered["furnishing"] == filters.furnishing]

    filtered = filtered[filtered["bedrooms"].between(filters.beds[0], filters.beds[1])]
    filtered = filtered[
        filtered["bathrooms"].between(filters.baths[0], filters.baths[1])
    ]
    filtered = filtered[filtered["price"].between(filters.price[0], filters.price[1])]

    for amenity in filters.amenities:
        if amenity in filtered.columns:
            filtered = filtered[filtered[amenity] == 1]

    return filtered


def summarize_prices(df: pd.DataFrame) -> PriceSummary:
    q25 = df["price"].quantile(0.25)
    q50 = df["price"].median()
    q75 = df["price"].quantile(0.75)
    return PriceSummary(
        listing_count=len(df),
        q25=q25,
        q50=q50,
        q75=q75,
        iqr=q75 - q25,
        std=df["price"].std(),
    )


def render_summary_metrics(summary: PriceSummary) -> None:
    st.markdown(
        metric_bar_html(
            [
                ("Listings", f"{summary.listing_count:,}"),
                ("Median Rent", f"₵{summary.q50:,.0f}"),
                ("IQR", f"₵{summary.q25:,.0f} – ₵{summary.q75:,.0f}"),
                ("Std. Dev.", f"₵{summary.std:,.0f}"),
            ]
        ),
        unsafe_allow_html=True,
    )


def render_overview_tab(df: pd.DataFrame, summary: PriceSummary) -> None:
    section_heading("Price Distribution")
    page_note(f"{summary.listing_count:,} listings · IQR fences shown as dashed lines.")

    p99 = df["price"].quantile(0.99)
    hist_df = df[df["price"] <= p99]
    fence_low = max(summary.q25 - 1.5 * summary.iqr, df["price"].min())
    fence_high = min(summary.q75 + 1.5 * summary.iqr, p99)

    dist_fig = px.histogram(
        hist_df,
        x="price",
        nbins=35,
        labels={"price": "Monthly Rent (₵)"},
        color_discrete_sequence=[BAR_COLOR],
    )
    dist_fig.add_vline(
        x=fence_low,
        line_dash="dash",
        line_color=BAR_DIM,
        line_width=1.5,
        annotation_text="Lower fence",
        annotation_font_size=10,
        annotation_font_color=BAR_DIM,
        annotation_position="top left",
    )
    dist_fig.add_vline(
        x=fence_high,
        line_dash="dash",
        line_color=BAR_DIM,
        line_width=1.5,
        annotation_text="Upper fence",
        annotation_font_size=10,
        annotation_font_color=BAR_DIM,
        annotation_position="top right",
    )
    dist_fig.add_vline(
        x=summary.q50,
        line_dash="dot",
        line_color=RED,
        line_width=1.5,
        annotation_text="Median",
        annotation_font_size=10,
        annotation_font_color=RED,
        annotation_position="top right",
    )
    dist_fig.update_layout(
        **PLOTLY_LAYOUT,
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
    st.plotly_chart(dist_fig, use_container_width=True, config=CHART_CFG)

    section_heading("Listing Volume by Location")
    page_note("Top 15 locations within filtered results.")

    location_volume = (
        df.groupby("loc")["price"]
        .agg(count="count", median="median")
        .reset_index()
        .nlargest(15, "count")
        .sort_values("count")
    )
    volume_fig = go.Figure(
        go.Bar(
            x=location_volume["count"],
            y=location_volume["loc"].str.title(),
            orientation="h",
            marker_color=BAR_COLOR,
            hovertemplate="<b>%{y}</b><br>Listings: %{x:,}<extra></extra>",
        )
    )
    volume_fig.update_layout(
        **PLOTLY_LAYOUT,
        height=380,
        xaxis=dict(
            showgrid=True,
            gridcolor=GRID_COLOR,
            title="Listings",
            title_font_size=11,
        ),
        yaxis=dict(showgrid=False, title=None, tickfont=dict(size=11)),
    )
    st.plotly_chart(volume_fig, use_container_width=True, config=CHART_CFG)


def render_compact_bar(
    df: pd.DataFrame,
    group_column: str,
    title: str,
    column: object,
) -> None:
    grouped = (
        df.groupby(group_column)["price"]
        .agg(count="count", median="median")
        .reset_index()
        .query("count >= 3")
        .sort_values("median")
    )
    fig = go.Figure(
        go.Bar(
            x=grouped["median"],
            y=grouped[group_column].str.title(),
            orientation="h",
            marker_color=BAR_COLOR,
            hovertemplate="<b>%{y}</b><br>Median: ₵%{x:,.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=max(200, len(grouped) * 30 + 50),
        margin=dict(l=0, r=0, t=36, b=0),
        title=dict(text=title, font_size=11, x=0, xanchor="left"),
        xaxis=dict(
            tickprefix="₵",
            showgrid=True,
            gridcolor=GRID_COLOR,
            title=None,
            tickfont_size=10,
        ),
        yaxis=dict(showgrid=False, title=None, tickfont_size=10),
    )
    with column:
        st.plotly_chart(fig, use_container_width=True, config=CHART_CFG)


def render_segments_tab(df: pd.DataFrame) -> None:
    section_heading("Median vs. Mean Rent by Location")
    page_note(
        "Top 15 locations. Gap between median and mean signals skew from luxury listings."
    )

    location_stats = (
        df.groupby("loc")["price"]
        .agg(count="count", median="median", mean="mean")
        .reset_index()
        .query("count >= 5")
        .nlargest(15, "count")
        .sort_values("median")
    )
    compare_fig = go.Figure()
    compare_fig.add_trace(
        go.Bar(
            name="Median",
            x=location_stats["median"],
            y=location_stats["loc"].str.title(),
            orientation="h",
            marker_color=BAR_COLOR,
            hovertemplate="<b>%{y}</b><br>Median: ₵%{x:,.0f}<extra></extra>",
        )
    )
    compare_fig.add_trace(
        go.Bar(
            name="Mean",
            x=location_stats["mean"],
            y=location_stats["loc"].str.title(),
            orientation="h",
            marker_color=BAR_DIM,
            hovertemplate="<b>%{y}</b><br>Mean: ₵%{x:,.0f}<extra></extra>",
        )
    )
    compare_fig.update_layout(
        **PLOTLY_LAYOUT,
        height=400,
        barmode="group",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.0,
            xanchor="right",
            x=1,
            font_size=11,
        ),
        xaxis=dict(tickprefix="₵", showgrid=True, gridcolor=GRID_COLOR, title=None),
        yaxis=dict(showgrid=False, title=None, tickfont=dict(size=11)),
    )
    st.plotly_chart(compare_fig, use_container_width=True, config=CHART_CFG)

    col_1, col_2, col_3 = st.columns(3, gap="medium")
    render_compact_bar(df, "house_type", "By Property Type", col_1)
    render_compact_bar(df, "furnishing", "By Furnishing", col_2)
    render_compact_bar(df, "condition", "By Condition", col_3)


def render_listings_tab(df: pd.DataFrame, listing_count: int) -> None:
    section_heading("Filtered Listings")
    page_note(
        f"Showing up to 500 of {listing_count:,} filtered listings, sorted by price."
    )

    display_columns = {
        "loc": "Location",
        "house_type": "Type",
        "bedrooms": "Beds",
        "bathrooms": "Baths",
        "condition": "Condition",
        "furnishing": "Furnishing",
        "price": "Rent (₵/mo)",
    }
    display_frame = (
        df[list(display_columns)]
        .rename(columns=display_columns)
        .sort_values("Rent (₵/mo)", ascending=False)
        .head(500)
    )
    for column_name in ["Location", "Type", "Condition", "Furnishing"]:
        if column_name in display_frame.columns:
            display_frame[column_name] = display_frame[column_name].str.title()
    display_frame["Rent (₵/mo)"] = display_frame["Rent (₵/mo)"].map("₵{:,.0f}".format)
    st.dataframe(
        display_frame.reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
    )


def main() -> None:
    configure_page()
    render_navigation()
    render_intro()

    full_data = load_data()
    filter_options = build_filter_options(full_data)
    filters = render_filter_form(filter_options)
    filtered_data = apply_filters(full_data, filters)

    if filtered_data.empty:
        st.warning("No listings match the current filters. Try relaxing your criteria.")
        st.stop()

    summary = summarize_prices(filtered_data)
    render_summary_metrics(summary)

    overview_tab, segments_tab, listings_tab = st.tabs(
        ["Overview", "Segments", "Listings"]
    )
    with overview_tab:
        render_overview_tab(filtered_data, summary)
    with segments_tab:
        render_segments_tab(filtered_data)
    with listings_tab:
        render_listings_tab(filtered_data, summary.listing_count)


main()
