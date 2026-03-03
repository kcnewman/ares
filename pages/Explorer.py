from dataclasses import dataclass

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.delta_generator as st_dg

from utils import (
    AMENITY_LABELS,
    BAR_COLOR,
    BAR_DIM,
    CHART_CFG,
    FULL_DATA_URL,
    confidence_tier,
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
LISTINGS_SAMPLE_SIZE = 500
LISTINGS_SAMPLE_RANDOM_STATE = 42


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
            format_func=lambda amenity: str(AMENITY_LABELS.get(amenity, amenity)),
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


def chart_gap(height: str = "1.25rem") -> None:
    st.markdown(f'<div style="height:{height};"></div>', unsafe_allow_html=True)


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
    chart_gap()

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
    chart_gap("1.1rem")

    st.markdown("---")
    chart_gap("0.5rem")
    section_heading("Advanced Insights")
    page_note(
        "Derived from current filters. Segment metrics use sample thresholds to avoid noisy results."
    )

    render_amenity_premium_chart(df)
    chart_gap()
    render_unit_economics(df)
    chart_gap()
    render_segment_volatility(df)
    chart_gap()
    render_furnishing_premium(df)
    chart_gap()
    render_opportunity_matrix(df)
    chart_gap()
    render_bed_bath_heatmap(df)


def render_amenity_premium_chart(df: pd.DataFrame) -> None:
    section_heading("Amenity Premium")
    page_note(
        "Median rent uplift for listings with each amenity versus listings without it. "
        "Minimum 25 listings in each group."
    )

    premium_rows: list[dict[str, float | int | str]] = []
    for amenity_column, amenity_label in AMENITY_LABELS.items():
        if amenity_column not in df.columns:
            continue

        with_amenity = df[df[amenity_column] == 1]["price"]
        without_amenity = df[df[amenity_column] == 0]["price"]
        if len(with_amenity) < 25 or len(without_amenity) < 25:
            continue

        median_with = float(with_amenity.median())
        median_without = float(without_amenity.median())
        if median_without <= 0:
            continue

        uplift_abs = median_with - median_without
        uplift_pct = ((median_with / median_without) - 1) * 100
        premium_rows.append(
            {
                "amenity": amenity_label,
                "uplift_abs": uplift_abs,
                "uplift_pct": uplift_pct,
                "with_count": int(len(with_amenity)),
                "without_count": int(len(without_amenity)),
            }
        )

    if not premium_rows:
        st.info("Not enough data to estimate amenity premiums for the current filters.")
        return

    premium_df = pd.DataFrame(premium_rows)
    premium_df = premium_df.reindex(
        premium_df["uplift_pct"].abs().sort_values(ascending=False).index
    ).head(12)
    premium_df = premium_df.sort_values("uplift_pct")

    amenity_fig = go.Figure(
        go.Bar(
            x=premium_df["uplift_pct"],
            y=premium_df["amenity"],
            orientation="h",
            marker_color=[
                BAR_COLOR if uplift >= 0 else RED for uplift in premium_df["uplift_pct"]
            ],
            customdata=premium_df[["uplift_abs", "with_count", "without_count"]],
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Uplift: %{x:.1f}%<br>"
                "Median delta: ₵%{customdata[0]:,.0f}<br>"
                "With amenity: %{customdata[1]:,}<br>"
                "Without amenity: %{customdata[2]:,}<extra></extra>"
            ),
        )
    )
    amenity_fig.update_layout(
        **PLOTLY_LAYOUT,
        height=max(260, len(premium_df) * 28 + 70),
        xaxis=dict(
            title="Median Rent Uplift (%)",
            ticksuffix="%",
            showgrid=True,
            gridcolor=GRID_COLOR,
            zeroline=True,
            zerolinecolor=BAR_DIM,
        ),
        yaxis=dict(showgrid=False, title=None),
    )
    st.plotly_chart(amenity_fig, use_container_width=True, config=CHART_CFG)


def render_unit_economics(df: pd.DataFrame) -> None:
    section_heading("Unit Economics")
    page_note(
        "Median rent normalized by bedroom and bathroom counts across locations and "
        "property types."
    )

    unit_df = df[(df["bedrooms"] > 0) & (df["bathrooms"] > 0)].copy()
    if unit_df.empty:
        st.info("Not enough valid bedroom/bathroom data to compute unit economics.")
        return

    unit_df["price_per_bedroom"] = unit_df["price"] / unit_df["bedrooms"]
    unit_df["price_per_bathroom"] = unit_df["price"] / unit_df["bathrooms"]

    loc_stats = (
        unit_df.groupby("loc")
        .agg(
            count=("price", "count"),
            pp_bed=("price_per_bedroom", "median"),
            pp_bath=("price_per_bathroom", "median"),
        )
        .reset_index()
        .query("count >= 8")
        .nlargest(10, "count")
    )
    type_stats = (
        unit_df.groupby("house_type")
        .agg(
            count=("price", "count"),
            pp_bed=("price_per_bedroom", "median"),
            pp_bath=("price_per_bathroom", "median"),
        )
        .reset_index()
        .query("count >= 8")
        .nlargest(10, "count")
    )

    left_col, right_col = st.columns(2, gap="medium")

    with left_col:
        st.markdown(
            "<p style='color:var(--t2);font-size:0.76rem;margin:0 0 0.45rem;'>"
            "Top locations by listing count</p>",
            unsafe_allow_html=True,
        )
        if loc_stats.empty:
            st.info("No location-level sample met the minimum threshold.")
        else:
            loc_plot = (
                loc_stats.assign(
                    segment=loc_stats["loc"].str.title(),
                    **{
                        "Price / Bedroom": loc_stats["pp_bed"],
                        "Price / Bathroom": loc_stats["pp_bath"],
                    },
                )[["segment", "count", "Price / Bedroom", "Price / Bathroom"]]
                .melt(
                    id_vars=["segment", "count"],
                    value_vars=["Price / Bedroom", "Price / Bathroom"],
                    var_name="metric",
                    value_name="value",
                )
            )
            segment_order = loc_stats.sort_values("pp_bed")["loc"].str.title().tolist()
            loc_fig = px.bar(
                loc_plot,
                x="value",
                y="segment",
                color="metric",
                orientation="h",
                barmode="group",
                category_orders={"segment": segment_order},
                color_discrete_map={
                    "Price / Bedroom": BAR_COLOR,
                    "Price / Bathroom": BAR_DIM,
                },
            )
            loc_fig.update_layout(
                **PLOTLY_LAYOUT,
                height=340,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.0,
                    xanchor="right",
                    x=1,
                    font_size=10,
                ),
                xaxis=dict(
                    tickprefix="₵",
                    title=None,
                    showgrid=True,
                    gridcolor=GRID_COLOR,
                ),
                yaxis=dict(showgrid=False, title=None, tickfont_size=10),
            )
            st.plotly_chart(loc_fig, use_container_width=True, config=CHART_CFG)

    with right_col:
        st.markdown(
            "<p style='color:var(--t2);font-size:0.76rem;margin:0 0 0.45rem;'>"
            "Top property types by listing count</p>",
            unsafe_allow_html=True,
        )
        if type_stats.empty:
            st.info("No property-type sample met the minimum threshold.")
        else:
            type_plot = (
                type_stats.assign(
                    segment=type_stats["house_type"].str.title(),
                    **{
                        "Price / Bedroom": type_stats["pp_bed"],
                        "Price / Bathroom": type_stats["pp_bath"],
                    },
                )[["segment", "count", "Price / Bedroom", "Price / Bathroom"]]
                .melt(
                    id_vars=["segment", "count"],
                    value_vars=["Price / Bedroom", "Price / Bathroom"],
                    var_name="metric",
                    value_name="value",
                )
            )
            segment_order = (
                type_stats.sort_values("pp_bed")["house_type"].str.title().tolist()
            )
            type_fig = px.bar(
                type_plot,
                x="value",
                y="segment",
                color="metric",
                orientation="h",
                barmode="group",
                category_orders={"segment": segment_order},
                color_discrete_map={
                    "Price / Bedroom": BAR_COLOR,
                    "Price / Bathroom": BAR_DIM,
                },
            )
            type_fig.update_layout(
                **PLOTLY_LAYOUT,
                height=340,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.0,
                    xanchor="right",
                    x=1,
                    font_size=10,
                ),
                xaxis=dict(
                    tickprefix="₵",
                    title=None,
                    showgrid=True,
                    gridcolor=GRID_COLOR,
                ),
                yaxis=dict(showgrid=False, title=None, tickfont_size=10),
            )
            st.plotly_chart(type_fig, use_container_width=True, config=CHART_CFG)


def render_segment_volatility(df: pd.DataFrame) -> None:
    section_heading("Segment Volatility Score")
    page_note("Volatility score = IQR / median rent by location + property type segment.")

    segment_stats = (
        df.groupby(["loc", "house_type"])["price"]
        .agg(
            count="count",
            median="median",
            q25=lambda series: series.quantile(0.25),
            q75=lambda series: series.quantile(0.75),
        )
        .reset_index()
    )
    segment_stats = segment_stats[(segment_stats["count"] >= 8) & (segment_stats["median"] > 0)]
    if segment_stats.empty:
        st.info("Not enough segment depth to compute volatility scores.")
        return

    segment_stats["volatility_pct"] = (
        (segment_stats["q75"] - segment_stats["q25"]) / segment_stats["median"]
    ) * 100
    segment_stats["segment"] = (
        segment_stats["loc"].str.title() + " · " + segment_stats["house_type"].str.title()
    )
    segment_stats = (
        segment_stats.nlargest(12, "count")
        .sort_values("volatility_pct")
        .reset_index(drop=True)
    )

    volatility_fig = go.Figure(
        go.Bar(
            x=segment_stats["volatility_pct"],
            y=segment_stats["segment"],
            orientation="h",
            marker_color=BAR_COLOR,
            customdata=segment_stats[["count", "median"]],
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Volatility: %{x:.1f}%<br>"
                "Median rent: ₵%{customdata[1]:,.0f}<br>"
                "Listings: %{customdata[0]:,}<extra></extra>"
            ),
        )
    )
    volatility_fig.update_layout(
        **PLOTLY_LAYOUT,
        height=max(280, len(segment_stats) * 26 + 85),
        xaxis=dict(
            title="Volatility Score (IQR / Median, %)",
            ticksuffix="%",
            showgrid=True,
            gridcolor=GRID_COLOR,
        ),
        yaxis=dict(showgrid=False, title=None, tickfont_size=10),
    )
    st.plotly_chart(volatility_fig, use_container_width=True, config=CHART_CFG)


def render_furnishing_premium(df: pd.DataFrame) -> None:
    section_heading("Furnishing Premium by Location")
    page_note(
        "Median rent difference between furnished and unfurnished listings. "
        "Minimum 5 listings in each furnishing state."
    )

    furnishing_df = df[df["furnishing"].isin(["furnished", "unfurnished"])].copy()
    if furnishing_df.empty:
        st.info("No furnished/unfurnished listings available in the current filters.")
        return

    median_pivot = furnishing_df.pivot_table(
        index="loc", columns="furnishing", values="price", aggfunc="median"
    )
    count_pivot = furnishing_df.pivot_table(
        index="loc", columns="furnishing", values="price", aggfunc="size", fill_value=0
    )
    if "furnished" not in median_pivot.columns or "unfurnished" not in median_pivot.columns:
        st.info("Need both furnished and unfurnished samples to compute premium.")
        return

    premium_df = pd.DataFrame(index=median_pivot.index)
    premium_df["furnished_median"] = median_pivot["furnished"]
    premium_df["unfurnished_median"] = median_pivot["unfurnished"]
    premium_df["furnished_count"] = count_pivot["furnished"]
    premium_df["unfurnished_count"] = count_pivot["unfurnished"]
    premium_df = premium_df.dropna()
    premium_df = premium_df[
        (premium_df["furnished_count"] >= 5)
        & (premium_df["unfurnished_count"] >= 5)
        & (premium_df["unfurnished_median"] > 0)
    ]
    if premium_df.empty:
        st.info("No locations met the sample threshold for furnishing premium.")
        return

    premium_df["premium_pct"] = (
        (premium_df["furnished_median"] / premium_df["unfurnished_median"]) - 1
    ) * 100
    premium_df["premium_abs"] = (
        premium_df["furnished_median"] - premium_df["unfurnished_median"]
    )
    premium_df["total_count"] = (
        premium_df["furnished_count"] + premium_df["unfurnished_count"]
    )
    premium_df = premium_df.nlargest(12, "total_count").sort_values("premium_pct")
    premium_df["location"] = premium_df.index.str.title()

    furnishing_fig = go.Figure(
        go.Bar(
            x=premium_df["premium_pct"],
            y=premium_df["location"],
            orientation="h",
            marker_color=[
                BAR_COLOR if premium >= 0 else RED
                for premium in premium_df["premium_pct"]
            ],
            customdata=premium_df[["premium_abs", "furnished_count", "unfurnished_count"]],
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Premium: %{x:.1f}%<br>"
                "Median delta: ₵%{customdata[0]:,.0f}<br>"
                "Furnished: %{customdata[1]:,}<br>"
                "Unfurnished: %{customdata[2]:,}<extra></extra>"
            ),
        )
    )
    furnishing_fig.update_layout(
        **PLOTLY_LAYOUT,
        height=max(260, len(premium_df) * 28 + 70),
        xaxis=dict(
            title="Furnished vs Unfurnished Premium (%)",
            ticksuffix="%",
            showgrid=True,
            gridcolor=GRID_COLOR,
            zeroline=True,
            zerolinecolor=BAR_DIM,
        ),
        yaxis=dict(showgrid=False, title=None, tickfont_size=10),
    )
    st.plotly_chart(furnishing_fig, use_container_width=True, config=CHART_CFG)


def render_opportunity_matrix(df: pd.DataFrame) -> None:
    section_heading("Segment Opportunity Matrix")
    page_note(
        "Bubbles represent location + property type segments. Larger bubbles indicate "
        "higher average amenity count."
    )

    amenity_columns = [column for column in AMENITY_LABELS if column in df.columns]
    matrix_df = df.copy()
    matrix_df["amenity_score"] = (
        matrix_df[amenity_columns].sum(axis=1) if amenity_columns else 0
    )

    segment_df = (
        matrix_df.groupby(["loc", "house_type"])
        .agg(
            listing_count=("price", "count"),
            median_rent=("price", "median"),
            q25=("price", lambda series: series.quantile(0.25)),
            q75=("price", lambda series: series.quantile(0.75)),
            amenity_score=("amenity_score", "mean"),
        )
        .reset_index()
    )
    segment_df = segment_df[
        (segment_df["listing_count"] >= 8) & (segment_df["median_rent"] > 0)
    ].copy()
    if segment_df.empty:
        st.info("Not enough segment coverage to build the opportunity matrix.")
        return

    segment_df["volatility_score"] = (
        (segment_df["q75"] - segment_df["q25"]) / segment_df["median_rent"]
    )
    segment_df["segment"] = (
        segment_df["loc"].str.title() + " · " + segment_df["house_type"].str.title()
    )
    segment_df = segment_df.nlargest(35, "listing_count")

    matrix_fig = px.scatter(
        segment_df,
        x="median_rent",
        y="listing_count",
        size="amenity_score",
        color="volatility_score",
        hover_name="segment",
        custom_data=["amenity_score"],
        color_continuous_scale=[[0, "#d4d4d8"], [1, "#18181b"]],
        labels={
            "median_rent": "Median Rent (₵)",
            "listing_count": "Listings",
            "volatility_score": "Volatility",
        },
    )
    matrix_fig.update_traces(
        hovertemplate=(
            "<b>%{hovertext}</b><br>"
            "Median rent: ₵%{x:,.0f}<br>"
            "Listings: %{y:,}<br>"
            "Avg. amenities: %{customdata[0]:.2f}<br>"
            "Volatility: %{marker.color:.2f}<extra></extra>"
        )
    )
    matrix_fig.update_layout(
        **PLOTLY_LAYOUT,
        height=520,
        xaxis=dict(tickprefix="₵", showgrid=True, gridcolor=GRID_COLOR, title=None),
        yaxis=dict(showgrid=True, gridcolor=GRID_COLOR, title=None),
        coloraxis_colorbar=dict(title="Volatility"),
    )
    st.plotly_chart(matrix_fig, use_container_width=True, config=CHART_CFG)


def render_bed_bath_heatmap(df: pd.DataFrame) -> None:
    section_heading("Bedroom-Bathroom Price Grid")
    page_note("Median rent by bedroom and bathroom combination. Minimum 8 listings per cell.")

    grid_df = (
        df[(df["bedrooms"] > 0) & (df["bathrooms"] > 0) & (df["bedrooms"] <= 6) & (df["bathrooms"] <= 6)]
        .groupby(["bedrooms", "bathrooms"])["price"]
        .agg(count="count", median="median")
        .reset_index()
        .query("count >= 8")
    )
    if grid_df.empty:
        st.info("No bedroom-bathroom combinations met the sample threshold.")
        return

    grid = (
        grid_df.pivot(index="bedrooms", columns="bathrooms", values="median")
        .sort_index()
        .sort_index(axis=1)
    )
    heatmap_fig = px.imshow(
        grid,
        aspect="auto",
        color_continuous_scale=[[0, "#f0f0ee"], [1, "#18181b"]],
        labels={
            "x": "Bathrooms",
            "y": "Bedrooms",
            "color": "Median Rent (₵)",
        },
    )
    heatmap_fig.update_traces(
        hovertemplate=(
            "Bedrooms: %{y}<br>"
            "Bathrooms: %{x}<br>"
            "Median rent: ₵%{z:,.0f}<extra></extra>"
        )
    )
    heatmap_fig.update_layout(
        **PLOTLY_LAYOUT,
        height=360,
        xaxis=dict(side="top", title=None),
        yaxis=dict(title=None),
    )
    st.plotly_chart(heatmap_fig, use_container_width=True, config=CHART_CFG)


def render_compact_bar(
    df: pd.DataFrame,
    group_column: str,
    title: str,
    column: st_dg.DeltaGenerator,
    note: str | None = None,
) -> None:
    grouped = (
        df.groupby(group_column)["price"]
        .agg(count="count", median="median")
        .reset_index()
        .query("count >= 3")
    )
    if grouped.empty:
        with column:
            st.info(f"No data available for {title.lower()}.")
        return

    if len(grouped) > 8:
        top_categories = grouped.nlargest(8, "count")[group_column]
        grouped = grouped[grouped[group_column].isin(top_categories)]

    grouped = grouped.sort_values("median")
    grouped["confidence"] = grouped["count"].apply(confidence_tier)
    grouped["row_label"] = (
        grouped[group_column].str.title() + " [" + grouped["confidence"] + "]"
    )
    fig = go.Figure(
        go.Bar(
            x=grouped["median"],
            y=grouped["row_label"],
            orientation="h",
            marker_color=BAR_COLOR,
            customdata=grouped[["count", "confidence"]],
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Median: ₵%{x:,.0f}<br>"
                "Listings: %{customdata[0]:,}<br>"
                "Confidence: %{customdata[1]}<extra></extra>"
            ),
        )
    )
    compact_layout = {**PLOTLY_LAYOUT, "margin": dict(l=0, r=0, t=36, b=0)}
    fig.update_layout(
        **compact_layout,
        height=min(300, max(220, len(grouped) * 24 + 65)),
        title=dict(text=title, font_size=11, x=0, xanchor="left"),
        bargap=0.2,
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
        if note:
            st.caption(note)


def build_location_type_segment_stats(
    df: pd.DataFrame, min_count: int = 8
) -> pd.DataFrame:
    segment_stats = (
        df.groupby(["loc", "house_type"])["price"]
        .agg(
            count="count",
            median="median",
            mean="mean",
            q25=lambda series: series.quantile(0.25),
            q75=lambda series: series.quantile(0.75),
        )
        .reset_index()
    )
    segment_stats = segment_stats[
        (segment_stats["count"] >= min_count) & (segment_stats["median"] > 0)
    ].copy()
    if segment_stats.empty:
        return segment_stats

    segment_stats["volatility_pct"] = (
        (segment_stats["q75"] - segment_stats["q25"]) / segment_stats["median"]
    ) * 100
    segment_stats["median_pct"] = segment_stats["median"].rank(pct=True) * 100
    segment_stats["volatility_rank_pct"] = (
        segment_stats["volatility_pct"].rank(pct=True) * 100
    )
    segment_stats["confidence"] = segment_stats["count"].apply(confidence_tier)
    segment_stats["segment"] = (
        segment_stats["loc"].str.title() + " · " + segment_stats["house_type"].str.title()
    )
    return segment_stats


def render_location_skew_stability(df: pd.DataFrame) -> None:
    section_heading("Location Skew & Stability Map")
    page_note(
        "Each bubble is a location. Rightward points indicate stronger luxury skew "
        "(mean above median). Higher points indicate wider price spread."
    )

    location_insights = (
        df.groupby("loc")["price"]
        .agg(
            count="count",
            median="median",
            mean="mean",
            q25=lambda series: series.quantile(0.25),
            q75=lambda series: series.quantile(0.75),
        )
        .reset_index()
    )
    location_insights = location_insights[
        (location_insights["count"] >= 8) & (location_insights["median"] > 0)
    ].copy()
    if location_insights.empty:
        st.info("Not enough location depth to build skew and stability insights.")
        return

    location_insights["skew_pct"] = (
        (location_insights["mean"] - location_insights["median"])
        / location_insights["median"]
    ) * 100
    location_insights["spread_pct"] = (
        (location_insights["q75"] - location_insights["q25"])
        / location_insights["median"]
    ) * 100
    location_insights["confidence"] = location_insights["count"].apply(confidence_tier)
    location_insights["location_name"] = location_insights["loc"].str.title()
    location_insights = location_insights.nlargest(25, "count")

    skew_fig = px.scatter(
        location_insights,
        x="skew_pct",
        y="spread_pct",
        size="count",
        color="median",
        hover_name="location_name",
        custom_data=["count", "median", "confidence"],
        color_continuous_scale=[[0, "#d4d4d8"], [1, "#18181b"]],
        labels={
            "skew_pct": "Skew (Mean-Median as % of Median)",
            "spread_pct": "Spread (IQR as % of Median)",
            "median": "Median Rent",
        },
    )
    skew_fig.update_traces(
        hovertemplate=(
            "<b>%{hovertext}</b><br>"
            "Skew: %{x:.1f}%<br>"
            "Spread: %{y:.1f}%<br>"
            "Listings: %{customdata[0]:,}<br>"
            "Median rent: ₵%{customdata[1]:,.0f}<br>"
            "Confidence: %{customdata[2]}<extra></extra>"
        )
    )
    skew_fig.add_vline(x=0, line_dash="dot", line_color=BAR_DIM, line_width=1)
    skew_fig.update_layout(
        **PLOTLY_LAYOUT,
        height=420,
        xaxis=dict(ticksuffix="%", showgrid=True, gridcolor=GRID_COLOR, title=None),
        yaxis=dict(ticksuffix="%", showgrid=True, gridcolor=GRID_COLOR, title=None),
        coloraxis_colorbar=dict(title="Median Rent"),
    )
    st.plotly_chart(skew_fig, use_container_width=True, config=CHART_CFG)
    st.caption(
        "Read it this way: upper-right locations are both skewed and volatile; "
        "lower-left locations are relatively stable and less skewed."
    )


def render_segment_leaderboard(df: pd.DataFrame) -> None:
    section_heading("Segment Leaderboard")
    page_note(
        "Top and bottom location + property-type segments by median rent "
        "(minimum 8 listings per segment)."
    )

    segment_stats = build_location_type_segment_stats(df)
    if segment_stats.empty:
        st.info("Not enough segment depth to build a leaderboard.")
        return

    highest = segment_stats.nlargest(5, "median").copy()
    highest["tier"] = "Highest Median"
    lowest = segment_stats.nsmallest(5, "median").copy()
    lowest["tier"] = "Lowest Median"

    leaderboard = pd.concat([highest, lowest], ignore_index=True)[
        [
            "tier",
            "segment",
            "count",
            "median",
            "volatility_pct",
            "median_pct",
            "volatility_rank_pct",
            "confidence",
        ]
    ]
    leaderboard = leaderboard.rename(
        columns={
            "tier": "Tier",
            "segment": "Segment",
            "count": "Listings",
            "median": "Median Rent",
            "volatility_pct": "Volatility",
            "median_pct": "Median Percentile",
            "volatility_rank_pct": "Volatility Percentile",
            "confidence": "Confidence",
        }
    )
    leaderboard["Median Rent"] = leaderboard["Median Rent"].map("₵{:,.0f}".format)
    leaderboard["Volatility"] = leaderboard["Volatility"].map("{:.1f}%".format)
    leaderboard["Median Percentile"] = leaderboard["Median Percentile"].map(
        "{:.0f}th".format
    )
    leaderboard["Volatility Percentile"] = leaderboard["Volatility Percentile"].map(
        "{:.0f}th".format
    )

    st.dataframe(leaderboard, hide_index=True, use_container_width=True)
    st.caption(
        "Use this as a shortlist: pair high-median segments with volatility to "
        "avoid segments that are expensive but unstable."
    )


def render_segment_comparison_panel(df: pd.DataFrame) -> None:
    section_heading("Compare Two Segments")
    page_note(
        "Select two location + property-type segments to compare price level, "
        "volatility, and sample confidence."
    )

    segment_stats = build_location_type_segment_stats(df)
    if len(segment_stats) < 2:
        st.info("Need at least two segments with enough listings for comparison.")
        return

    options = segment_stats["segment"].sort_values().tolist()
    left_col, right_col = st.columns(2, gap="medium")
    with left_col:
        segment_a_label = st.selectbox("Segment A", options, key="segment_compare_a")
    with right_col:
        default_index = 1 if len(options) > 1 else 0
        segment_b_label = st.selectbox(
            "Segment B",
            options,
            index=default_index,
            key="segment_compare_b",
        )

    segment_a = segment_stats[segment_stats["segment"] == segment_a_label].iloc[0]
    segment_b = segment_stats[segment_stats["segment"] == segment_b_label].iloc[0]

    median_gap = float(segment_a["median"] - segment_b["median"])
    volatility_gap = float(segment_a["volatility_pct"] - segment_b["volatility_pct"])
    listing_gap = int(segment_a["count"] - segment_b["count"])

    metric_col_1, metric_col_2, metric_col_3 = st.columns(3, gap="small")
    metric_col_1.metric(
        "Median Gap (A-B)",
        f"₵{median_gap:,.0f}",
        delta=f"{median_gap:+,.0f}",
    )
    metric_col_2.metric(
        "Volatility Gap (A-B)",
        f"{volatility_gap:+.1f}%",
        delta=f"{volatility_gap:+.1f} pp",
    )
    metric_col_3.metric(
        "Listing Depth Gap (A-B)",
        f"{listing_gap:+,}",
        delta=f"{listing_gap:+,} listings",
    )

    comparison_table = pd.DataFrame(
        {
            "Metric": [
                "Median Rent",
                "Volatility",
                "Listings",
                "Median Percentile",
                "Volatility Percentile",
                "Confidence",
            ],
            "Segment A": [
                f"₵{segment_a['median']:,.0f}",
                f"{segment_a['volatility_pct']:.1f}%",
                f"{int(segment_a['count']):,}",
                f"{segment_a['median_pct']:.0f}th",
                f"{segment_a['volatility_rank_pct']:.0f}th",
                str(segment_a["confidence"]),
            ],
            "Segment B": [
                f"₵{segment_b['median']:,.0f}",
                f"{segment_b['volatility_pct']:.1f}%",
                f"{int(segment_b['count']):,}",
                f"{segment_b['median_pct']:.0f}th",
                f"{segment_b['volatility_rank_pct']:.0f}th",
                str(segment_b["confidence"]),
            ],
        }
    )
    st.dataframe(comparison_table, hide_index=True, use_container_width=True)
    st.caption(
        "Percentiles are computed across all qualifying segments in the current filter."
    )


def render_segment_percentile_map(df: pd.DataFrame) -> None:
    section_heading("Segment Percentile Position")
    page_note(
        "Position of each segment on median-rent percentile (x-axis) and volatility "
        "percentile (y-axis)."
    )

    segment_stats = build_location_type_segment_stats(df)
    if segment_stats.empty:
        st.info("Not enough segment depth for percentile indicators.")
        return

    percentile_df = segment_stats.nlargest(30, "count").copy()
    percentile_fig = px.scatter(
        percentile_df,
        x="median_pct",
        y="volatility_rank_pct",
        size="count",
        color="confidence",
        hover_name="segment",
        custom_data=["median", "volatility_pct", "count", "confidence"],
        color_discrete_map={
            "High": BAR_COLOR,
            "Moderate": BAR_DIM,
            "Low": RED,
        },
        labels={
            "median_pct": "Median Rent Percentile",
            "volatility_rank_pct": "Volatility Percentile",
        },
    )
    percentile_fig.update_traces(
        hovertemplate=(
            "<b>%{hovertext}</b><br>"
            "Median percentile: %{x:.0f}th<br>"
            "Volatility percentile: %{y:.0f}th<br>"
            "Median rent: ₵%{customdata[0]:,.0f}<br>"
            "Volatility: %{customdata[1]:.1f}%<br>"
            "Listings: %{customdata[2]:,}<br>"
            "Confidence: %{customdata[3]}<extra></extra>"
        )
    )
    percentile_fig.add_vline(x=50, line_dash="dot", line_color=BAR_DIM, line_width=1)
    percentile_fig.add_hline(y=50, line_dash="dot", line_color=BAR_DIM, line_width=1)
    percentile_fig.update_layout(
        **PLOTLY_LAYOUT,
        height=430,
        xaxis=dict(range=[0, 100], ticksuffix="th", showgrid=True, gridcolor=GRID_COLOR),
        yaxis=dict(range=[0, 100], ticksuffix="th", showgrid=True, gridcolor=GRID_COLOR),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.0,
            xanchor="right",
            x=1,
            font_size=10,
            title=None,
        ),
    )
    st.plotly_chart(percentile_fig, use_container_width=True, config=CHART_CFG)
    st.caption(
        "Upper-right segments are both premium-priced and relatively volatile in the "
        "current market slice."
    )


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
    location_stats["confidence"] = location_stats["count"].apply(confidence_tier)
    location_stats["row_label"] = (
        location_stats["loc"].str.title() + " [" + location_stats["confidence"] + "]"
    )
    compare_fig = go.Figure()
    compare_fig.add_trace(
        go.Bar(
            name="Median",
            x=location_stats["median"],
            y=location_stats["row_label"],
            orientation="h",
            marker_color=BAR_COLOR,
            customdata=location_stats[["count", "confidence"]],
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Median: ₵%{x:,.0f}<br>"
                "Listings: %{customdata[0]:,}<br>"
                "Confidence: %{customdata[1]}<extra></extra>"
            ),
        )
    )
    compare_fig.add_trace(
        go.Bar(
            name="Mean",
            x=location_stats["mean"],
            y=location_stats["row_label"],
            orientation="h",
            marker_color=BAR_DIM,
            customdata=location_stats[["count", "confidence"]],
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Mean: ₵%{x:,.0f}<br>"
                "Listings: %{customdata[0]:,}<br>"
                "Confidence: %{customdata[1]}<extra></extra>"
            ),
        )
    )
    compare_layout = {**PLOTLY_LAYOUT, "showlegend": True}
    compare_fig.update_layout(
        **compare_layout,
        height=400,
        barmode="group",
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
    st.caption(
        "If mean is much higher than median, a few expensive listings are pulling "
        "the average upward."
    )
    chart_gap()

    render_location_skew_stability(df)
    chart_gap()
    render_segment_percentile_map(df)
    chart_gap()

    col_1, col_2, col_3 = st.columns(3, gap="medium")
    render_compact_bar(
        df,
        "house_type",
        "By Property Type",
        col_1,
        note="Top categories by listing depth, sorted by median rent.",
    )
    render_compact_bar(
        df,
        "furnishing",
        "By Furnishing",
        col_2,
        note="Shows how furnishing state shifts median pricing.",
    )
    render_compact_bar(
        df,
        "condition",
        "By Condition",
        col_3,
        note="Tracks quality-condition premium across the filtered market.",
    )
    chart_gap()

    render_segment_comparison_panel(df)
    chart_gap()
    render_segment_leaderboard(df)


def render_map_tab(df: pd.DataFrame, listing_count: int) -> None:
    section_heading("Accra Listing Concentration")
    page_note(
        "Interactive map of filtered listings. Switch between hotspot bubbles and "
        "density to explore where supply is clustering."
    )

    required_columns = {"lat", "lng"}
    if not required_columns.issubset(df.columns):
        st.info("Map unavailable: this dataset does not include `lat` and `lng` columns.")
        return

    map_df = df.copy()
    map_df["lat"] = pd.to_numeric(map_df["lat"], errors="coerce")
    map_df["lng"] = pd.to_numeric(map_df["lng"], errors="coerce")
    map_df = map_df.dropna(subset=["lat", "lng"])
    map_df = map_df[map_df["lat"].between(4.8, 6.4) & map_df["lng"].between(-1.0, 0.4)]

    if map_df.empty:
        st.info("No valid coordinates found for the current filter selection.")
        return

    center_lat = float(map_df["lat"].median())
    center_lng = float(map_df["lng"].median())
    coverage_pct = (len(map_df) / listing_count) * 100 if listing_count else 0
    location_points = (
        map_df.groupby("loc")
        .agg(
            listings=("price", "count"),
            median_rent=("price", "median"),
            q25=("price", lambda series: series.quantile(0.25)),
            q75=("price", lambda series: series.quantile(0.75)),
            lat=("lat", "median"),
            lng=("lng", "median"),
            primary_type=(
                "house_type",
                lambda series: (
                    series.dropna().astype(str).str.title().value_counts().index[0]
                    if not series.dropna().empty
                    else "Unknown"
                ),
            ),
            property_types=(
                "house_type",
                lambda series: ", ".join(
                    series.dropna()
                    .astype(str)
                    .str.title()
                    .value_counts()
                    .head(3)
                    .index.tolist()
                ),
            ),
            property_types_count=("house_type", "nunique"),
        )
        .reset_index()
    )
    location_points["primary_type"] = location_points["primary_type"].replace(
        "", "Unknown"
    )
    location_points["property_types"] = location_points["property_types"].replace(
        "", "Unknown"
    )
    location_points["location_label"] = location_points["loc"].str.title()

    st.markdown(
        metric_bar_html(
            [
                ("Mapped Listings", f"{len(map_df):,}"),
                ("Mapped Locations", f"{location_points['loc'].nunique():,}"),
                ("Coverage", f"{coverage_pct:.1f}%"),
                ("Median Mapped Rent", f"₵{map_df['price'].median():,.0f}"),
            ]
        ),
        unsafe_allow_html=True,
    )

    controls_col_1, controls_col_2, controls_col_3 = st.columns(3, gap="small")
    with controls_col_1:
        view_mode = st.radio(
            "Map Layer",
            ["Hotspots", "Density"],
            horizontal=True,
            key="ex_map_view_mode",
        )
    with controls_col_2:
        max_locations = min(45, max(1, len(location_points)))
        hotspot_count = st.slider(
            "Hotspots Shown",
            min_value=1,
            max_value=max_locations,
            value=min(22, max_locations),
            key="ex_map_hotspot_count",
        )
    with controls_col_3:
        density_radius = st.slider(
            "Density Radius",
            min_value=8,
            max_value=30,
            value=18,
            step=1,
            key="ex_map_density_radius",
        )

    map_layout = {
        **PLOTLY_LAYOUT,
        "height": 560,
        "margin": dict(l=0, r=0, t=0, b=0),
        "mapbox": dict(
            style="carto-positron",
            zoom=10.2,
            center={"lat": center_lat, "lon": center_lng},
        ),
    }
    map_config = {**CHART_CFG, "scrollZoom": True}

    if view_mode == "Hotspots":
        hotspot_points = location_points.nlargest(hotspot_count, "listings").copy()
        listing_min = float(hotspot_points["listings"].min())
        listing_max = float(hotspot_points["listings"].max())
        if listing_max == listing_min:
            hotspot_points["marker_size"] = 24.0
        else:
            hotspot_points["marker_size"] = 14 + (
                (hotspot_points["listings"] - listing_min)
                / (listing_max - listing_min)
            ) * 30

        bubble_fig = go.Figure(
            go.Scattermapbox(
                lat=hotspot_points["lat"],
                lon=hotspot_points["lng"],
                mode="markers",
                marker=dict(
                    size=hotspot_points["marker_size"],
                    color=hotspot_points["median_rent"],
                    colorscale="YlOrRd",
                    opacity=0.84,
                    allowoverlap=True,
                    showscale=True,
                    colorbar=dict(
                        title="Median Rent (₵)",
                        tickprefix="₵",
                    ),
                ),
                customdata=hotspot_points[
                    [
                        "location_label",
                        "listings",
                        "median_rent",
                        "q25",
                        "q75",
                        "primary_type",
                        "property_types",
                        "property_types_count",
                    ]
                ],
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Listings: %{customdata[1]:,}<br>"
                    "Median rent: ₵%{customdata[2]:,.0f}<br>"
                    "IQR: ₵%{customdata[3]:,.0f} - ₵%{customdata[4]:,.0f}<br>"
                    "Primary type: %{customdata[5]}<br>"
                    "Top types: %{customdata[6]}<br>"
                    "Type variety: %{customdata[7]}<extra></extra>"
                ),
            )
        )
        bubble_fig.update_layout(**map_layout)
        st.plotly_chart(bubble_fig, use_container_width=True, config=map_config)
        st.caption(
            "Bubble size = listing count, color = median rent. Hover or zoom to inspect hotspots."
        )
    else:
        density_fig = px.density_mapbox(
            map_df,
            lat="lat",
            lon="lng",
            z="price",
            radius=density_radius,
            center={"lat": center_lat, "lon": center_lng},
            zoom=10,
            mapbox_style="carto-positron",
            color_continuous_scale=[
                [0.0, "#fff7bc"],
                [0.35, "#fec44f"],
                [0.65, "#fe9929"],
                [1.0, "#d95f0e"],
            ],
            hover_data={
                "loc": True,
                "house_type": True,
                "price": ":,.0f",
                "lat": ":.4f",
                "lng": ":.4f",
            },
        )
        density_fig.update_layout(
            **map_layout,
            coloraxis_colorbar=dict(
                title="Weighted Density",
                tickfont=dict(size=10),
                titlefont=dict(size=10),
            ),
        )
        st.plotly_chart(density_fig, use_container_width=True, config=map_config)
        st.caption(
            "Density is weighted by listing price. Increase radius for smoother market clusters."
        )

    chart_gap("1.0rem")

    section_heading("Top Hotspots")
    page_note(
        "Locations with the highest number of mapped listings in this filter slice."
    )

    hotspot_df = (
        location_points.sort_values("listings", ascending=False)
        .head(10)
        [["location_label", "listings", "median_rent"]]
    )
    hotspot_df["median_rent"] = hotspot_df["median_rent"].map("₵{:,.0f}".format)
    hotspot_df = hotspot_df.rename(
        columns={
            "location_label": "Location",
            "listings": "Listings",
            "median_rent": "Median Rent",
        }
    )
    st.dataframe(hotspot_df, hide_index=True, use_container_width=True)


def build_representative_sample(
    df: pd.DataFrame,
    sample_size: int = LISTINGS_SAMPLE_SIZE,
    stratify_cols: tuple[str, str] = ("loc", "house_type"),
    random_state: int = LISTINGS_SAMPLE_RANDOM_STATE,
) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    if len(df) <= sample_size:
        return df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    if not set(stratify_cols).issubset(df.columns):
        return df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)

    sampled_df = df.copy()
    sampled_df["_stratum"] = (
        sampled_df[list(stratify_cols)]
        .fillna("Unknown")
        .astype(str)
        .agg(" | ".join, axis=1)
    )

    stratum_sizes = sampled_df["_stratum"].value_counts().sort_index()
    expected = (stratum_sizes / stratum_sizes.sum()) * sample_size
    allocations = expected.astype(int)
    remainder = sample_size - int(allocations.sum())

    if remainder > 0:
        fractional = (expected - allocations).sort_values(ascending=False)
        for stratum_name in fractional.head(remainder).index:
            allocations.loc[stratum_name] += 1

    sampled_parts: list[pd.DataFrame] = []
    for stratum_name, take_count in allocations.items():
        if int(take_count) <= 0:
            continue
        stratum_rows = sampled_df[sampled_df["_stratum"] == stratum_name]
        sampled_parts.append(
            stratum_rows.sample(n=int(take_count), random_state=random_state)
        )

    if not sampled_parts:
        return df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)

    sampled_rows = pd.concat(sampled_parts, axis=0)
    sampled_rows = sampled_rows.sample(frac=1, random_state=random_state)
    return sampled_rows.drop(columns="_stratum").reset_index(drop=True)


def render_listings_tab(df: pd.DataFrame, listing_count: int) -> None:
    section_heading("Filtered Listings")
    page_note(
        f"Showing a representative sample of up to {LISTINGS_SAMPLE_SIZE:,} "
        f"from {listing_count:,} filtered listings."
    )
    st.markdown(
        "<p style='color:var(--t3);font-size:0.78rem;margin:-0.25rem 0 0.7rem;'>"
        f"Full source data: <a href='{FULL_DATA_URL}' target='_blank'>"
        "ScrapeAccraProperties outputs</a></p>",
        unsafe_allow_html=True,
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
    sampled_df = build_representative_sample(df)
    display_frame = (
        sampled_df[list(display_columns)]
        .rename(columns=display_columns)
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

    overview_tab, segments_tab, map_tab, listings_tab = st.tabs(
        ["Overview", "Segments", "Map", "Listings"]
    )
    with overview_tab:
        render_overview_tab(filtered_data, summary)
    with segments_tab:
        render_segments_tab(filtered_data)
    with map_tab:
        render_map_tab(filtered_data, summary.listing_count)
    with listings_tab:
        render_listings_tab(filtered_data, summary.listing_count)


main()
