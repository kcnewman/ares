import numpy as np
import pandas as pd

from lib.utils import (
    AMENITY_COLUMNS,
    CATEGORICAL_COLUMNS,
    DROP_COLUMNS,
    LUXURY_AMENITIES,
    NUMERIC_COLUMNS,
    RAW_COLUMN_MAP,
    logger,
)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "locality" in df.columns:
        df = df.drop(columns=["locality"])
    df = df.rename(columns=RAW_COLUMN_MAP)
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).str.lower().str.strip()
            df[col] = df[col].replace("nan", np.nan)
    for col in AMENITY_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    if "price" in df.columns:
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
    return df


def remove_price_outliers(df: pd.DataFrame, multiplier: float = 1.5) -> pd.DataFrame:
    prices = df["price"].dropna()
    if prices.empty:
        return df
    log_prices = np.log(prices.clip(lower=1))
    q1 = log_prices.quantile(0.25)
    q3 = log_prices.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    before = len(df)
    df = df[log_prices.between(lower, upper)].copy()
    logger.info(f"Removed {before - len(df)} price outliers (multiplier={multiplier})")
    return df


def compute_location_stats(df: pd.DataFrame) -> dict:
    stats = {}
    if "loc" not in df.columns or "price" not in df.columns:
        return stats

    log_prices = df["price"].apply(lambda x: np.log(max(x, 1)))
    df_with_log = df.assign(log_price=log_prices)
    grouped = df_with_log.groupby("loc")["log_price"].agg(
        [
            "count",
            "mean",
            "std",
            lambda x: np.percentile(x.dropna(), 75) - np.percentile(x.dropna(), 25),
        ]
    )
    grouped.columns = ["count", "mean", "std", "iqr"]

    for loc, row in grouped.iterrows():
        stats[str(loc)] = {
            "count": int(row["count"]),
            "median_log_price": float(row["mean"]),
            "std_log_price": float(row["std"]) if pd.notna(row["std"]) else 0.0,
            "iqr_log_price": float(row["iqr"]) if pd.notna(row["iqr"]) else 0.0,
        }

    return stats


def prepare_features(
    df: pd.DataFrame,
    location_stats: dict | None = None,
    categories: dict | None = None,
    fit: bool = False,
) -> pd.DataFrame:
    df = df.copy()

    if "price" in df.columns:
        df["log_price"] = np.log(df["price"].clip(lower=1))
    df["total_amenities"] = df[AMENITY_COLUMNS].sum(axis=1)
    df["luxury_count"] = df[[c for c in AMENITY_COLUMNS if c in LUXURY_AMENITIES]].sum(
        axis=1
    )

    if fit:
        categories = {}
        for col in CATEGORICAL_COLUMNS:
            if col in df.columns:
                categories[col] = sorted(df[col].dropna().unique())

    if categories:
        for col in CATEGORICAL_COLUMNS:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col)
                for cat_col in dummies.columns:
                    df[cat_col] = dummies[cat_col]
                if fit:
                    categories[col] = sorted(df[col].dropna().unique())

    if location_stats:
        global_median = (
            np.median([s["median_log_price"] for s in location_stats.values()])
            if location_stats
            else 0.0
        )
        global_iqr = (
            np.median([s["iqr_log_price"] for s in location_stats.values()])
            if location_stats
            else 0.0
        )

        df["loc_median_price"] = df["loc"].apply(
            lambda x: location_stats.get(str(x), {}).get(
                "median_log_price", global_median
            )
        )
        df["loc_volatility"] = df["loc"].apply(
            lambda x: location_stats.get(str(x), {}).get("iqr_log_price", global_iqr)
        )
        df["loc_count"] = df["loc"].apply(
            lambda x: location_stats.get(str(x), {}).get("count", 0)
        )
    else:
        df["loc_median_price"] = 0.0
        df["loc_volatility"] = 0.0
        df["loc_count"] = 0

    feature_cols = (
        NUMERIC_COLUMNS
        + AMENITY_COLUMNS
        + [
            "total_amenities",
            "luxury_count",
            "loc_median_price",
            "loc_volatility",
            "loc_count",
        ]
    )
    if categories:
        for col in CATEGORICAL_COLUMNS:
            for cat in categories.get(col, []):
                feature_cols.append(f"{col}_{cat}")

    for col in DROP_COLUMNS:
        if col in df.columns:
            df = df.drop(columns=[col])

    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    available = [c for c in feature_cols if c in df.columns]
    df = df[available + (["log_price"] if "log_price" in df.columns else [])]

    return df, {"categories": categories, "feature_columns": available}
