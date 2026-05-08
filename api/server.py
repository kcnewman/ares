from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException

from api.schemas import ExplainResponse, Factor, HouseFeatures, PredictionResponse
from lib.predict import predict_from_dict
from lib.utils import logger

app = FastAPI(title="ARES API", description="Accra Rental Estimation API")

MODEL_DIR = Path("models")
DATA_PATH = Path("data/raw.csv")

_GENERIC_ERROR = "An internal error occurred. Please try again later."


def load_market_data() -> pd.DataFrame | None:
    try:
        if not DATA_PATH.exists():
            return None
        df = pd.read_csv(DATA_PATH)
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
        for col in ["loc", "house_type", "condition", "furnishing"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.lower().str.strip()
        return df
    except Exception:
        logger.exception("Failed to load market data")
        return None


def compute_segment(
    df: pd.DataFrame | None, loc: str, house_type: str, furnishing: str
) -> dict[str, Any]:
    if df is None or df.empty:
        return {
            "label": "All Greater Accra",
            "count": 0,
            "median": 0,
            "q25": 0,
            "q75": 0,
        }

    required = {"loc", "house_type", "furnishing", "price"}
    if not required.issubset(df.columns):
        logger.warning("Market data missing required columns for segmentation")
        return {
            "label": "All Greater Accra",
            "count": 0,
            "median": 0,
            "q25": 0,
            "q75": 0,
        }

    candidates = [
        (
            df[
                (df["loc"] == loc)
                & (df["house_type"] == house_type)
                & (df["furnishing"] == furnishing)
            ],
            f"{furnishing} {house_type} in {loc}",
        ),
        (
            df[(df["loc"] == loc) & (df["house_type"] == house_type)],
            f"{house_type} in {loc}",
        ),
        (df[df["loc"] == loc], f"All types in {loc}"),
    ]
    seg_df, label = df, "All Greater Accra"
    for cand_df, cand_label in candidates:
        if len(cand_df) >= 5:
            seg_df, label = cand_df, cand_label
            break

    prices = pd.to_numeric(seg_df["price"], errors="coerce").dropna()
    return {
        "label": label,
        "count": len(prices),
        "median": float(prices.median()) if not prices.empty else 0,
        "q25": float(prices.quantile(0.25)) if not prices.empty else 0,
        "q75": float(prices.quantile(0.75)) if not prices.empty else 0,
    }


@app.get("/")
def root():
    return {"message": "ARES API is running"}


@app.get("/health")
def health():
    model_path = MODEL_DIR / "model.joblib"
    return {"status": "active", "model_exists": model_path.exists()}


@app.post("/predict", response_model=PredictionResponse)
async def get_prediction(features: HouseFeatures):
    try:
        data = features.model_dump(by_alias=True)
        result = predict_from_dict(data, MODEL_DIR)
        return PredictionResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=_GENERIC_ERROR)


@app.post("/explain", response_model=ExplainResponse)
async def get_explanation(features: HouseFeatures):
    try:
        data = features.model_dump(by_alias=True)
        prediction = predict_from_dict(data, MODEL_DIR)

        house_type = data.get("house_type", "property")
        loc = data.get("loc", "the area")
        furnishing = data.get("furnishing", "unfurnished")

        market_df = load_market_data()
        segment = compute_segment(market_df, loc, house_type, furnishing)

        try:
            from lib.llm import explain_prediction

            explanation = explain_prediction(data, prediction, segment)
        except Exception as e:
            logger.warning(f"LLM explanation failed, using fallback: {e}")
            explanation = {
                "summary": f"This {house_type} in {loc} is estimated at GHS {prediction['estimated_price']:,.0f}, based on {segment.get('count', 0):,} comparable listings.",
                "key_factors": [],
                "risks": [],
                "confidence": "Moderate",
            }

        return ExplainResponse(
            summary=explanation.get("summary", ""),
            key_factors=[Factor(**f) for f in explanation.get("key_factors", [])],
            risks=explanation.get("risks", []),
            confidence=explanation.get("confidence", "Moderate"),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Explain failed: {e}")
        raise HTTPException(status_code=500, detail=_GENERIC_ERROR)
