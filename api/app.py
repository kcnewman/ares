from pathlib import Path

import httpx
import pandas as pd
from fastapi import FastAPI, HTTPException
from starlette.concurrency import run_in_threadpool

from api.schemas import ExplainResponse, Factor, HouseFeatures, PredictionResponse
from core.config import load_config
from core.logger import logger
from core.pipeline.inference import predict

app = FastAPI(
    title="ARES Housing API",
    description="Rental Listing Price Prediction with Uncertainty Bands",
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MARKET_DATA_PATH = PROJECT_ROOT / "artifacts" / "data_processing" / "preprocessed_train.csv"


def _load_market_data() -> pd.DataFrame | None:
    try:
        df = pd.read_csv(MARKET_DATA_PATH)
        return df
    except Exception:
        return None


def _compute_segment(df: pd.DataFrame, loc: str, house_type: str, furnishing: str) -> dict:
    candidates = [
        (df[(df["loc"] == loc) & (df["house_type"] == house_type) & (df["furnishing"] == furnishing)], f"{furnishing} {house_type} in {loc}"),
        (df[(df["loc"] == loc) & (df["house_type"] == house_type)], f"{house_type} in {loc}"),
        (df[df["loc"] == loc], f"All types in {loc}"),
    ]
    for seg_df, label in candidates:
        if len(seg_df) >= 5:
            break
    else:
        seg_df = df
        label = "All Greater Accra"

    return {
        "label": label,
        "count": len(seg_df),
        "median": float(seg_df["price"].median()),
        "q25": float(seg_df["price"].quantile(0.25)),
        "q75": float(seg_df["price"].quantile(0.75)),
    }


@app.get("/")
def root():
    return {"message": "API is running"}


@app.get("/health")
def health_check():
    return {"status": "active", "version": "1.0.0"}


@app.post("/predict", response_model=PredictionResponse)
async def get_prediction(features: HouseFeatures):
    try:
        data_dict = features.model_dump(by_alias=True)
        df = pd.DataFrame([data_dict])
        prediction = await run_in_threadpool(predict, df)
        result = prediction.iloc[0]

        return PredictionResponse(
            estimated_price=result["estimated_price"],
            lower_band=result["lower_band"],
            upper_band=result["upper_band"],
            market_volatility_idx=result["market_volatility_idx"],
            market_volatility_pct=result["market_volatility_pct"],
            market_volatility_tier=result["market_volatility_tier"],
        )
    except Exception as e:
        logger.error(f"API Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error during inference.")


@app.post("/explain", response_model=ExplainResponse)
async def get_explanation(features: HouseFeatures):
    try:
        data_dict = features.model_dump(by_alias=True)
        df = pd.DataFrame([data_dict])
        prediction = await run_in_threadpool(predict, df)
        result = prediction.iloc[0]

        market_data = _load_market_data()
        segment = _compute_segment(
            market_data,
            data_dict.get("loc", ""),
            data_dict.get("house_type", ""),
            data_dict.get("furnishing", ""),
        ) if market_data is not None else {}

        config = load_config()
        llm_url = config.get("llm_service_url", "http://localhost:8001")

        resp = httpx.post(
            f"{llm_url}/explain",
            json={
                "property": data_dict,
                "prediction": {
                    "estimated_price": float(result["estimated_price"]),
                    "lower_band": float(result["lower_band"]),
                    "upper_band": float(result["upper_band"]),
                },
                "market_context": segment,
            },
            timeout=15,
        )
        if resp.status_code != 200:
            raise HTTPException(status_code=502, detail="LLM service error")

        data = resp.json()
        return ExplainResponse(
            summary=data.get("summary", ""),
            key_factors=[Factor(**f) for f in data.get("key_factors", [])],
            risks=data.get("risks", []),
            confidence=data.get("confidence", "Moderate"),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"API Explain failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate explanation.")
