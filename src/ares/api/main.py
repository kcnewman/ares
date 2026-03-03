from fastapi import FastAPI, HTTPException
import pandas as pd
from starlette.concurrency import run_in_threadpool
from ares.api.schemas import HouseFeatures, PredictionResponse
from ares.pipeline.inference import predict
from ares import logger


app = FastAPI(
    title="ARES Housing API",
    description="Rental Listing Price Prediction with Uncertainty Bands",
)


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
        raise HTTPException(
            status_code=500, detail="Internal Server Error during inference."
        )
