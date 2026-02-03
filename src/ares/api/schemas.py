from pydantic import BaseModel, Field


class HouseFeatures(BaseModel):
    house_type: str = Field(..., examples=["duplex"])
    condition: str = Field(..., examples=["new"])
    furnishing: str = Field(..., examples=["furnished"])
    loc: str = Field(..., examples=["tesano"])
    bathrooms: int = Field(..., ge=0)
    bedrooms: int = Field(..., ge=0)
    electricity_24h: int = Field(..., alias="24_hour_electricity", ge=0, le=1)
    air_conditioning: int = Field(..., ge=0, le=1)
    apartment: int = Field(..., ge=0, le=1)
    balcony: int = Field(..., ge=0, le=1)
    chandelier: int = Field(..., ge=0, le=1)
    dining_area: int = Field(..., ge=0, le=1)
    dishwasher: int = Field(..., ge=0, le=1)
    hot_water: int = Field(..., ge=0, le=1)
    kitchen_cabinets: int = Field(..., ge=0, le=1)
    kitchen_shelf: int = Field(..., ge=0, le=1)
    microwave: int = Field(..., ge=0, le=1)
    pop_ceiling: int = Field(..., ge=0, le=1)
    pre_paid_meter: int = Field(..., ge=0, le=1)
    refrigerator: int = Field(..., ge=0, le=1)
    tv: int = Field(..., ge=0, le=1)
    tiled_floor: int = Field(..., ge=0, le=1)
    wardrobe: int = Field(..., ge=0, le=1)
    wi_fi: int = Field(..., ge=0, le=1)

    class Config:
        populate_by_name = True


class PredictionResponse(BaseModel):
    estimated_price: float
    lower_band: float
    upper_band: float
    market_volatility_idx: float
