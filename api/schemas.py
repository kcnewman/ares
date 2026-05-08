from typing import Literal

from pydantic import BaseModel, Field


class HouseFeatures(BaseModel):
    house_type: str = Field(..., examples=["duplex"])
    condition: str = Field(..., examples=["newly built"])
    furnishing: str = Field(..., examples=["furnished"])
    loc: str = Field(..., examples=["tesano"])
    bathrooms: int = Field(..., ge=0)
    bedrooms: int = Field(..., ge=0)
    elec_24h: int = Field(default=0, alias="24_hour_electricity", ge=0, le=1)
    air_conditioning: int = Field(default=0, ge=0, le=1)
    apartment: int = Field(default=0, ge=0, le=1)
    balcony: int = Field(default=0, ge=0, le=1)
    chandelier: int = Field(default=0, ge=0, le=1)
    dining_area: int = Field(default=0, ge=0, le=1)
    dishwasher: int = Field(default=0, ge=0, le=1)
    hot_water: int = Field(default=0, ge=0, le=1)
    kitchen_cabinets: int = Field(default=0, ge=0, le=1)
    kitchen_shelf: int = Field(default=0, ge=0, le=1)
    microwave: int = Field(default=0, ge=0, le=1)
    pop_ceiling: int = Field(default=0, ge=0, le=1)
    pre_paid_meter: int = Field(default=0, ge=0, le=1)
    refrigerator: int = Field(default=0, ge=0, le=1)
    tv: int = Field(default=0, ge=0, le=1)
    tiled_floor: int = Field(default=0, ge=0, le=1)
    wardrobe: int = Field(default=0, ge=0, le=1)
    wi_fi: int = Field(default=0, ge=0, le=1)

    class Config:
        populate_by_name = True


class Factor(BaseModel):
    factor: str
    impact: str
    direction: Literal["up", "down", "neutral"]


class PredictionResponse(BaseModel):
    estimated_price: float
    lower_band: float
    upper_band: float
    market_volatility_idx: float
    market_volatility_pct: float
    market_volatility_tier: str


class ExplainResponse(BaseModel):
    estimated_price: float
    lower_band: float
    upper_band: float
    market_volatility_idx: float
    market_volatility_pct: float
    market_volatility_tier: str
    summary: str
    key_factors: list[Factor]
    risks: list[str]
    confidence: str
