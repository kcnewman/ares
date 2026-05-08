from pydantic import BaseModel
from typing import Literal

LOCATION_TIERS = Literal["prime", "established", "high_density", "developing_commuter", "industrial_traffic", "satellite_hub", "other"]

class LocClassifyRequest(BaseModel):
    location: str

class LocClassifyResponse(BaseModel):
    location: str
    tier: LOCATION_TIERS
    is_elite: bool

class BatchLocClassifyRequest(BaseModel):
    locations: list[str]

class BatchLocClassifyResponse(BaseModel):
    classifications: list[LocClassifyResponse]

AMENITY_TIERS = Literal["luxury", "standard"]

class AmenityClassifyRequest(BaseModel):
    amenities: list[str]

class AmenityClassification(BaseModel):
    amenity: str
    tier: AMENITY_TIERS

class AmenityClassifyResponse(BaseModel):
    classifications: list[AmenityClassification]

class Factor(BaseModel):
    factor: str
    impact: str
    direction: Literal["up", "down", "neutral"]

class ExplainRequest(BaseModel):
    property: dict
    prediction: dict
    market_context: dict

class ExplainResponse(BaseModel):
    summary: str
    key_factors: list[Factor]
    risks: list[str]
    confidence: str
