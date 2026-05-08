import json
from pathlib import Path

from fastapi import FastAPI
from groq import Groq

from llm.cache import JsonFileCache
from llm.config import CACHE_DIR, GROQ_API_KEY, GROQ_MAX_TOKENS, GROQ_MODEL
from llm.schemas import (
    AmenityClassification,
    AmenityClassifyRequest,
    AmenityClassifyResponse,
    BatchLocClassifyRequest,
    BatchLocClassifyResponse,
    ExplainRequest,
    ExplainResponse,
    Factor,
    LocClassifyResponse,
)

app = FastAPI(title="ARES LLM Service")

client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

cache_dir = Path(CACHE_DIR)
location_cache = JsonFileCache(cache_dir / "location_tiers.json")
amenity_cache = JsonFileCache(cache_dir / "amenity_tiers.json")

LOCATION_SYSTEM_PROMPT = """You are a real estate analyst for Accra, Ghana. Classify each location into exactly one of these tiers:

- prime: Most desirable areas, high-end neighborhoods, diplomatic enclaves (e.g., Airport Residential, Ridge, East Legon, Cantonments)
- established: Well-developed middle-class residential areas with good infrastructure (e.g., Tesano, Adabraka, Asylum Down, Labone)
- high_density: Dense, busy areas with mixed-use development, many apartment buildings (e.g., Dansoman, Madina, Circle, Nima)
- developing_commuter: Rapidly growing peri-urban areas, newer developments on city outskirts (e.g., Amrahia, Oyibi, Pantang, Ashongman)
- industrial_traffic: Areas dominated by industry, warehouses, heavy traffic, or major road corridors (e.g., Spintex Road, Tema, Ashaiman)
- satellite_hub: Regional towns outside Accra that serve as commercial centers (e.g., Kasoa, Nsawam, Dodowa)
- other: Any location that doesn't fit above categories

Respond with a JSON array of objects, each with: location, tier, reasoning. Use only the exact tier names listed above."""

AMENITY_SYSTEM_PROMPT = """You are a real estate analyst for Accra, Ghana. Classify each amenity as either "luxury" or "standard".

- luxury: Significantly increases property value, expensive to install/maintain, desirable but not essential (e.g., air conditioning, dishwasher, microwave, chandelier)
- standard: Normal, expected, lower cost amenities that most rentals have (e.g., tiled floor, wardrobe, water, electricity)

Respond with a JSON object where keys are amenity names and values are "luxury" or "standard".
If uncertain about any amenity, default to "standard"."""


def _call_groq(system_prompt: str, user_prompt: str) -> str | None:
    if client is None:
        return None
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=GROQ_MAX_TOKENS,
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        return response.choices[0].message.content
    except Exception:
        return None


def _parse_json_response(content: str | None) -> dict | list | None:
    if content is None:
        return None
    try:
        return json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return None


@app.get("/health")
def health():
    return {"status": "active"}


@app.post("/classify-locations", response_model=BatchLocClassifyResponse)
def classify_locations(request: BatchLocClassifyRequest):
    results: list[LocClassifyResponse] = []
    uncached: list[str] = []

    for loc in request.locations:
        loc_key = loc.strip().lower()
        cached = location_cache.get(loc_key)
        if cached is not None:
            results.append(
                LocClassifyResponse(
                    location=loc_key,
                    tier=cached["tier"],
                    is_elite=cached["is_elite"],
                )
            )
        else:
            uncached.append(loc_key)

    if uncached:
        result = _classify_locations_batch(uncached)
        for loc in uncached:
            entry = result.get(loc, {"tier": "other", "is_elite": False})
            location_cache.set(loc, entry)
            results.append(
                LocClassifyResponse(
                    location=loc,
                    tier=entry["tier"],
                    is_elite=entry["is_elite"],
                )
            )

    return BatchLocClassifyResponse(classifications=results)


def _classify_locations_batch(locations: list[str]) -> dict:
    content = _call_groq(
        LOCATION_SYSTEM_PROMPT,
        f"Classify these Accra locations into tiers. Return a JSON array: {json.dumps(locations)}",
    )
    parsed = _parse_json_response(content)
    if isinstance(parsed, list):
        result = {}
        for item in parsed:
            loc = item.get("location", "").strip().lower()
            tier = item.get("tier", "other")
            if tier not in {"prime", "established", "high_density", "developing_commuter",
                            "industrial_traffic", "satellite_hub", "other"}:
                tier = "other"
            result[loc] = {"tier": tier, "is_elite": tier == "prime"}
        return result
    return {loc: {"tier": "other", "is_elite": False} for loc in locations}


@app.post("/classify-amenities", response_model=AmenityClassifyResponse)
def classify_amenities(request: AmenityClassifyRequest):
    results: list[AmenityClassification] = []
    uncached: list[str] = []

    for amenity in request.amenities:
        cached = amenity_cache.get(amenity)
        if cached is not None:
            results.append(AmenityClassification(amenity=amenity, tier=cached))
        else:
            uncached.append(amenity)

    if uncached:
        tiers = _classify_amenities_batch(uncached)
        for amenity in uncached:
            tier = tiers.get(amenity, "standard")
            amenity_cache.set(amenity, tier)
            results.append(AmenityClassification(amenity=amenity, tier=tier))

    return AmenityClassifyResponse(classifications=results)


def _classify_amenities_batch(amenities: list[str]) -> dict:
    content = _call_groq(
        AMENITY_SYSTEM_PROMPT,
        f"Classify each amenity as luxury or standard. Return JSON: {json.dumps(amenities)}",
    )
    parsed = _parse_json_response(content)
    if isinstance(parsed, dict):
        result = {}
        for amenity, tier in parsed.items():
            if isinstance(tier, str) and tier.lower() in {"luxury", "standard"}:
                result[amenity] = tier.lower()
        return result
    return {a: "standard" for a in amenities}


EXPLAIN_SYSTEM_PROMPT = """You are a rental market analyst for Accra, Ghana. Given property details, prediction, and market context, write a concise 2-3 sentence explanation. Be specific with percentages and amounts. Then list key factors driving the price up or down, and list risks/uncertainties.

Respond with JSON only using this schema:
{
  "summary": "string",
  "key_factors": [{"factor": "string", "impact": "string", "direction": "up|down|neutral"}],
  "risks": ["string"],
  "confidence": "string (Low|Moderate|High)".
}"""


@app.post("/explain", response_model=ExplainResponse)
def explain(request: ExplainRequest):
    content = _call_groq(
        EXPLAIN_SYSTEM_PROMPT,
        f"Property: {json.dumps(request.property)}\nPrediction: {json.dumps(request.prediction)}\nMarket Context: {json.dumps(request.market_context)}",
    )
    parsed = _parse_json_response(content)
    if parsed is None:
        return ExplainResponse(
            summary="Explanation unavailable.",
            key_factors=[],
            risks=["LLM service could not generate explanation."],
            confidence="Low",
        )

    key_factors = []
    for f in parsed.get("key_factors", []):
        key_factors.append(
            Factor(
                factor=f.get("factor", ""),
                impact=f.get("impact", ""),
                direction=f.get("direction", "neutral"),
            )
        )

    return ExplainResponse(
        summary=parsed.get("summary", ""),
        key_factors=key_factors,
        risks=parsed.get("risks", []),
        confidence=parsed.get("confidence", "Moderate"),
    )
