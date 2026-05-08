import json
import os
from pathlib import Path

from lib.utils import logger

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_MAX_TOKENS = int(os.environ.get("GROQ_MAX_TOKENS", "512"))
CACHE_DIR = Path(os.environ.get("CACHE_DIR", "models"))


class JsonFileCache:
    def __init__(self, path: Path):
        self.path = path
        self._data = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            with open(self.path) as f:
                return json.load(f)
        return {}

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self._data, f, indent=2)

    def get(self, key: str):
        return self._data.get(key)

    def set(self, key: str, value):
        self._data[key] = value
        self._save()

    def has(self, key: str) -> bool:
        return key in self._data


location_cache = JsonFileCache(CACHE_DIR / "location_tiers.json")
amenity_cache = JsonFileCache(CACHE_DIR / "amenity_tiers.json")

LOCATION_TIER_PROMPT = """Classify this Accra, Ghana location into one of these tiers:
- prime: Wealthy, high-end residential areas (e.g., Airport Residential, Ridge, Cantonments, East Legon)
- established: Well-developed middle-upper areas (e.g., Dzorwulu, Labone, Osu, Tesano)
- high_density: Dense, popular areas with mixed use (e.g., Circle, Nima, Madina, Adenta)
- developing_commuter: Growing suburbs further from city center (e.g., Kasoa, Pokuase, Amasaman)
- industrial_traffic: Areas with heavy commercial/industrial activity (e.g., North Industrial Area, Spintex)
- satellite_hub: Towns that function as their own urban centers (e.g., Tema, Ashaiman)
- other: Does not fit any category above

Return ONLY the tier name, nothing else."""

AMENITY_TIER_PROMPT = """Classify this amenity/feature in a rental property in Ghana as either "luxury" or "standard":
- luxury: High-end features like air conditioning, chandelier, dishwasher, hot water, microwave, refrigerator, TV, Wi-Fi
- standard: Basic/common features like 24-hour electricity, balcony, dining area, kitchen cabinets, tiled floor, wardrobe

Return ONLY "luxury" or "standard", nothing else."""


def _get_groq_client():
    if not GROQ_API_KEY:
        return None
    try:
        from groq import Groq

        return Groq(api_key=GROQ_API_KEY)
    except Exception:
        return None


def _call_groq(system_prompt: str, user_message: str) -> str | None:
    client = _get_groq_client()
    if not client:
        return None
    try:
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=GROQ_MAX_TOKENS,
            temperature=0.1,
        )
        return resp.choices[0].message.content.strip().lower()
    except Exception as e:
        logger.warning(f"Groq call failed: {e}")
        return None


def classify_location(location: str) -> str:
    cached = location_cache.get(location.lower())
    if cached:
        return cached

    result = _call_groq(LOCATION_TIER_PROMPT, location)
    if result and result in (
        "prime",
        "established",
        "high_density",
        "developing_commuter",
        "industrial_traffic",
        "satellite_hub",
        "other",
    ):
        location_cache.set(location.lower(), result)
        return result

    return "other"


def classify_amenity(amenity: str) -> str:
    cached = amenity_cache.get(amenity.lower())
    if cached:
        return cached

    result = _call_groq(AMENITY_TIER_PROMPT, amenity)
    if result in ("luxury", "standard"):
        amenity_cache.set(amenity.lower(), result)
        return result

    return "standard"


def explain_prediction(
    property_details: dict, prediction: dict, market_context: dict | None = None
) -> dict:
    context_str = ""
    if market_context:
        context_str = (
            f"Market segment ({market_context.get('label', 'N/A')}): "
            f"median GHS {market_context.get('median', 0):,.0f}, "
            f"based on {market_context.get('count', 0)} comparable listings. "
        )

    prompt = f"""You are a rental market analyst for Accra, Ghana. Explain this property valuation:

Property: {json.dumps(property_details)}
Estimated Rent: GHS {prediction["estimated_price"]:,.0f}
Range: GHS {prediction["lower_band"]:,.0f} - GHS {prediction["upper_band"]:,.0f}
{context_str}

Respond in JSON with:
- "summary": 2-3 sentence plain-English explanation of why the price is what it is
- "key_factors": list of objects {{"factor": str, "impact": str, "direction": "up"|"down"|"neutral"}}
- "risks": list of potential risks to this valuation
- "confidence": "High"|"Moderate"|"Low" based on market data available"""

    result = _call_groq(
        "You are a rental market analyst for Accra, Ghana. Respond in JSON only.",
        prompt,
    )
    if result:
        try:
            cleaned = result.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1]
                cleaned = cleaned.rsplit("\n", 1)[0]
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
                cleaned = cleaned.rsplit("```", 1)[0]
            return json.loads(cleaned)
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Failed to parse LLM explanation: {e}")

    return {
        "summary": f"This {property_details.get('house_type', 'property')} in {property_details.get('loc', 'the area')} is estimated at GHS {prediction['estimated_price']:,.0f}, based on comparable listings and market trends.",
        "key_factors": [
            {
                "factor": f"Property type: {property_details.get('house_type', 'N/A')}",
                "impact": "Standard for this property type",
                "direction": "neutral",
            },
            {
                "factor": f"Location: {property_details.get('loc', 'N/A')}",
                "impact": "Based on local market data",
                "direction": "neutral",
            },
        ],
        "risks": ["Market conditions may change rapidly"],
        "confidence": "Moderate",
    }
