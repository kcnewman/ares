import json
import os
from typing import Any

from lib.utils import logger

GROQ_API_KEY: str = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL: str = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_MAX_TOKENS: int = int(os.environ.get("GROQ_MAX_TOKENS", "512"))


def _call_groq(system: str, user: str) -> str | None:
    if not GROQ_API_KEY:
        return None
    try:
        from groq import Groq

        client = Groq(api_key=GROQ_API_KEY)
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=GROQ_MAX_TOKENS,
            temperature=0.1,
        )
        content = resp.choices[0].message.content
        return content.strip() if content else None
    except Exception as e:
        logger.warning(f"Groq call failed: {e}")
        return None


def explain_prediction(
    property_details: dict[str, Any],
    prediction: dict[str, float],
    market_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    loc = property_details.get("loc", "the area")
    house_type = property_details.get("house_type", "property")
    estimated = prediction.get("estimated_price", 0)
    low = prediction.get("lower_band", 0)
    high = prediction.get("upper_band", 0)

    if market_context and market_context.get("count", 0) > 0:
        seg_label = market_context.get("label", "Greater Accra")
        seg_count = market_context.get("count", 0)
        seg_median = market_context.get("median", 0)
        seg_delta = ((estimated - seg_median) / seg_median * 100) if seg_median > 0 else 0
        seg_delta_str = f"{seg_delta:+.0f}%" if abs(seg_delta) > 1 else "near the market median"
        confidence = "High" if seg_count >= 50 else "Moderate" if seg_count >= 15 else "Low"

        prompt = (
            f"A {house_type} in {loc} is estimated at GHS {estimated:,.0f}/mo "
            f"(range GHS {low:,.0f}–{high:,.0f}). "
            f"Segment: {seg_label} ({seg_count} listings, median GHS {seg_median:,.0f}). "
            f"The estimate is {seg_delta_str} vs the segment median.\n\n"
            "Explain this valuation in 2–3 factual sentences. "
            "Then list key factors driving the price (max 3), risks to the valuation (max 2), "
            "and assign confidence as High/Moderate/Low based on listing count.\n\n"
            "Respond ONLY with JSON:\n"
            '{"summary": "...", "key_factors": [{"factor": "...", "impact": "...", "direction": "up"|"down"|"neutral"}], "risks": ["..."], "confidence": "..."}'
        )
    else:
        confidence = "Moderate"
        prompt = (
            f"A {house_type} in {loc} is estimated at GHS {estimated:,.0f}/mo "
            f"(range GHS {low:,.0f}–{high:,.0f}). "
            "No market segment data available.\n\n"
            "Explain this valuation in 2–3 factual sentences. "
            "Include relevant risks and assign confidence as Moderate.\n\n"
            "Respond ONLY with JSON:\n"
            '{"summary": "...", "key_factors": [], "risks": ["..."], "confidence": "Moderate"}'
        )

    result = _call_groq(
        "You are a rental market analyst for Accra, Ghana. Be concise and factual. Output JSON only.",
        prompt,
    )

    if result:
        try:
            cleaned = result.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            parsed = json.loads(cleaned)
            parsed.setdefault("confidence", confidence)
            return parsed
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Failed to parse LLM explanation: {e}")

    return {
        "summary": "AI explanation is not available.",
        "key_factors": [],
        "risks": [],
        "confidence": confidence,
    }
