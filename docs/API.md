# API Reference

ARES exposes a FastAPI application for online inference.

- App module: `src/ares/api/main.py`
- Default local base URL: `http://127.0.0.1:8000`

## Run Locally

```bash
uv run uvicorn ares.api.main:app --host 127.0.0.1 --port 8000 --reload
```

Interactive docs are available at:

- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/redoc`

## Endpoints

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/` | Service banner response |
| `GET` | `/health` | Health check with version payload |
| `POST` | `/predict` | Returns predicted rent with uncertainty bands |

## `POST /predict` Request Contract

All fields are required.

| Field | Type | Constraints |
| --- | --- | --- |
| `house_type` | string | expected to match trained categories |
| `condition` | string | expected to match trained categories |
| `furnishing` | string | expected to match trained categories |
| `loc` | string | expected to match trained localities |
| `bathrooms` | integer | `>= 0` |
| `bedrooms` | integer | `>= 0` |
| `24_hour_electricity` | integer | `0` or `1` |
| `air_conditioning` | integer | `0` or `1` |
| `apartment` | integer | `0` or `1` |
| `balcony` | integer | `0` or `1` |
| `chandelier` | integer | `0` or `1` |
| `dining_area` | integer | `0` or `1` |
| `dishwasher` | integer | `0` or `1` |
| `hot_water` | integer | `0` or `1` |
| `kitchen_cabinets` | integer | `0` or `1` |
| `kitchen_shelf` | integer | `0` or `1` |
| `microwave` | integer | `0` or `1` |
| `pop_ceiling` | integer | `0` or `1` |
| `pre_paid_meter` | integer | `0` or `1` |
| `refrigerator` | integer | `0` or `1` |
| `tv` | integer | `0` or `1` |
| `tiled_floor` | integer | `0` or `1` |
| `wardrobe` | integer | `0` or `1` |
| `wi_fi` | integer | `0` or `1` |

## Example Request

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "house_type": "apartment",
    "condition": "good",
    "furnishing": "furnished",
    "loc": "east legon",
    "bathrooms": 2,
    "bedrooms": 2,
    "24_hour_electricity": 1,
    "air_conditioning": 1,
    "apartment": 1,
    "balcony": 1,
    "chandelier": 0,
    "dining_area": 1,
    "dishwasher": 0,
    "hot_water": 1,
    "kitchen_cabinets": 1,
    "kitchen_shelf": 1,
    "microwave": 0,
    "pop_ceiling": 0,
    "pre_paid_meter": 1,
    "refrigerator": 1,
    "tv": 0,
    "tiled_floor": 1,
    "wardrobe": 1,
    "wi_fi": 1
  }'
```

## Example Response

```json
{
  "estimated_price": 7400.0,
  "lower_band": 6500.0,
  "upper_band": 8500.0,
  "market_volatility_idx": 0.27
}
```

## Error Behavior

- Request schema violations return standard FastAPI validation errors.
- Inference/runtime failures return:
  - status code: `500`
  - body: `{"detail":"Internal Server Error during inference."}`

## Notes

- Predictions are returned in price space after exponentiating model outputs.
- Uncertainty bands are locality-aware and derived from stored locality volatility statistics.
- The API accepts `24_hour_electricity` (alias) for electricity input.
