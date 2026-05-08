# ARES — Accra Rental Estimation System

Estimates market rent for residential properties across Greater Accra, Ghana, using gradient-boosted regression. ARES combines a CatBoost pricing model with a FastAPI serving layer and a Streamlit frontend for exploration, prediction, and reporting.

## Architecture

```
┌──────────┐     ┌──────────┐     ┌──────────┐
│  Client  │────▶│  FastAPI │────▶│ CatBoost │
│ (Stream- │     │  (port   │     │  Model   │
│  lit)    │     │   8000)  │     │          │
└──────────┘     └──────────┘     └──────────┘
                       │
                       ▼
                  ┌──────────┐
                  │   Groq   │
                  │   LLM    │ (optional
                  │          │  explanation)
                  └──────────┘
```

- **CatBoostRegressor** — Gradient-boosted model trained on ~19k historical listings
- **FastAPI** — Synchronous REST API with Pydantic-validated request/response schemas
- **Streamlit** — Three-page UI: market explorer, valuation predictor, and detailed report
- **Groq LLM** — Optional natural-language explanation of valuation results (falls back gracefully)

## Quick Start

### Prerequisites

- Python 3.12
- `uv` package manager

### Setup

```bash
# Install dependencies (including dev tools)
uv sync --dev

# Train the pricing model
uv run python -m lib.train

# Start the API server
uv run uvicorn api.server:app --host 127.0.0.1 --port 8000

# In a separate terminal, launch the UI
BACKEND_URL="http://127.0.0.1:8000" uv run streamlit run ui/app.py --server.port=8080
```

Or use the combined launcher:

```bash
./run.sh
```

### Run Tests

```bash
uv run pytest -q --cov
```

## Project Structure

```
├── api/                  # REST API layer
│   ├── server.py         # FastAPI app — /predict, /explain, /health
│   └── schemas.py        # Pydantic models for validation
├── lib/                  # Core ML pipeline
│   ├── train.py          # Training orchestration with hyperparameters
│   ├── features.py       # Feature engineering (encoding, stats, cleaning)
│   ├── predict.py        # Inference: point estimate + uncertainty bands
│   ├── llm.py            # Groq-powered market explanation (optional)
│   └── utils.py          # Shared config, logging, I/O helpers
├── ui/                   # Streamlit frontend
│   ├── app.py            # Landing page with market snapshot
│   ├── utils.py          # CSS, chart config, HTML builders
│   └── pages/
│       ├── Explorer.py   # Filter, visualize, and segment listings
│       ├── Predictor.py  # Property form → model valuation
│       └── Report.py     # Detailed report with comparables
├── data/raw.csv          # Training dataset (Jiji.com.gh listings)
├── models/               # Serialized model + training metadata
├── tests/                # Pytest suite
├── pyproject.toml        # Dependency + tool config
└── Dockerfile            # Containerised deployment
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Service info |
| `GET` | `/health` | Health check + model existence |
| `POST` | `/predict` | Estimate rent from property features |
| `POST` | `/explain` | Estimate + natural-language explanation |

### Example: `/predict`

```json
{
  "house_type": "apartment",
  "condition": "newly built",
  "furnishing": "furnished",
  "loc": "tesano",
  "bathrooms": 2,
  "bedrooms": 2,
  "air_conditioning": 1,
  "balcony": 1,
  "wi_fi": 1
}
```

Response:

```json
{
  "estimated_price": 3850.00,
  "lower_band": 3200.00,
  "upper_band": 4620.00,
  "market_volatility_idx": 0.38,
  "market_volatility_pct": 46.2,
  "market_volatility_tier": "Moderate"
}
```

## Feature Engineering

The model uses three categories of features:

- **Numerical** — bedrooms, bathrooms
- **Categorical** — location, property type, condition, furnishing (one-hot encoded)
- **Amenity flags** — 18 binary indicators (air conditioning, Wi-Fi, balcony, etc.)
- **Derived** — total amenity count, luxury amenity count, location-level statistics (median log-price, volatility, listing density)

The prediction outputs an **uncertainty band** derived from the log-scale interquartile range of comparable listings, adjusted for listing depth at the location level.

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `GROQ_API_KEY` | — | API key for LLM explanations |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model name |
| `GROQ_MAX_TOKENS` | `512` | Max tokens per LLM call |
| `CACHE_DIR` | `models` | Directory for LLM response cache |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

## Deployment

### Docker

```bash
docker compose up --build
```

### Environment

Copy `.env.example` to `.env` and populate `GROQ_API_KEY` if LLM explanations are desired. The API and model will function without it using a rule-based fallback.

## License

MIT
