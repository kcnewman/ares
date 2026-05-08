# ARES — Accra Rental Estimation System

End-to-end rental price estimation for Greater Accra, Ghana. Uses CatBoost for predictions, FastAPI for serving, and Streamlit for the UI.

## Tech Stack

- **Model:** CatBoostRegressor
- **API:** FastAPI + Uvicorn
- **UI:** Streamlit
- **LLM:** Groq (llama-3.3-70b-versatile) for optional market explanations
- **Env/Deps:** `uv`

## Quick Start

```bash
# Install
uv sync --dev

# Train model
uv run python -m lib.train

# Run API (port 8000)
uv run uvicorn api.server:app --host 127.0.0.1 --port 8000

# Run UI (port 8080, separate terminal)
BACKEND_URL="http://127.0.0.1:8000" uv run streamlit run ui/app.py --server.port=8080

# Or run both via script
./run.sh

# Run tests
uv run pytest -q --cov
```

## Project Layout

```
├── api/              # FastAPI app + Pydantic schemas
├── lib/              # Training, features, prediction, LLM
│   ├── train.py      # CatBoost training pipeline
│   ├── features.py   # Feature engineering
│   ├── predict.py    # Model loading + inference
│   ├── llm.py        # Groq-based market explanations
│   └── utils.py      # Shared helpers, constants, logging
├── ui/               # Streamlit frontend
│   ├── app.py        # Home page
│   ├── utils.py      # UI helpers + styles
│   └── pages/        # Explorer, Predictor, Report
├── data/raw.csv      # Training data (Jiji.com.gh listings)
├── models/           # Trained model + metadata
├── tests/            # Pytest suite
├── Dockerfile
├── docker-compose.yml
└── entrypoint.sh
```
