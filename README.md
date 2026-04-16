# AI Real Estate Agent

A two-stage LLM pipeline that converts natural language property descriptions into price estimates, powered by a Random Forest model trained on the Ames Housing dataset and Gemini Flash 2.5.

---

## Architecture

```
User Query (natural language)
        │
        ▼
┌───────────────────┐
│  Stage 1 — LLM    │  Gemini Flash 2.5
│  Feature          │  Extracts 12 structured features
│  Extraction       │  Reports missing fields to user
└────────┬──────────┘
         │ ExtractionResult (12 features)
         ▼
┌───────────────────┐
│  ML Predictor     │  Random Forest (sklearn Pipeline)
│                   │  Trained on 1,758 Ames homes
│                   │  Test RMSE: $29,408 | R²: 0.873
└────────┬──────────┘
         │ PredictionResult (price + market context)
         ▼
┌───────────────────┐
│  Stage 2 — LLM    │  Gemini Flash 2.5
│  Interpretation   │  Generates 2–3 sentence analysis
└────────┬──────────┘
         │ AgentResponse
         ▼
   FastAPI  ──►  Streamlit UI
```

### 12 Selected Features

| Feature | Type | Description |
|---|---|---|
| Overall Qual | Numeric | Overall material and finish quality (1–10) |
| Gr Liv Area | Numeric | Above-grade living area (sqft) |
| Garage Area | Numeric | Garage size (sqft) |
| Total Bsmt SF | Numeric | Total basement area (sqft) |
| Year Built | Numeric | Original construction year |
| Full Bath | Numeric | Full bathrooms above grade |
| Mas Vnr Area | Numeric | Masonry veneer area (sqft) |
| Bsmt Qual | Ordinal | Basement quality (NA/Po/Fa/TA/Gd/Ex) |
| Exter Qual | Ordinal | Exterior quality (NA/Po/Fa/TA/Gd/Ex) |
| Kitchen Qual | Ordinal | Kitchen quality (NA/Po/Fa/TA/Gd/Ex) |
| Fireplace Qu | Ordinal | Fireplace quality (NA/Po/Fa/TA/Gd/Ex) |
| Neighborhood | Nominal | One of 28 Ames neighbourhoods |

---

## Project Structure

```
├── app/
│   ├── main.py                   # FastAPI app
│   ├── config.py                 # Pydantic settings (.env loader)
│   ├── chain/
│   │   └── pipeline.py           # AgentChain orchestrator
│   ├── llm/
│   │   ├── client.py             # GeminiClient (google-genai)
│   │   ├── stage1_extractor.py   # Feature extraction
│   │   ├── stage2_interpreter.py # Price interpretation
│   │   └── prompts/
│   │       ├── extraction_v1.txt # Few-shot prompt (production)
│   │       ├── extraction_v2.txt # Zero-shot prompt (comparison)
│   │       └── interpretation.txt
│   ├── ml/
│   │   ├── predictor.py          # HousePredictor (loads pipeline)
│   │   └── stats_loader.py       # TrainingStats (market context)
│   ├── schemas/
│   │   ├── features.py           # ExtractedFeatures + security validators
│   │   ├── prediction.py         # PredictionResult
│   │   └── response.py           # AgentResponse
│   └── utils/
│       ├── errors.py             # Custom exceptions
│       └── logger.py             # Structured logging
├── artifacts/
│   ├── pricing_pipeline.joblib   # Trained sklearn pipeline
│   ├── training_stats.json       # Model metrics + neighbourhood medians
│   └── feature_metadata.json     # Feature descriptions for user messages
├── notebooks/
│   └── HousePricingPredictor.ipynb
├── ui/
│   └── streamlit_app.py
├── tests/
│   ├── conftest.py
│   ├── test_schemas.py
│   ├── test_predictor.py
│   ├── test_llm_extractor.py
│   └── test_chain.py
├── scripts/
│   └── prompt_eval.py            # Compare v1 vs v2 prompt performance
├── Dockerfile
├── requirements.txt
└── .env.example
```

---

## Setup

### Prerequisites
- Python 3.11+
- Gemini API key ([aistudio.google.com](https://aistudio.google.com))

### Install

```bash
pip install -r requirements.txt
```

### Configure

```bash
cp .env.example .env
# Edit .env and add your key:
# GEMINI_API_KEY=your-key-here
```

---

## Running Locally

**Terminal 1 — API:**
```bash
uvicorn app.main:app --reload
```

**Terminal 2 — UI:**
```bash
streamlit run ui/streamlit_app.py
```

The UI defaults to `http://localhost:8000`. Override with:
```bash
API_URL=http://localhost:8000 streamlit run ui/streamlit_app.py
```

**API docs** available at `http://localhost:8000/docs`

---

## Running Tests

```bash
pytest tests/ -v
```

Tests mock all external calls (Gemini API, sklearn pipeline) — no API key required.

---

## Docker

```bash
docker build -t real-estate-agent .
docker run -p 8000:8000 -e GEMINI_API_KEY=your-key real-estate-agent
```

---

## Deployment

### FastAPI → Railway
1. Push repo to GitHub
2. New project on [railway.app](https://railway.app) → Deploy from GitHub
3. Add `GEMINI_API_KEY` in Railway **Variables**
4. Railway detects the `Dockerfile` and deploys automatically
5. Generate a domain under **Settings → Networking**

### Streamlit UI → Streamlit Community Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Connect repo → set main file to `ui/streamlit_app.py`
3. Under **Advanced settings → Secrets**, add:
```toml
API_URL = "https://your-api.up.railway.app"
```

---

## ML Results

| Model | Val RMSE | Val R² |
|---|---|---|
| Lasso | ~$38k | ~0.81 |
| Ridge | ~$36k | ~0.82 |
| ElasticNet | ~$38k | ~0.81 |
| XGBoost | ~$30k | ~0.87 |
| **Random Forest** | **$28,600** | **0.882** |

**Final test evaluation (Random Forest):**
- Test RMSE: $29,408
- Test R²: 0.873
