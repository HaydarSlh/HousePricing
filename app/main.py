import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.chain.pipeline import ChainResult, get_chain
from app.utils.errors import LLMAPIError, PredictionError
from app.utils.logger import setup_logger

setup_logger("app")
logger = logging.getLogger("app.main")


# ── Startup / shutdown ───────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the sklearn pipeline and Gemini client once at startup."""
    logger.info("Loading AgentChain (pipeline + LLM clients)...")
    get_chain()
    logger.info("AgentChain ready — accepting requests")
    yield
    logger.info("Shutting down")


# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AI Real Estate Agent",
    description="Natural language property description → price estimate + interpretation",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


# ── Request schema ───────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    query: str


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    """Railway health-check probe."""
    return {"status": "ok"}


@app.post("/predict", response_model=ChainResult)
def predict(request: PredictRequest):
    """Run the two-stage agent pipeline.

    Always returns 200 with a ChainResult.

    - is_complete=False  → missing_details tells the caller which features
                           are needed; append the user's answer to the
                           original query and call /predict again.
    - is_complete=True   → response contains the price + interpretation.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    chain = get_chain()
    try:
        return chain.run(request.query)
    except LLMAPIError as exc:
        logger.error("LLM API failure: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="The AI service is temporarily unavailable. Please try again shortly.",
        )
    except PredictionError as exc:
        logger.error("Prediction failure: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="The price prediction model failed. Please try again.",
        )
    except Exception as exc:
        logger.error("Unexpected chain error: %s", exc)
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")
