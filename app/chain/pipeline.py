import logging

from pydantic import BaseModel

from app.llm.stage1_extractor import FeatureExtractor
from app.llm.stage2_interpreter import PriceInterpreter
from app.ml.predictor import HousePredictor
from app.schemas.response import AgentResponse

logger = logging.getLogger("app.chain")


class ChainResult(BaseModel):
    """Returned by AgentChain.run() on every call.

    If is_complete is False the caller should surface missing_details to the
    user and call run() again with the enriched query.
    If is_complete is True, response is populated and ready to return.
    """

    is_complete: bool
    response: AgentResponse | None = None
    missing_fields: list[str] = []
    missing_details: list[dict] = []
    extracted_fields: list[str] = []   # what was found — shown to user during review
    extracted_values: dict = {}        # field → value for already-extracted features


class AgentChain:
    """Orchestrates Stage 1 → ML → Stage 2.

    Instantiate once at app startup (loads the sklearn pipeline from disk).
    Call run(user_query) for every user turn.

    Multi-turn strategy
    -------------------
    The chain is stateless — it does not remember previous turns.
    The caller (FastAPI endpoint / Streamlit UI) is responsible for
    accumulating context: when the user provides missing information,
    append it to the original query and call run() again with the
    combined text.
    """

    def __init__(self):
        self._extractor = FeatureExtractor(prompt_version="v1")
        self._predictor = HousePredictor()
        self._interpreter = PriceInterpreter()
        logger.info("AgentChain initialised (Stage 1 + ML + Stage 2 ready)")

    def run(self, user_query: str) -> ChainResult:
        """Run the full pipeline for one user turn.

        Args:
            user_query: The accumulated conversation text from the user.

        Returns:
            ChainResult — either complete (with AgentResponse) or
            incomplete (with missing_details for the caller to show).
        """
        logger.info("Chain run started | query length=%d chars", len(user_query))

        # ── Stage 1: extract features ────────────────────────────────────────
        extraction = self._extractor.extract(user_query)

        if not extraction.is_complete:
            logger.info(
                "Extraction incomplete — missing %d fields: %s",
                len(extraction.missing_fields),
                extraction.missing_fields,
            )
            return ChainResult(
                is_complete=False,
                missing_fields=extraction.missing_fields,
                missing_details=extraction.missing_details,
                extracted_fields=extraction.extracted_fields,
                extracted_values=extraction.partial_values,
            )

        # ── ML: predict price ────────────────────────────────────────────────
        prediction = self._predictor.predict(extraction)
        logger.info(
            "Prediction: $%.0f | tier=%s | vs_neighbourhood=%+.1f%%",
            prediction.predicted_price,
            prediction.price_tier,
            prediction.pct_vs_neighborhood,
        )

        # ── Stage 2: interpret prediction ────────────────────────────────────
        interpretation = self._interpreter.interpret(extraction, prediction)

        # ── Assemble final response ──────────────────────────────────────────
        response = AgentResponse(
            predicted_price=prediction.predicted_price,
            price_tier=prediction.price_tier,
            pct_vs_neighborhood=prediction.pct_vs_neighborhood,
            pct_vs_dataset=prediction.pct_vs_dataset,
            neighborhood_median=prediction.neighborhood_median,
            dataset_median=prediction.dataset_median,
            interpretation=interpretation,
            extracted_features=extraction.features.model_dump(),
        )

        logger.info("Chain run complete")
        return ChainResult(is_complete=True, response=response)


# ── Module-level singleton ───────────────────────────────────────────────────
# The sklearn pipeline is ~100 MB — load once at startup, not per request.
_chain: AgentChain | None = None


def get_chain() -> AgentChain:
    """Return the shared AgentChain instance, creating it on first call."""
    global _chain
    if _chain is None:
        _chain = AgentChain()
    return _chain
