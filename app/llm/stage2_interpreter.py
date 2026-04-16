import logging
from pathlib import Path

from app.llm.client import GeminiClient
from app.schemas.features import ExtractionResult
from app.schemas.prediction import PredictionResult
from app.utils.errors import LLMAPIError

logger = logging.getLogger("app.llm.stage2")

PROMPT_DIR = Path(__file__).parent / "prompts"


class PriceInterpreter:
    """Stage 2: ML prediction + market context → natural language interpretation."""

    def __init__(self, client: GeminiClient | None = None):
        self._client = client or GeminiClient()
        path = PROMPT_DIR / "interpretation.txt"
        if not path.exists():
            raise FileNotFoundError(f"Interpretation prompt not found: {path}")
        self._prompt_template = path.read_text(encoding="utf-8")

    def interpret(self, result: ExtractionResult, prediction: PredictionResult) -> str:
        """Generate a natural language interpretation of the price prediction.

        Args:
            result: Completed ExtractionResult with all 12 features.
            prediction: PredictionResult from HousePredictor.

        Returns:
            A 2–3 sentence interpretation string.

        Raises:
            LLMAPIError: If Gemini fails after all retries.
        """
        f = result.features
        context = {
            # Prediction context
            "predicted_price":    f"${prediction.predicted_price:,.0f}",
            "neighborhood":       prediction.neighborhood,
            "price_tier":         prediction.price_tier.replace("_", " "),
            "neighborhood_median": f"${prediction.neighborhood_median:,.0f}",
            "dataset_median":     f"${prediction.dataset_median:,.0f}",
            "pct_vs_neighborhood": f"{prediction.pct_vs_neighborhood:+.1f}%",
            "pct_vs_dataset":     f"{prediction.pct_vs_dataset:+.1f}%",
            # House features
            "overall_qual":  f"{f.overall_qual}/10",
            "gr_liv_area":   f"{f.gr_liv_area:,} sqft",
            "year_built":    str(f.year_built),
            "exter_qual":    f.exter_qual,
            "kitchen_qual":  f.kitchen_qual,
            "bsmt_qual":     f.bsmt_qual,
            "fireplace_qu":  f.fireplace_qu,
            "garage_area":   f"{f.garage_area} sqft",
            "total_bsmt_sf": f"{f.total_bsmt_sf} sqft",
            "full_bath":     str(f.full_bath),
            "mas_vnr_area":  f"{f.mas_vnr_area} sqft",
        }

        prompt = self._prompt_template
        for key, value in context.items():
            prompt = prompt.replace(f"{{{key}}}", value)

        try:
            raw = self._client.generate_json(prompt)
        except LLMAPIError:
            raise

        interpretation = raw.get("interpretation", "").strip()

        if not interpretation:
            # Fallback: build a minimal interpretation from the data directly
            interpretation = (
                f"Based on the provided features, this {prediction.neighborhood} home is "
                f"estimated at ${prediction.predicted_price:,.0f} — "
                f"{abs(prediction.pct_vs_neighborhood):.1f}% "
                f"{'above' if prediction.pct_vs_neighborhood >= 0 else 'below'} "
                f"the neighbourhood median."
            )
            logger.warning("LLM returned empty interpretation; using fallback.")

        logger.info("Stage 2 interpretation generated (%d chars)", len(interpretation))
        return interpretation
