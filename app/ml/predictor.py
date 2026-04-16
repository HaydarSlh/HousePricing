import logging

import joblib
import pandas as pd

from app.config import get_settings
from app.ml.stats_loader import TrainingStats
from app.schemas.features import ExtractionResult
from app.schemas.prediction import PredictionResult
from app.utils.errors import PredictionError

logger = logging.getLogger("app.ml.predictor")


class HousePredictor:
    """Loads pricing_pipeline.joblib once and runs inference on demand."""

    def __init__(self, settings=None):
        self._settings = settings or get_settings()
        path = self._settings.pipeline_path
        if not path.exists():
            raise FileNotFoundError(
                f"Pipeline not found at '{path}'. "
                "Run the notebook to generate artifacts/pricing_pipeline.joblib."
            )
        self._pipeline = joblib.load(path)
        self._stats = TrainingStats(settings=self._settings)
        logger.info("HousePredictor loaded pipeline from %s", path)

    def predict(self, result: ExtractionResult) -> PredictionResult:
        """Run the sklearn pipeline and return a PredictionResult with market context.

        Args:
            result: A *complete* ExtractionResult (is_complete must be True).

        Raises:
            ValueError: If the extraction result is incomplete.
            PredictionError: If the pipeline raises during inference.
        """
        row = result.to_model_input()  # raises ValueError if is_complete is False
        df = pd.DataFrame([row])

        try:
            price = float(self._pipeline.predict(df)[0])
        except Exception as exc:
            raise PredictionError(f"Pipeline prediction failed: {exc}") from exc

        logger.info("Predicted price: $%.2f for neighbourhood %s",
                    price, result.features.neighborhood)

        return self._stats.get_price_context(price, result.features.neighborhood)
