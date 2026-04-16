import json
import logging

from app.config import get_settings
from app.schemas.prediction import PredictionResult

logger = logging.getLogger("app.ml.stats_loader")


class TrainingStats:
    """Loads training_stats.json once and provides price context helpers."""

    # IQR-based tier thresholds (from training data)
    _IQR_LOW = 129_425.0
    _IQR_HIGH = 210_000.0
    _PREMIUM_THRESHOLD = 300_000.0

    def __init__(self, settings=None):
        self._settings = settings or get_settings()
        path = self._settings.training_stats_path
        if not path.exists():
            raise FileNotFoundError(
                f"Training stats not found at '{path}'. "
                "Run the notebook to generate artifacts/training_stats.json."
            )
        with open(path, encoding="utf-8") as f:
            self._stats = json.load(f)

        self._dataset_median: float = self._stats["target"]["median"]
        self._neighborhood_medians: dict[str, float] = {
            k: float(v) for k, v in self._stats["neighborhood_medians"].items()
        }
        logger.info("TrainingStats loaded from %s", path)

    def get_neighborhood_median(self, neighborhood: str) -> float:
        """Return neighbourhood median, falling back to dataset median if unknown."""
        return self._neighborhood_medians.get(neighborhood, self._dataset_median)

    def get_price_context(self, predicted_price: float, neighborhood: str) -> PredictionResult:
        """Compute tier and % comparisons, return a PredictionResult."""
        neigh_median = self.get_neighborhood_median(neighborhood)

        pct_vs_neighborhood = (predicted_price - neigh_median) / neigh_median * 100
        pct_vs_dataset = (predicted_price - self._dataset_median) / self._dataset_median * 100

        if predicted_price < self._IQR_LOW:
            tier = "budget"
        elif predicted_price <= self._IQR_HIGH:
            tier = "mid_market"
        elif predicted_price <= self._PREMIUM_THRESHOLD:
            tier = "above_market"
        else:
            tier = "premium"

        return PredictionResult(
            predicted_price=round(predicted_price, 2),
            neighborhood=neighborhood,
            neighborhood_median=neigh_median,
            dataset_median=self._dataset_median,
            price_tier=tier,
            pct_vs_neighborhood=round(pct_vs_neighborhood, 1),
            pct_vs_dataset=round(pct_vs_dataset, 1),
        )
