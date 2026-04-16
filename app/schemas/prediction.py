from pydantic import BaseModel


class PredictionResult(BaseModel):
    """Carries the ML prediction and market context for Stage 2 interpretation."""

    predicted_price: float
    neighborhood: str
    neighborhood_median: float
    dataset_median: float
    price_tier: str          # "budget" | "mid_market" | "above_market" | "premium"
    pct_vs_neighborhood: float   # e.g. +12.4 means 12.4% above neighbourhood median
    pct_vs_dataset: float        # e.g. +38.1 means 38.1% above overall dataset median