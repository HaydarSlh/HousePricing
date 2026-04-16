from pydantic import BaseModel


class AgentResponse(BaseModel):
    """Final response returned by the chain to FastAPI and the Streamlit UI."""

    # Price
    predicted_price: float
    price_tier: str                 # budget | mid_market | above_market | premium
    pct_vs_neighborhood: float      # e.g. +12.4
    pct_vs_dataset: float           # e.g. +38.1
    neighborhood_median: float
    dataset_median: float

    # LLM interpretation
    interpretation: str

    # Feature snapshot (for UI display)
    extracted_features: dict
