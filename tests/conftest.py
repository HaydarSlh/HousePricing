"""Shared pytest fixtures."""
from pathlib import Path
from unittest.mock import MagicMock

import pytest


# ── Reusable valid feature dict ───────────────────────────────────────────────
@pytest.fixture
def valid_data():
    return {
        "overall_qual":  8,
        "gr_liv_area":   2400,
        "garage_area":   600,
        "total_bsmt_sf": 1200,
        "year_built":    2005,
        "full_bath":     2,
        "mas_vnr_area":  200,
        "bsmt_qual":     "Gd",
        "exter_qual":    "Ex",
        "kitchen_qual":  "Ex",
        "fireplace_qu":  "Gd",
        "neighborhood":  "StoneBr",
    }


# ── Mock settings (avoids needing a real .env in CI) ─────────────────────────
@pytest.fixture
def mock_settings():
    s = MagicMock()
    s.training_stats_path = Path("artifacts/training_stats.json")
    s.pipeline_path = Path("artifacts/pricing_pipeline.joblib")
    s.model_name = "gemini-2.5-flash"
    s.llm_temperature = 0.0
    s.llm_max_output_tokens = 2048
    s.max_retries = 3
    return s
