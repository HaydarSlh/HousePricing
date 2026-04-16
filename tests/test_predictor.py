"""Tests for TrainingStats and HousePredictor."""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.ml.stats_loader import TrainingStats
from app.ml.predictor import HousePredictor
from app.schemas.features import validate_extraction
from app.utils.errors import PredictionError


# ── TrainingStats ─────────────────────────────────────────────────────────────

class TestTrainingStats:

    @pytest.fixture(autouse=True)
    def setup(self, mock_settings):
        self.stats = TrainingStats(settings=mock_settings)

    def test_loads_dataset_median(self):
        assert self.stats._dataset_median == 160100.0

    def test_known_neighborhood_median(self):
        assert self.stats.get_neighborhood_median("StoneBr") == 334582.0

    def test_unknown_neighborhood_falls_back_to_dataset_median(self):
        result = self.stats.get_neighborhood_median("UnknownPlace")
        assert result == self.stats._dataset_median

    def test_budget_tier(self):
        result = self.stats.get_price_context(100_000.0, "NAmes")
        assert result.price_tier == "budget"

    def test_mid_market_tier(self):
        result = self.stats.get_price_context(170_000.0, "NAmes")
        assert result.price_tier == "mid_market"

    def test_above_market_tier(self):
        result = self.stats.get_price_context(250_000.0, "NAmes")
        assert result.price_tier == "above_market"

    def test_premium_tier(self):
        result = self.stats.get_price_context(400_000.0, "StoneBr")
        assert result.price_tier == "premium"

    def test_pct_vs_dataset_at_median_is_zero(self):
        result = self.stats.get_price_context(160_100.0, "NAmes")
        assert result.pct_vs_dataset == 0.0

    def test_pct_vs_neighborhood_computed_correctly(self):
        # StoneBr median = 334582 → price 334582 should be 0%
        result = self.stats.get_price_context(334_582.0, "StoneBr")
        assert result.pct_vs_neighborhood == 0.0

    def test_result_carries_neighborhood(self):
        result = self.stats.get_price_context(200_000.0, "CollgCr")
        assert result.neighborhood == "CollgCr"

    def test_missing_stats_file_raises(self, tmp_path):
        bad_settings = MagicMock()
        bad_settings.training_stats_path = tmp_path / "nonexistent.json"
        with pytest.raises(FileNotFoundError):
            TrainingStats(settings=bad_settings)


# ── HousePredictor ────────────────────────────────────────────────────────────

class TestHousePredictor:

    @pytest.fixture
    def mock_pipeline(self):
        pipeline = MagicMock()
        pipeline.predict.return_value = [220_000.0]
        return pipeline

    @pytest.fixture
    def predictor(self, mock_pipeline, mock_settings):
        with patch("app.ml.predictor.joblib.load", return_value=mock_pipeline):
            return HousePredictor(settings=mock_settings)

    def test_predict_returns_prediction_result(self, predictor, valid_data):
        extraction = validate_extraction(valid_data, "test")
        result = predictor.predict(extraction)
        assert result.predicted_price == 220_000.0
        assert result.neighborhood == "StoneBr"

    def test_predict_sets_price_tier(self, predictor, valid_data):
        extraction = validate_extraction(valid_data, "test")
        result = predictor.predict(extraction)
        assert result.price_tier in ("budget", "mid_market", "above_market", "premium")

    def test_predict_raises_on_incomplete_extraction(self, predictor, valid_data):
        valid_data["overall_qual"] = None
        incomplete = validate_extraction(valid_data, "test")
        with pytest.raises(ValueError):
            predictor.predict(incomplete)

    def test_predict_wraps_pipeline_error(self, mock_settings, valid_data):
        broken_pipeline = MagicMock()
        broken_pipeline.predict.side_effect = RuntimeError("model error")
        with patch("app.ml.predictor.joblib.load", return_value=broken_pipeline):
            predictor = HousePredictor(settings=mock_settings)
        extraction = validate_extraction(valid_data, "test")
        with pytest.raises(PredictionError):
            predictor.predict(extraction)

    def test_missing_pipeline_file_raises(self, tmp_path, mock_settings):
        mock_settings.pipeline_path = tmp_path / "nonexistent.joblib"
        with pytest.raises(FileNotFoundError):
            HousePredictor(settings=mock_settings)
