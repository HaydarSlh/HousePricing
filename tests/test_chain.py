"""Tests for AgentChain — all three components are mocked."""
from unittest.mock import MagicMock, patch

import pytest

from app.chain.pipeline import AgentChain, ChainResult, get_chain
from app.schemas.features import validate_extraction
from app.schemas.prediction import PredictionResult
from app.schemas.response import AgentResponse


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_prediction(price: float = 220_000.0) -> PredictionResult:
    return PredictionResult(
        predicted_price=price,
        neighborhood="StoneBr",
        neighborhood_median=334_582.0,
        dataset_median=160_100.0,
        price_tier="above_market",
        pct_vs_neighborhood=-34.2,
        pct_vs_dataset=37.4,
    )


def make_chain(
    valid_data: dict,
    price: float = 220_000.0,
    interpretation: str = "Test interpretation.",
) -> AgentChain:
    """Build an AgentChain with all external dependencies mocked."""
    complete_extraction = validate_extraction(valid_data, "test")
    mock_prediction = make_prediction(price)

    mock_extractor = MagicMock()
    mock_extractor.extract.return_value = complete_extraction

    mock_predictor = MagicMock()
    mock_predictor.predict.return_value = mock_prediction

    mock_interpreter = MagicMock()
    mock_interpreter.interpret.return_value = interpretation

    with (
        patch("app.chain.pipeline.FeatureExtractor", return_value=mock_extractor),
        patch("app.chain.pipeline.HousePredictor", return_value=mock_predictor),
        patch("app.chain.pipeline.PriceInterpreter", return_value=mock_interpreter),
    ):
        return AgentChain()


def make_incomplete_chain(valid_data: dict) -> AgentChain:
    """Build a chain whose extractor always returns an incomplete result."""
    valid_data["overall_qual"] = None
    incomplete = validate_extraction(valid_data, "test")

    mock_extractor = MagicMock()
    mock_extractor.extract.return_value = incomplete

    with (
        patch("app.chain.pipeline.FeatureExtractor", return_value=mock_extractor),
        patch("app.chain.pipeline.HousePredictor"),
        patch("app.chain.pipeline.PriceInterpreter"),
    ):
        return AgentChain()


# ── Incomplete extraction ─────────────────────────────────────────────────────

def test_incomplete_extraction_returns_is_complete_false(valid_data):
    chain = make_incomplete_chain(valid_data)
    result = chain.run("sparse query")
    assert not result.is_complete


def test_incomplete_extraction_response_is_none(valid_data):
    chain = make_incomplete_chain(valid_data)
    result = chain.run("sparse query")
    assert result.response is None


def test_incomplete_extraction_missing_fields_populated(valid_data):
    chain = make_incomplete_chain(valid_data)
    result = chain.run("sparse query")
    assert "overall_qual" in result.missing_fields


def test_incomplete_extraction_missing_details_populated(valid_data):
    chain = make_incomplete_chain(valid_data)
    result = chain.run("sparse query")
    assert len(result.missing_details) >= 1


# ── Complete extraction ───────────────────────────────────────────────────────

def test_complete_extraction_returns_is_complete_true(valid_data):
    chain = make_chain(valid_data)
    result = chain.run("full description")
    assert result.is_complete


def test_complete_extraction_response_is_agent_response(valid_data):
    chain = make_chain(valid_data)
    result = chain.run("full description")
    assert isinstance(result.response, AgentResponse)


def test_response_price_matches_predictor(valid_data):
    chain = make_chain(valid_data, price=220_000.0)
    result = chain.run("test")
    assert result.response.predicted_price == 220_000.0


def test_response_interpretation_matches_interpreter(valid_data):
    chain = make_chain(valid_data, interpretation="Great value home.")
    result = chain.run("test")
    assert result.response.interpretation == "Great value home."


def test_response_contains_extracted_features(valid_data):
    chain = make_chain(valid_data)
    result = chain.run("test")
    assert "neighborhood" in result.response.extracted_features
    assert result.response.extracted_features["neighborhood"] == "StoneBr"


def test_response_market_context_fields_present(valid_data):
    chain = make_chain(valid_data)
    result = chain.run("test")
    r = result.response
    assert r.price_tier == "above_market"
    assert r.neighborhood_median == 334_582.0
    assert r.dataset_median == 160_100.0


# ── ChainResult schema ────────────────────────────────────────────────────────

def test_chain_result_is_pydantic_model(valid_data):
    chain = make_chain(valid_data)
    result = chain.run("test")
    assert isinstance(result, ChainResult)
    # Serialisable to dict (FastAPI needs this)
    assert isinstance(result.model_dump(), dict)
