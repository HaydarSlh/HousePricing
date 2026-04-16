"""Tests for Stage 1 FeatureExtractor — mocks the Gemini client."""
import pytest
from unittest.mock import MagicMock

from app.llm.stage1_extractor import FeatureExtractor
from app.utils.errors import LLMAPIError


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_extractor(llm_response: dict) -> FeatureExtractor:
    """Return a FeatureExtractor backed by a mock client."""
    mock_client = MagicMock()
    mock_client.generate_json.return_value = llm_response
    return FeatureExtractor(prompt_version="v1", client=mock_client)


# ── Complete response ─────────────────────────────────────────────────────────

def test_complete_response_is_complete(valid_data):
    extractor = make_extractor(valid_data)
    result = extractor.extract("full description")
    assert result.is_complete
    assert result.missing_fields == []


def test_complete_response_extracts_all_12_features(valid_data):
    extractor = make_extractor(valid_data)
    result = extractor.extract("full description")
    assert len(result.extracted_fields) == 12


def test_complete_response_values_correct(valid_data):
    extractor = make_extractor(valid_data)
    result = extractor.extract("test")
    assert result.features.overall_qual == 8
    assert result.features.neighborhood == "StoneBr"


# ── Partial response ──────────────────────────────────────────────────────────

def test_partial_response_is_incomplete(valid_data):
    valid_data["overall_qual"] = None
    valid_data["kitchen_qual"] = None
    extractor = make_extractor(valid_data)
    result = extractor.extract("partial description")
    assert not result.is_complete


def test_partial_response_missing_fields_identified(valid_data):
    valid_data["overall_qual"] = None
    valid_data["kitchen_qual"] = None
    extractor = make_extractor(valid_data)
    result = extractor.extract("partial description")
    assert "overall_qual" in result.missing_fields
    assert "kitchen_qual" in result.missing_fields


def test_missing_details_populated_for_missing_fields(valid_data):
    valid_data["overall_qual"] = None
    extractor = make_extractor(valid_data)
    result = extractor.extract("test")
    assert len(result.missing_details) == 1
    assert result.missing_details[0]["field"] == "overall_qual"
    assert "description" in result.missing_details[0]


# ── Extra keys from LLM ───────────────────────────────────────────────────────

def test_extra_llm_keys_are_stripped(valid_data):
    """LLM sometimes adds 'analysis' or 'reasoning' keys — they must be removed."""
    valid_data["analysis"] = "Here is my reasoning..."
    valid_data["confidence"] = 0.95
    extractor = make_extractor(valid_data)
    result = extractor.extract("test")
    assert result.is_complete


# ── Error handling ────────────────────────────────────────────────────────────

def test_llm_api_error_propagates(valid_data):
    mock_client = MagicMock()
    mock_client.generate_json.side_effect = LLMAPIError("Gemini unavailable")
    extractor = FeatureExtractor(prompt_version="v1", client=mock_client)
    with pytest.raises(LLMAPIError):
        extractor.extract("test query")


def test_absence_signals_map_to_na(valid_data):
    """'no basement', 'no fireplace' → LLM should return 'NA' — validate it passes."""
    valid_data["bsmt_qual"] = "NA"
    valid_data["fireplace_qu"] = "NA"
    valid_data["garage_area"] = 0
    valid_data["total_bsmt_sf"] = 0
    extractor = make_extractor(valid_data)
    result = extractor.extract("no basement, no fireplace, no garage")
    assert result.is_complete
    assert result.features.bsmt_qual == "NA"
    assert result.features.fireplace_qu == "NA"
