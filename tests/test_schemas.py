"""Tests for Pydantic feature schema — validation, security, and extraction helpers."""
import pytest
from pydantic import ValidationError

from app.schemas.features import ExtractedFeatures, validate_extraction


# ── ExtractedFeatures validation ──────────────────────────────────────────────

def test_valid_data_passes(valid_data):
    f = ExtractedFeatures.model_validate(valid_data)
    assert f.overall_qual == 8
    assert f.neighborhood == "StoneBr"
    assert f.exter_qual == "Ex"


def test_missing_field_raises(valid_data):
    del valid_data["overall_qual"]
    with pytest.raises(ValidationError):
        ExtractedFeatures.model_validate(valid_data)


def test_overall_qual_above_max_raises(valid_data):
    valid_data["overall_qual"] = 11
    with pytest.raises(ValidationError):
        ExtractedFeatures.model_validate(valid_data)


def test_overall_qual_below_min_raises(valid_data):
    valid_data["overall_qual"] = 0
    with pytest.raises(ValidationError):
        ExtractedFeatures.model_validate(valid_data)


def test_gr_liv_area_below_min_raises(valid_data):
    valid_data["gr_liv_area"] = 100
    with pytest.raises(ValidationError):
        ExtractedFeatures.model_validate(valid_data)


def test_year_built_out_of_range_raises(valid_data):
    valid_data["year_built"] = 1800
    with pytest.raises(ValidationError):
        ExtractedFeatures.model_validate(valid_data)


def test_invalid_quality_scale_raises(valid_data):
    valid_data["kitchen_qual"] = "Amazing"
    with pytest.raises(ValidationError):
        ExtractedFeatures.model_validate(valid_data)


def test_invalid_neighborhood_raises(valid_data):
    valid_data["neighborhood"] = "FakePlace"
    with pytest.raises(ValidationError):
        ExtractedFeatures.model_validate(valid_data)


def test_extra_field_rejected(valid_data):
    valid_data["price"] = 200000
    with pytest.raises(ValidationError):
        ExtractedFeatures.model_validate(valid_data)


# ── Security validators ───────────────────────────────────────────────────────

def test_sql_injection_rejected(valid_data):
    valid_data["neighborhood"] = "DROP TABLE houses"
    with pytest.raises(ValidationError):
        ExtractedFeatures.model_validate(valid_data)


def test_xss_injection_rejected(valid_data):
    valid_data["kitchen_qual"] = "<script>alert(1)</script>"
    with pytest.raises(ValidationError):
        ExtractedFeatures.model_validate(valid_data)


def test_value_too_long_rejected(valid_data):
    valid_data["neighborhood"] = "A" * 25
    with pytest.raises(ValidationError):
        ExtractedFeatures.model_validate(valid_data)


# ── validate_extraction ───────────────────────────────────────────────────────

def test_validate_extraction_complete(valid_data):
    result = validate_extraction(valid_data, "test query")
    assert result.is_complete
    assert result.missing_fields == []
    assert result.features is not None


def test_validate_extraction_null_field_is_missing(valid_data):
    valid_data["overall_qual"] = None
    valid_data["kitchen_qual"] = None
    result = validate_extraction(valid_data, "test")
    assert not result.is_complete
    assert "overall_qual" in result.missing_fields
    assert "kitchen_qual" in result.missing_fields


def test_validate_extraction_missing_details_populated(valid_data):
    valid_data["overall_qual"] = None
    result = validate_extraction(valid_data, "test")
    assert len(result.missing_details) == 1
    assert result.missing_details[0]["field"] == "overall_qual"


def test_completeness_ratio(valid_data):
    result = validate_extraction(valid_data, "test")
    assert result.completeness_ratio == 1.0


# ── to_model_input ────────────────────────────────────────────────────────────

def test_to_model_input_maps_column_names(valid_data):
    result = validate_extraction(valid_data, "test")
    row = result.to_model_input()
    assert "Overall Qual" in row
    assert "Gr Liv Area" in row
    assert "Neighborhood" in row
    assert row["Overall Qual"] == 8


def test_to_model_input_raises_when_incomplete(valid_data):
    valid_data["overall_qual"] = None
    result = validate_extraction(valid_data, "test")
    with pytest.raises(ValueError, match="incomplete"):
        result.to_model_input()
