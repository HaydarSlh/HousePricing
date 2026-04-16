from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ── Quality scale shared by all 4 ordinal features ──────────────────────────
QualityScale = Literal["NA", "Po", "Fa", "TA", "Gd", "Ex"]

# ── 28 valid Ames neighbourhoods ────────────────────────────────────────────
NeighborhoodType = Literal[
    "Blmngtn", "Blueste", "BrDale", "BrkSide", "ClearCr", "CollgCr",
    "Crawfor", "Edwards", "Gilbert", "Greens", "GrnHill", "IDOTRR",
    "Landmrk", "MeadowV", "Mitchel", "NAmes", "NPkVill", "NWAmes",
    "NoRidge", "NridgHt", "OldTown", "SWISU", "Sawyer", "SawyerW",
    "Somerst", "StoneBr", "Timber", "Veenker",
]

# Regex for suspicious injection patterns
_INJECTION_RE = re.compile(
    r"(\{|\}|<script|__"
    r"|DROP\s|SELECT\s|INSERT\s|DELETE\s|UNION\s|UPDATE\s|ALTER\s)",
    re.IGNORECASE,
)


class ExtractedFeatures(BaseModel):
    """All 12 features the ML pipeline expects. Every field is REQUIRED."""

    model_config = ConfigDict(extra="forbid")

    # ── Numeric (7) ─────────────────────────────────────────────────────────
    overall_qual: int = Field(ge=1, le=10)
    gr_liv_area: int = Field(ge=334, le=5095)
    garage_area: int = Field(ge=0, le=1488)
    total_bsmt_sf: int = Field(ge=0, le=5095)
    year_built: int = Field(ge=1872, le=2010)
    full_bath: int = Field(ge=0, le=4)
    mas_vnr_area: int = Field(ge=0, le=1290)

    # ── Ordinal (4) ─────────────────────────────────────────────────────────
    bsmt_qual: QualityScale
    exter_qual: QualityScale
    kitchen_qual: QualityScale
    fireplace_qu: QualityScale

    # ── Nominal (1) ─────────────────────────────────────────────────────────
    neighborhood: NeighborhoodType

    # ── Security validators ─────────────────────────────────────────────────
    @field_validator("bsmt_qual", "exter_qual", "kitchen_qual",
                     "fireplace_qu", "neighborhood", mode="before")
    @classmethod
    def reject_injection(cls, v: Any) -> Any:
        if not isinstance(v, str):
            return v
        if len(v) > 20:
            raise ValueError(f"Value too long ({len(v)} chars), max 20")
        if _INJECTION_RE.search(v):
            raise ValueError("Suspicious pattern detected in value")
        return v.strip()

    @model_validator(mode="before")
    @classmethod
    def reject_unknown_keys(cls, data: Any) -> Any:
        if isinstance(data, dict):
            allowed = set(cls.model_fields.keys())
            extra = set(data.keys()) - allowed
            if extra:
                raise ValueError(
                    f"Unexpected fields rejected: {extra}"
                )
        return data


# ── Snake-case → original column name mapping ──────────────────────────────
FIELD_NAME_MAP = {
    "overall_qual": "Overall Qual",
    "gr_liv_area": "Gr Liv Area",
    "garage_area": "Garage Area",
    "total_bsmt_sf": "Total Bsmt SF",
    "year_built": "Year Built",
    "full_bath": "Full Bath",
    "mas_vnr_area": "Mas Vnr Area",
    "bsmt_qual": "Bsmt Qual",
    "exter_qual": "Exter Qual",
    "kitchen_qual": "Kitchen Qual",
    "fireplace_qu": "Fireplace Qu",
    "neighborhood": "Neighborhood",
}


def _load_feature_metadata() -> dict:
    """Load feature_metadata.json for user-facing descriptions."""
    path = Path("artifacts/feature_metadata.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _build_missing_detail(field_name: str, metadata: dict) -> dict:
    """Build a user-facing description of a missing/invalid field."""
    for category in ("numeric_features", "ordinal_features", "nominal_features"):
        if field_name in metadata.get(category, {}):
            info = metadata[category][field_name]
            detail = {
                "field": field_name,
                "display_name": info.get("display_name", field_name),
                "description": info.get("description", ""),
            }
            if "min" in info and "max" in info:
                unit = info.get("unit", "")
                detail["valid_range"] = f"{info['min']}–{info['max']}" + (
                    f" {unit}" if unit else ""
                )
            if "scale" in info:
                detail["valid_options"] = info["scale"]
            if "valid_values" in info:
                detail["valid_options"] = info["valid_values"]
            return detail
    return {"field": field_name, "description": "Unknown feature"}


class ExtractionResult(BaseModel):
    """Wraps the extraction attempt — may be complete or missing fields."""

    features: Optional[ExtractedFeatures] = None
    extracted_fields: list[str] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    missing_details: list[dict] = Field(default_factory=list)
    is_complete: bool = False
    raw_query: str = ""

    def to_model_input(self) -> dict:
        """Convert to the dict the sklearn pipeline expects.

        Raises ValueError if extraction is incomplete.
        """
        if not self.is_complete or self.features is None:
            raise ValueError(
                "Cannot build model input — extraction is incomplete. "
                f"Missing: {self.missing_fields}"
            )
        return {
            FIELD_NAME_MAP[k]: v
            for k, v in self.features.model_dump().items()
        }

    @property
    def completeness_ratio(self) -> float:
        return len(self.extracted_fields) / 12


def validate_extraction(raw_data: dict, user_query: str) -> ExtractionResult:
    """Try to validate raw LLM JSON against ExtractedFeatures.

    Returns an ExtractionResult — always succeeds, but is_complete may be False.
    """
    all_fields = set(ExtractedFeatures.model_fields.keys())
    metadata = _load_feature_metadata()

    try:
        features = ExtractedFeatures.model_validate(raw_data)
        return ExtractionResult(
            features=features,
            extracted_fields=sorted(all_fields),
            missing_fields=[],
            missing_details=[],
            is_complete=True,
            raw_query=user_query,
        )
    except Exception as exc:
        # Parse which fields failed — collect from Pydantic errors
        failed_fields: set[str] = set()

        if hasattr(exc, "errors"):
            for err in exc.errors():
                loc = err.get("loc", ())
                if loc:
                    failed_fields.add(str(loc[0]))

        # Also flag fields that are null / missing from the raw dict
        for field in all_fields:
            if field not in raw_data or raw_data[field] is None:
                failed_fields.add(field)

        # Build extracted vs missing
        extracted = sorted(all_fields - failed_fields)
        missing = sorted(failed_fields)
        missing_details = [_build_missing_detail(f, metadata) for f in missing]

        return ExtractionResult(
            features=None,
            extracted_fields=extracted,
            missing_fields=missing,
            missing_details=missing_details,
            is_complete=False,
            raw_query=user_query,
        )
