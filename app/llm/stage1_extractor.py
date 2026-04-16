import json
import logging
from pathlib import Path

from app.config import get_settings
from app.llm.client import GeminiClient
from app.schemas.features import ExtractionResult, validate_extraction
from app.utils.errors import ExtractionError, LLMAPIError, SchemaValidationError

logger = logging.getLogger("app.llm.stage1")

PROMPT_DIR = Path(__file__).parent / "prompts"


class FeatureExtractor:
    """Stage 1: natural language property description → structured features."""

    def __init__(self, prompt_version: str = "v1", client: GeminiClient | None = None):
        self._client = client or GeminiClient()
        self._prompt_template = self._load_prompt(prompt_version)
        self._neighborhood_list = self._load_neighborhoods()
        self._version = prompt_version

    def _load_prompt(self, version: str) -> str:
        path = PROMPT_DIR / f"extraction_{version}.txt"
        if not path.exists():
            raise FileNotFoundError(f"Prompt template not found: {path}")
        return path.read_text(encoding="utf-8")

    def _load_neighborhoods(self) -> str:
        settings = get_settings()
        with open(settings.training_stats_path, encoding="utf-8") as f:
            stats = json.load(f)
        categories = stats["features"]["nominal"]["Neighborhood"]["categories"]
        return ", ".join(categories)

    def extract(self, user_query: str) -> ExtractionResult:
        """Run one extraction attempt. Returns ExtractionResult (always).

        If all 12 features are valid → is_complete=True.
        If any are missing/invalid → is_complete=False with missing_details
        populated so the caller can tell the user what to provide.
        """
        # Build prompt
        prompt = (
            self._prompt_template
            .replace("{user_query}", user_query)
            .replace("{neighborhood_list}", self._neighborhood_list)
        )

        # Call Gemini
        try:
            raw = self._client.generate_json(prompt)
        except LLMAPIError:
            raise
        except Exception as exc:
            raise LLMAPIError(f"Unexpected error calling Gemini: {exc}") from exc

        logger.info("Raw LLM response (v%s): %s", self._version, json.dumps(raw))

        # Strip non-feature keys the LLM might add (e.g. "analysis" from v2)
        feature_keys = {
            "overall_qual", "gr_liv_area", "garage_area", "total_bsmt_sf",
            "year_built", "full_bath", "mas_vnr_area", "bsmt_qual",
            "exter_qual", "kitchen_qual", "fireplace_qu", "neighborhood",
        }
        cleaned = {k: v for k, v in raw.items() if k in feature_keys}

        # Validate through Pydantic
        result = validate_extraction(cleaned, user_query)

        logger.info(
            "Extraction (v%s): %d/12 features — %s",
            self._version,
            len(result.extracted_fields),
            "COMPLETE" if result.is_complete else f"missing {result.missing_fields}",
        )

        return result
