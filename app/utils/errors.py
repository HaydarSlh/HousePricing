class ExtractionError(Exception):
    """Base error for the feature extraction stage."""


class LLMAPIError(ExtractionError):
    """Gemini API call failed after all retries."""


class SchemaValidationError(ExtractionError):
    """LLM output failed Pydantic validation."""


class PredictionError(Exception):
    """sklearn pipeline raised during inference."""
