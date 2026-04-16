import json
import logging
import re
import time

from google import genai
from google.genai import types

from app.config import get_settings
from app.utils.errors import LLMAPIError

logger = logging.getLogger("app.llm.client")

# Regex to strip markdown code fences Gemini sometimes wraps around JSON
_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?", re.MULTILINE)


class GeminiClient:
    """Thin wrapper around google-genai for structured JSON output."""

    def __init__(self, settings=None):
        self._settings = settings or get_settings()
        self._client = genai.Client(api_key=self._settings.gemini_api_key)
        self._generation_config = types.GenerateContentConfig(
            temperature=self._settings.llm_temperature,
            max_output_tokens=self._settings.llm_max_output_tokens,
            response_mime_type="application/json",
        )

    def generate_json(self, prompt: str) -> dict:
        """Send a prompt and return parsed JSON. Retries on transient errors."""
        last_error = None

        for attempt in range(1, self._settings.max_retries + 1):
            try:
                response = self._client.models.generate_content(
                    model=self._settings.model_name,
                    contents=prompt,
                    config=self._generation_config,
                )
                text = response.text.strip()

                # Strip markdown code fences if present
                text = _FENCE_RE.sub("", text).rstrip("`").strip()

                return json.loads(text)

            except json.JSONDecodeError as exc:
                logger.warning("Attempt %d: JSON parse failed: %s", attempt, exc)
                last_error = exc

            except Exception as exc:
                logger.warning("Attempt %d: API error: %s", attempt, exc)
                last_error = exc
                if attempt < self._settings.max_retries:
                    time.sleep(2 ** attempt)

        raise LLMAPIError(
            f"Gemini API failed after {self._settings.max_retries} attempts: "
            f"{last_error}"
        )
