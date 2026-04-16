from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # Gemini
    gemini_api_key: str
    model_name: str = "gemini-2.5-flash"

    # LLM behaviour
    llm_temperature: float = 0.0
    llm_max_output_tokens: int = 2048
    max_retries: int = 3

    # Paths
    artifacts_dir: Path = Path("artifacts")
    training_stats_path: Path = Path("artifacts/training_stats.json")
    feature_metadata_path: Path = Path("artifacts/feature_metadata.json")
    pipeline_path: Path = Path("artifacts/pricing_pipeline.joblib")

    # Server — Railway injects PORT at runtime
    port: int = 8000

    # Logging
    log_level: str = "INFO"


def get_settings() -> Settings:
    return Settings()
