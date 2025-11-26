"""Application configuration and settings loading."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Base settings loaded from environment or .env file."""

    data_dir: Path = Path("data")
    gemini_api_key: str | None = None
    gemini_model: str | None = "models/gemini-2.5-flash"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings instance."""
    settings = Settings()
    settings.data_dir = Path(settings.data_dir)
    return settings


settings = get_settings()
