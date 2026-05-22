"""Application configuration loaded from environment variables."""

from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for the Gemini vision pipeline.

    Attributes:
        google_api_key: API key for the Gemini API.
        gemini_model: Default Gemini model identifier.
        default_temperature: Default sampling temperature.
        default_thinking_budget: Default thinking budget; 0 disables thinking.
    """

    google_api_key: SecretStr
    gemini_model: str = "gemini-3.5-flash"
    default_temperature: float = 0.0
    default_thinking_budget: int = 0

    model_config = SettingsConfigDict(env_file=".env", env_prefix="")


@lru_cache
def get_settings() -> Settings:
    """Return the cached application settings.

    Returns:
        The singleton ``Settings`` instance.
    """
    return Settings()
