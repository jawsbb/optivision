"""Gemini API client factory."""

from functools import lru_cache

from google import genai

from gemini_vision.config import get_settings


@lru_cache
def get_client() -> genai.Client:
    """Return a cached Gemini API client.

    The client is built from application settings and is never created from
    Colab ``userdata`` — that was the original notebook's main debt.

    Returns:
        A configured ``genai.Client`` instance.
    """
    settings = get_settings()
    return genai.Client(api_key=settings.google_api_key.get_secret_value())
