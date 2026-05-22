"""Shared pytest fixtures."""

import json
from unittest.mock import MagicMock

import pytest

from gemini_vision.client import get_client
from gemini_vision.config import get_settings


@pytest.fixture(autouse=True)
def _isolated_settings(monkeypatch):
    """Give each test a dummy API key and a clean settings/client cache."""
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    get_settings.cache_clear()
    get_client.cache_clear()
    yield
    get_settings.cache_clear()
    get_client.cache_clear()


@pytest.fixture
def detection_json() -> str:
    """A fixed JSON array of two car detections."""
    return json.dumps(
        [
            {"label": "car", "confidence": 0.9, "box_2d": [10, 20, 30, 40]},
            {"label": "car", "confidence": 0.8, "box_2d": [50, 60, 70, 80]},
        ]
    )


@pytest.fixture
def fake_gemini_client(detection_json):
    """A mock Gemini client returning a fixed JSON detection response."""
    client = MagicMock()
    client.models.generate_content.return_value = MagicMock(text=detection_json)
    return client
