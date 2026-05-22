"""Tests for the ObjectDetector pipeline (Gemini API mocked)."""

from unittest.mock import MagicMock

from PIL import Image

from gemini_vision.detector import ObjectDetector


def test_detect_parses_response(monkeypatch, fake_gemini_client):
    monkeypatch.setattr(
        "gemini_vision.detector.get_client", lambda: fake_gemini_client
    )
    detector = ObjectDetector(model="gemini-3.5-flash")
    result = detector.detect(Image.new("RGB", (100, 80)), ["car"])
    assert len(result.detections) == 2
    assert result.detections[0].label == "car"
    assert result.detections[0].confidence == 0.9
    assert result.classes == ["car"]
    assert result.model == "gemini-3.5-flash"
    assert result.image_size == (100, 80)


def test_detect_handles_markdown_fenced_json(
    monkeypatch, fake_gemini_client, detection_json
):
    fenced = f"```json\n{detection_json}\n```"
    fake_gemini_client.models.generate_content.return_value = MagicMock(
        text=fenced
    )
    monkeypatch.setattr(
        "gemini_vision.detector.get_client", lambda: fake_gemini_client
    )
    detector = ObjectDetector(model="gemini-3.5-flash")
    result = detector.detect(Image.new("RGB", (10, 10)), ["car"])
    assert len(result.detections) == 2


def test_detect_and_annotate_calls_from_vlm(monkeypatch, fake_gemini_client):
    monkeypatch.setattr(
        "gemini_vision.detector.get_client", lambda: fake_gemini_client
    )
    captured = {}

    def fake_from_vlm(**kwargs):
        captured.update(kwargs)
        return MagicMock(name="sv.Detections")

    monkeypatch.setattr(
        "gemini_vision.detector.sv.Detections.from_vlm", fake_from_vlm
    )
    monkeypatch.setattr(
        "gemini_vision.detector.annotate",
        lambda image, detections, with_labels=True: Image.new("RGB", (10, 10)),
    )
    detector = ObjectDetector(model="gemini-3.5-flash")
    result, annotated = detector.detect_and_annotate(
        Image.new("RGB", (64, 48)), ["car"]
    )
    assert len(result.detections) == 2
    assert captured["classes"] == ["car"]
    assert captured["resolution_wh"] == (64, 48)
    assert isinstance(annotated, Image.Image)
