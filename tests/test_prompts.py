"""Tests for prompt construction."""

from gemini_vision.prompts import build_detection_prompt


def test_build_detection_prompt_injects_class_list():
    prompt = build_detection_prompt(["car", "truck", "bus"])
    assert "Valid object classes: car, truck, bus" in prompt


def test_build_detection_prompt_uses_first_class_as_example():
    prompt = build_detection_prompt(["avocado", "pit"])
    assert '"label": "avocado"' in prompt


def test_build_detection_prompt_is_stripped_and_non_empty():
    prompt = build_detection_prompt(["car"])
    assert prompt == prompt.strip()
    assert len(prompt) > 0
