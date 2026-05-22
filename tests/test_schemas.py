"""Tests for the detection Pydantic models."""

import pytest
from pydantic import ValidationError

from gemini_vision.schemas import Detection, DetectionResult


def test_detection_accepts_valid_values():
    d = Detection(label="car", confidence=0.9, box_2d=[10, 20, 30, 40])
    assert d.label == "car"
    assert d.confidence == 0.9
    assert d.box_2d == [10, 20, 30, 40]


def test_detection_rejects_confidence_above_one():
    with pytest.raises(ValidationError):
        Detection(label="car", confidence=1.5, box_2d=[10, 20, 30, 40])


def test_detection_rejects_confidence_below_zero():
    with pytest.raises(ValidationError):
        Detection(label="car", confidence=-0.1, box_2d=[10, 20, 30, 40])


def test_detection_accepts_confidence_boundary_values():
    Detection(label="car", confidence=0.0, box_2d=[0, 0, 0, 0])
    Detection(label="car", confidence=1.0, box_2d=[0, 0, 0, 0])


def test_detection_rejects_wrong_box_length():
    with pytest.raises(ValidationError):
        Detection(label="car", confidence=0.5, box_2d=[10, 20, 30])
    with pytest.raises(ValidationError):
        Detection(label="car", confidence=0.5, box_2d=[10, 20, 30, 40, 50])


def test_detection_result_holds_metadata():
    d = Detection(label="car", confidence=0.5, box_2d=[1, 2, 3, 4])
    result = DetectionResult(
        detections=[d],
        classes=["car"],
        model="gemini-3.5-flash",
        image_size=(640, 480),
    )
    assert result.detections == [d]
    assert result.classes == ["car"]
    assert result.image_size == (640, 480)
    assert result.raw_response is None
