"""Zero-shot object detection pipeline powered by Google Gemini 3.5 Flash."""

from gemini_vision.detector import ObjectDetector
from gemini_vision.schemas import Detection, DetectionResult

__all__ = ["Detection", "DetectionResult", "ObjectDetector"]
