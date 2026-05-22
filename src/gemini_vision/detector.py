"""Zero-shot object detection with the Gemini API."""

import json
import re
from pathlib import Path

import supervision as sv
from google.genai import types
from PIL import Image

from gemini_vision.annotator import annotate
from gemini_vision.client import get_client
from gemini_vision.config import get_settings
from gemini_vision.prompts import build_detection_prompt
from gemini_vision.schemas import Detection, DetectionResult

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$")


def _load_image(image: Image.Image | str | Path) -> Image.Image:
    """Return a PIL image, opening it from disk if a path was given.

    A PIL image passed in is returned by reference, not copied.

    Args:
        image: A PIL image, or a path to an image file.

    Returns:
        The loaded PIL image.
    """
    if isinstance(image, (str, Path)):
        return Image.open(image)
    return image


def _parse_detections(text: str) -> list[Detection]:
    """Parse a model response into detections, tolerating Markdown fences.

    Args:
        text: Raw text returned by the model.

    Returns:
        The parsed detections.
    """
    cleaned = _FENCE_RE.sub("", text.strip())
    return [Detection(**item) for item in json.loads(cleaned)]


class ObjectDetector:
    """Runs zero-shot object detection through the Gemini API."""

    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.0,
        thinking_budget: int = 0,
    ) -> None:
        """Initialize the detector.

        Args:
            model: Gemini model identifier. Defaults to the configured model.
            temperature: Sampling temperature.
            thinking_budget: Thinking budget; 0 disables thinking.
        """
        self.model = model or get_settings().gemini_model
        self.temperature = temperature
        self.thinking_budget = thinking_budget

    def _build_config(
        self, structured_output: bool
    ) -> types.GenerateContentConfig:
        """Build the Gemini generation config for a detection call.

        Args:
            structured_output: Whether to force schema-constrained JSON.

        Returns:
            The generation config.
        """
        config_kwargs: dict[str, object] = {
            "temperature": self.temperature,
            "thinking_config": types.ThinkingConfig(
                thinking_budget=self.thinking_budget
            ),
        }
        if structured_output:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_schema"] = list[Detection]
        return types.GenerateContentConfig(**config_kwargs)

    def detect(
        self,
        image: Image.Image | str | Path,
        classes: list[str],
        structured_output: bool = False,
    ) -> DetectionResult:
        """Detect objects of the given classes in an image.

        Args:
            image: A PIL image, or a path to an image file.
            classes: Object classes to detect.
            structured_output: Force schema-constrained JSON output.

        Returns:
            The detection result.
        """
        pil_image = _load_image(image)
        prompt = build_detection_prompt(classes)
        response = get_client().models.generate_content(
            model=self.model,
            contents=[pil_image, prompt],
            config=self._build_config(structured_output),
        )
        if response.text is None:
            raise ValueError(
                "Gemini returned no text; the response may have been blocked "
                "or truncated."
            )
        return DetectionResult(
            detections=_parse_detections(response.text),
            classes=classes,
            model=self.model,
            image_size=pil_image.size,
            raw_response=response.text,
        )

    def detect_and_annotate(
        self,
        image: Image.Image | str | Path,
        classes: list[str],
        with_labels: bool = True,
        structured_output: bool = False,
    ) -> tuple[DetectionResult, Image.Image]:
        """Detect objects and return the result with an annotated image.

        Args:
            image: A PIL image, or a path to an image file.
            classes: Object classes to detect.
            with_labels: Draw class labels on the boxes.
            structured_output: Force schema-constrained JSON output.

        Returns:
            A tuple of the detection result and the annotated image.
        """
        pil_image = _load_image(image)
        result = self.detect(
            pil_image, classes, structured_output=structured_output
        )
        sv_detections = sv.Detections.from_vlm(
            vlm=sv.VLM.GOOGLE_GEMINI_3_5,
            result=result.raw_response,
            resolution_wh=pil_image.size,
            classes=classes,
        )
        annotated = annotate(pil_image, sv_detections, with_labels=with_labels)
        return result, annotated
