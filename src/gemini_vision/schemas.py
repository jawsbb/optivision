"""Pydantic models for detection results."""

from pydantic import BaseModel, Field


class Detection(BaseModel):
    """A single detected object.

    Attributes:
        label: Class name of the detected object.
        confidence: Model certainty, between 0 and 1.
        box_2d: Bounding box [ymin, xmin, ymax, xmax] normalized to 0-1000.
    """

    label: str
    confidence: float = Field(ge=0, le=1)
    box_2d: list[int] = Field(min_length=4, max_length=4)


class DetectionResult(BaseModel):
    """The full result of one detection run.

    Attributes:
        detections: The detected objects.
        classes: The object classes requested for detection.
        model: The Gemini model used.
        image_size: Source image size as (width, height).
        raw_response: Raw text response from the model, if retained.
    """

    detections: list[Detection]
    classes: list[str]
    model: str
    image_size: tuple[int, int]
    raw_response: str | None = None
