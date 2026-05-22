"""Detection visualization built on the supervision library."""

import supervision as sv
from PIL import Image

COLOR = sv.ColorPalette.from_hex(
    [
        "#ffff00", "#ff9b00", "#ff66ff", "#3399ff", "#ff66b2", "#ff8080",
        "#b266ff", "#9999ff", "#66ffff", "#33ff99", "#66ff66", "#99ff00",
    ]
)


def annotate(
    image: Image.Image,
    detections: sv.Detections,
    with_labels: bool = True,
    max_size: int | None = None,
) -> Image.Image:
    """Draw detection boxes, and optionally labels, on an image.

    Args:
        image: The source image.
        detections: The detections to draw.
        with_labels: Whether to draw class labels on the boxes.
        max_size: If set, the result is downscaled so its longest side is at
            most this many pixels. If ``None``, the image is not resized.

    Returns:
        A new annotated image; the source image is left unchanged.
    """
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=image.size)
    thickness = sv.calculate_optimal_line_thickness(resolution_wh=image.size)

    annotated = image.copy()
    annotated = sv.BoxAnnotator(color=COLOR, thickness=thickness).annotate(
        annotated, detections
    )
    if with_labels:
        annotated = sv.LabelAnnotator(
            color=COLOR,
            text_color=sv.Color.BLACK,
            text_scale=text_scale,
            text_thickness=thickness,
            smart_position=True,
        ).annotate(annotated, detections)

    if max_size is not None:
        annotated.thumbnail((max_size, max_size))
    return annotated
