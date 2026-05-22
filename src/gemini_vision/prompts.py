"""Prompt templates for Gemini object detection."""

DETECTION_PROMPT_TEMPLATE = """
Carefully examine this image and detect ALL visible objects, including small, distant, or partially visible ones.
IMPORTANT: Focus on finding as many objects as possible, even if you are only moderately confident.
Make sure each bounding box is as tight as possible.
Valid object classes: {class_list}
For each detected object, provide:
- "label": the exact class name from the list above
- "confidence": your certainty (between 0.0 and 1.0)
- "box_2d": the bounding box [ymin, xmin, ymax, xmax] normalized to 0-1000
Detect everything that matches the valid classes. Do not be conservative; include objects even with moderate confidence.
Return a JSON array, for example:
[
    {{"label": "{class_example}", "confidence": 0.95, "box_2d": [100, 200, 300, 400]}}
]
"""


def build_detection_prompt(classes: list[str]) -> str:
    """Build a detection prompt for the given classes.

    Args:
        classes: Object classes to detect. Must be non-empty.

    Returns:
        The formatted prompt string.
    """
    return DETECTION_PROMPT_TEMPLATE.format(
        class_list=", ".join(classes),
        class_example=classes[0],
    ).strip()
