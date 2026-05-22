"""Structured output for a dense scene: people in a pool.

Run `bash scripts/download_examples.sh` first to fetch the image.
"""

from gemini_vision import ObjectDetector

IMAGE = (
    "assets/top-view-of-people-relaxing-in-the-pool-"
    "on-yellow-2026-03-24-21-54-59-utc.jpg"
)

detector = ObjectDetector()
result, annotated = detector.detect_and_annotate(
    IMAGE, ["person"], with_labels=False, structured_output=True
)

print(f"Detected {len(result.detections)} person(s).")
annotated.save("people_annotated.jpg")
print("Saved people_annotated.jpg")
