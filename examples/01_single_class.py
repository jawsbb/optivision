"""Single-class detection: air balloons in the sky.

Run `bash scripts/download_examples.sh` first to fetch the image.
"""

from gemini_vision import ObjectDetector

IMAGE = "assets/pexels-eyup-sayar-290427017-18373303.jpg"

detector = ObjectDetector()
result, annotated = detector.detect_and_annotate(
    IMAGE, ["air balloon"], with_labels=False
)

print(f"Detected {len(result.detections)} air balloon(s).")
annotated.save("balloons_annotated.jpg")
print("Saved balloons_annotated.jpg")
