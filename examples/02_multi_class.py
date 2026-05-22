"""Multi-class detection: avocados with and without their pit.

Run `bash scripts/download_examples.sh` first to fetch the image.
"""

from gemini_vision import ObjectDetector

IMAGE = "assets/pexels-vanessa-loring-5966631.jpg"
CLASSES = ["avocado with the pit", "avocado without the pit", "pit"]

detector = ObjectDetector()
result, annotated = detector.detect_and_annotate(IMAGE, CLASSES)

print(f"Detected {len(result.detections)} object(s) across {len(CLASSES)} classes.")
annotated.save("avocados_annotated.jpg")
print("Saved avocados_annotated.jpg")
