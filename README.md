# Gemini Vision Pipeline

Zero-shot object detection powered by Google Gemini 3.5 Flash, with results
parsed and visualized through [supervision](https://github.com/roboflow/supervision).

## Features

- **Zero-shot detection** — detect any object class by name, no training required
- **Structured output** — schema-constrained JSON for dense scenes that would
  otherwise truncate mid-array
- **N-class detection** — detect many classes in a single API call
- **CLI** — run detections from the terminal, no code required
- **Python API** — one `ObjectDetector` call replaces the notebook's repeated blocks

## Installation

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Or with pip:

```bash
pip install -e .
```

## Configuration

Copy the example environment file and add your Gemini API key:

```bash
cp .env.example .env
```

Get a key from [Google AI Studio](https://aistudio.google.com/apikey), then set
it in `.env`:

```
GOOGLE_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-3.5-flash
```

## Usage — CLI

```bash
# Detect and save an annotated image
gemini-vision detect image.jpg --classes "car,truck,bus" --output annotated.jpg

# Force structured JSON output for dense scenes
gemini-vision detect image.jpg --classes "person" --structured

# Export detections as JSON, with no labels drawn on the boxes
gemini-vision detect image.jpg --classes "car" --no-labels --json-out detections.json
```

## Usage — Python

```python
from gemini_vision import ObjectDetector

detector = ObjectDetector()
result, annotated = detector.detect_and_annotate(
    "image.jpg",
    classes=["car", "truck", "bus"],
)

print(f"Detected {len(result.detections)} objects")
annotated.save("annotated.jpg")
```

## Examples

Three runnable scripts live in [`examples/`](examples/):

- `01_single_class.py` — single-class detection (air balloons)
- `02_multi_class.py` — multi-class detection (avocados)
- `03_structured_output.py` — structured output for a dense scene (people)

Download the example images first:

```bash
bash scripts/download_examples.sh
python examples/01_single_class.py
```

## Project structure

```
src/gemini_vision/   the package: config, client, prompts, schemas,
                     detector, annotator, CLI
examples/            standalone usage scripts
notebooks/           the original Colab notebook, kept for reference
scripts/             image download helper
tests/               unit tests (Gemini API mocked)
```

## Roadmap / Notes

- `supervision` is currently installed from the Git branch
  `add-gemini-3.5-vlm-support`. Replace it with the PyPI release once Gemini 3.5
  support is merged upstream.

## License

Apache-2.0. See [LICENSE](LICENSE).
