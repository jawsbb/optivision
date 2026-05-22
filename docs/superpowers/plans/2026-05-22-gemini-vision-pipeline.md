# Gemini Vision Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn a Colab object-detection notebook into a packaged repo with a `gemini-vision` CLI and a reusable `ObjectDetector` Python API.

**Architecture:** A `src`-layout package, `gemini_vision`, splits the notebook's duplicated block into focused modules — config, client, prompts, schemas, detector, annotator, CLI. The detector calls Gemini once per image and returns a validated `DetectionResult`; `detect_and_annotate` adds an annotated image. Tests mock the Gemini API; no real calls.

**Tech Stack:** Python 3.11+, uv, google-genai, supervision (Git branch), pydantic v2, pydantic-settings, typer, pytest, ruff, pillow.

**Notes for the executor:**
- Work happens directly in the repo root (`optivision/`). It is a fresh standalone Git repo with no remote — no worktree needed.
- Every commit ends with the trailer `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>` (shown in each commit command).
- `uv` is required (https://docs.astral.sh/uv/). If missing: `curl -LsSf https://astral.sh/uv/install.sh | sh`.
- The reference spec is `docs/superpowers/specs/2026-05-22-gemini-vision-pipeline-design.md`.

---

## Task 1: Project bootstrap

**Files:**
- Create: `pyproject.toml`, `.python-version`, `.gitignore`, `.env.example`, `LICENSE`
- Create: `src/gemini_vision/__init__.py`, `tests/__init__.py`, `assets/.gitkeep`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "gemini-vision-pipeline"
version = "0.1.0"
description = "Zero-shot object detection pipeline powered by Google Gemini 3.5 Flash"
readme = "README.md"
requires-python = ">=3.11"
license = "Apache-2.0"
authors = [{ name = "Jules Koehler", email = "jules@levelups.fr" }]
dependencies = [
    "google-genai>=1.0",
    "supervision",
    "pydantic>=2.5",
    "pydantic-settings>=2.1",
    "typer>=0.12",
    "python-dotenv>=1.0",
    "pillow>=10.0",
]

[project.scripts]
gemini-vision = "gemini_vision.cli:app"

[dependency-groups]
dev = ["pytest>=8.0", "ruff>=0.6"]

[tool.uv.sources]
supervision = { git = "https://github.com/roboflow/supervision.git", branch = "add-gemini-3.5-vlm-support" }

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/gemini_vision"]

[tool.ruff]
line-length = 88
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "ANN", "D"]

[tool.ruff.lint.pydocstyle]
convention = "google"

[tool.ruff.lint.flake8-bugbear]
extend-immutable-calls = ["typer.Argument", "typer.Option"]

[tool.ruff.lint.per-file-ignores]
"tests/*" = ["ANN", "D"]
"examples/*" = ["D"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create `.python-version`**

```
3.11
```

- [ ] **Step 3: Create `.gitignore`**

```gitignore
# Python
__pycache__/
*.py[cod]
*.egg-info/
.pytest_cache/
.ruff_cache/

# Virtual environment
.venv/

# Environment secrets
.env

# Example images (downloaded, not committed)
assets/*.jpg
assets/*.jpeg
assets/*.png

# Example script outputs
*_annotated.jpg
```

- [ ] **Step 4: Create `.env.example`**

```
GOOGLE_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-3.5-flash
```

- [ ] **Step 5: Create `LICENSE`**

Download the verbatim Apache License 2.0 text:

```bash
curl -sSL https://www.apache.org/licenses/LICENSE-2.0.txt -o LICENSE
```

Expected: a `LICENSE` file of ~11 KB beginning with `Apache License`.

- [ ] **Step 6: Create the package and test skeleton**

`src/gemini_vision/__init__.py` — minimal for now, updated in Task 5:

```python
"""Zero-shot object detection pipeline powered by Google Gemini 3.5 Flash."""
```

`tests/__init__.py` — empty file.

`assets/.gitkeep` — empty file.

- [ ] **Step 7: Install dependencies**

Run: `uv sync`
Expected: resolves and installs all dependencies (including `supervision` from the Git branch), creates `.venv/` and `uv.lock`. This may take 1–3 minutes.

If `uv sync` fails resolving `supervision`, stop and report — the Git branch `add-gemini-3.5-vlm-support` may have moved or merged (see the spec's Risks section).

- [ ] **Step 8: Verify the package imports**

Run: `uv run python -c "import gemini_vision; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml uv.lock .python-version .gitignore .env.example LICENSE \
        instructions.md src/gemini_vision/__init__.py tests/__init__.py assets/.gitkeep
git commit -m "chore: bootstrap project structure and tooling" \
           -m "Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Schemas (TDD)

**Files:**
- Create: `tests/test_schemas.py`
- Create: `src/gemini_vision/schemas.py`

- [ ] **Step 1: Write the failing test**

`tests/test_schemas.py`:

```python
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


def test_detection_rejects_wrong_box_length():
    with pytest.raises(ValidationError):
        Detection(label="car", confidence=0.5, box_2d=[10, 20, 30])


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_schemas.py -v`
Expected: collection error — `ModuleNotFoundError: No module named 'gemini_vision.schemas'`.

- [ ] **Step 3: Write the implementation**

`src/gemini_vision/schemas.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_schemas.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add tests/test_schemas.py src/gemini_vision/schemas.py
git commit -m "feat: add Detection and DetectionResult schemas" \
           -m "Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Prompts (TDD)

**Files:**
- Create: `tests/test_prompts.py`
- Create: `src/gemini_vision/prompts.py`

- [ ] **Step 1: Write the failing test**

`tests/test_prompts.py`:

```python
"""Tests for prompt construction."""

from gemini_vision.prompts import build_detection_prompt


def test_build_detection_prompt_injects_class_list():
    prompt = build_detection_prompt(["car", "truck", "bus"])
    assert "Valid object classes: car, truck, bus" in prompt


def test_build_detection_prompt_uses_first_class_as_example():
    prompt = build_detection_prompt(["avocado", "pit"])
    assert '"label": "avocado"' in prompt


def test_build_detection_prompt_is_stripped_and_non_empty():
    prompt = build_detection_prompt(["car"])
    assert prompt == prompt.strip()
    assert len(prompt) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_prompts.py -v`
Expected: collection error — `ModuleNotFoundError: No module named 'gemini_vision.prompts'`.

- [ ] **Step 3: Write the implementation**

`src/gemini_vision/prompts.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_prompts.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add tests/test_prompts.py src/gemini_vision/prompts.py
git commit -m "feat: add detection prompt template and builder" \
           -m "Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Support modules — config, client, annotator

These three modules have no unit tests (config and client are thin wrappers; the annotator needs `sv.Detections` objects that the detector test exercises). They are verified by import.

**Files:**
- Create: `src/gemini_vision/config.py`
- Create: `src/gemini_vision/client.py`
- Create: `src/gemini_vision/annotator.py`

- [ ] **Step 1: Create `src/gemini_vision/config.py`**

```python
"""Application configuration loaded from environment variables."""

from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for the Gemini vision pipeline.

    Attributes:
        google_api_key: API key for the Gemini API.
        gemini_model: Default Gemini model identifier.
        default_temperature: Default sampling temperature.
        default_thinking_budget: Default thinking budget; 0 disables thinking.
    """

    google_api_key: SecretStr
    gemini_model: str = "gemini-3.5-flash"
    default_temperature: float = 0.0
    default_thinking_budget: int = 0

    model_config = SettingsConfigDict(env_file=".env", env_prefix="")


@lru_cache
def get_settings() -> Settings:
    """Return the cached application settings.

    Returns:
        The singleton ``Settings`` instance.
    """
    return Settings()
```

- [ ] **Step 2: Create `src/gemini_vision/client.py`**

```python
"""Gemini API client factory."""

from functools import lru_cache

from google import genai

from gemini_vision.config import get_settings


@lru_cache
def get_client() -> genai.Client:
    """Return a cached Gemini API client.

    The client is built from application settings and is never created from
    Colab ``userdata`` — that was the original notebook's main debt.

    Returns:
        A configured ``genai.Client`` instance.
    """
    settings = get_settings()
    return genai.Client(api_key=settings.google_api_key.get_secret_value())
```

- [ ] **Step 3: Create `src/gemini_vision/annotator.py`**

```python
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
```

- [ ] **Step 4: Verify the modules import**

Run:
```bash
uv run python -c "from gemini_vision.config import get_settings, Settings; \
from gemini_vision.client import get_client; \
from gemini_vision.annotator import annotate, COLOR; print('ok')"
```
Expected: prints `ok`.

- [ ] **Step 5: Commit**

```bash
git add src/gemini_vision/config.py src/gemini_vision/client.py \
        src/gemini_vision/annotator.py
git commit -m "feat: add config, client, and annotator modules" \
           -m "Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Detector (TDD)

**Files:**
- Create: `tests/conftest.py`
- Create: `tests/test_detector.py`
- Create: `src/gemini_vision/detector.py`
- Modify: `src/gemini_vision/__init__.py`

- [ ] **Step 1: Write `tests/conftest.py`**

```python
"""Shared pytest fixtures."""

import json
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _dummy_api_key(monkeypatch):
    """Provide a dummy API key so Settings never fails during tests."""
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")


@pytest.fixture
def detection_json() -> str:
    """A fixed JSON array of two car detections."""
    return json.dumps(
        [
            {"label": "car", "confidence": 0.9, "box_2d": [10, 20, 30, 40]},
            {"label": "car", "confidence": 0.8, "box_2d": [50, 60, 70, 80]},
        ]
    )


@pytest.fixture
def fake_gemini_client(detection_json):
    """A mock Gemini client returning a fixed JSON detection response."""
    client = MagicMock()
    client.models.generate_content.return_value = MagicMock(text=detection_json)
    return client
```

- [ ] **Step 2: Write the failing test**

`tests/test_detector.py`:

```python
"""Tests for the ObjectDetector pipeline (Gemini API mocked)."""

from unittest.mock import MagicMock

from PIL import Image

from gemini_vision.detector import ObjectDetector


def test_detect_parses_response(monkeypatch, fake_gemini_client):
    monkeypatch.setattr(
        "gemini_vision.detector.get_client", lambda: fake_gemini_client
    )
    detector = ObjectDetector(model="gemini-3.5-flash")
    result = detector.detect(Image.new("RGB", (100, 80)), ["car"])
    assert len(result.detections) == 2
    assert result.detections[0].label == "car"
    assert result.detections[0].confidence == 0.9
    assert result.classes == ["car"]
    assert result.model == "gemini-3.5-flash"
    assert result.image_size == (100, 80)


def test_detect_handles_markdown_fenced_json(
    monkeypatch, fake_gemini_client, detection_json
):
    fenced = f"```json\n{detection_json}\n```"
    fake_gemini_client.models.generate_content.return_value = MagicMock(
        text=fenced
    )
    monkeypatch.setattr(
        "gemini_vision.detector.get_client", lambda: fake_gemini_client
    )
    detector = ObjectDetector(model="gemini-3.5-flash")
    result = detector.detect(Image.new("RGB", (10, 10)), ["car"])
    assert len(result.detections) == 2


def test_detect_and_annotate_calls_from_vlm(monkeypatch, fake_gemini_client):
    monkeypatch.setattr(
        "gemini_vision.detector.get_client", lambda: fake_gemini_client
    )
    captured = {}

    def fake_from_vlm(**kwargs):
        captured.update(kwargs)
        return MagicMock(name="sv.Detections")

    monkeypatch.setattr(
        "gemini_vision.detector.sv.Detections.from_vlm", fake_from_vlm
    )
    monkeypatch.setattr(
        "gemini_vision.detector.annotate",
        lambda image, detections, with_labels=True: Image.new("RGB", (10, 10)),
    )
    detector = ObjectDetector(model="gemini-3.5-flash")
    result, annotated = detector.detect_and_annotate(
        Image.new("RGB", (64, 48)), ["car"]
    )
    assert len(result.detections) == 2
    assert captured["classes"] == ["car"]
    assert captured["resolution_wh"] == (64, 48)
    assert isinstance(annotated, Image.Image)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_detector.py -v`
Expected: collection error — `ModuleNotFoundError: No module named 'gemini_vision.detector'`.

- [ ] **Step 4: Write the implementation**

`src/gemini_vision/detector.py`:

```python
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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_detector.py -v`
Expected: 3 passed.

- [ ] **Step 6: Update `src/gemini_vision/__init__.py`**

Replace the file's contents with:

```python
"""Zero-shot object detection pipeline powered by Google Gemini 3.5 Flash."""

from gemini_vision.detector import ObjectDetector
from gemini_vision.schemas import Detection, DetectionResult

__all__ = ["Detection", "DetectionResult", "ObjectDetector"]
```

- [ ] **Step 7: Run the full test suite**

Run: `uv run pytest -v`
Expected: 11 passed (5 schemas + 3 prompts + 3 detector).

- [ ] **Step 8: Commit**

```bash
git add tests/conftest.py tests/test_detector.py \
        src/gemini_vision/detector.py src/gemini_vision/__init__.py
git commit -m "feat: add ObjectDetector pipeline and package exports" \
           -m "Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: CLI

**Files:**
- Create: `src/gemini_vision/cli.py`

- [ ] **Step 1: Create `src/gemini_vision/cli.py`**

```python
"""Command-line interface for the Gemini vision pipeline."""

from pathlib import Path

import typer

from gemini_vision.detector import ObjectDetector

app = typer.Typer(
    help="Zero-shot object detection with Google Gemini 3.5 Flash."
)


@app.command()
def detect(
    image: Path = typer.Argument(..., help="Path to the input image."),
    classes: str = typer.Option(
        ..., "--classes", help="Comma-separated class names."
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Path for the annotated image."
    ),
    json_out: Path | None = typer.Option(
        None, "--json-out", help="Path for the detections JSON."
    ),
    structured: bool = typer.Option(
        False, "--structured", help="Force schema-constrained JSON output."
    ),
    labels: bool = typer.Option(
        True, "--labels/--no-labels", help="Draw class labels on the boxes."
    ),
    model: str | None = typer.Option(
        None, "--model", help="Override the Gemini model."
    ),
) -> None:
    """Detect objects in an image, with optional annotated and JSON output."""
    class_list = [c.strip() for c in classes.split(",") if c.strip()]
    detector = ObjectDetector(model=model)

    if output is not None:
        result, annotated = detector.detect_and_annotate(
            image,
            class_list,
            with_labels=labels,
            structured_output=structured,
        )
        annotated.save(output)
        typer.echo(f"Annotated image saved to {output}")
    else:
        result = detector.detect(
            image, class_list, structured_output=structured
        )

    typer.echo(f"Detected {len(result.detections)} object(s).")

    if json_out is not None:
        json_out.write_text(result.model_dump_json(indent=2))
        typer.echo(f"Detections saved to {json_out}")
```

- [ ] **Step 2: Verify the CLI loads**

Run: `uv run gemini-vision --help`
Expected: usage text listing the `detect` command.

Run: `uv run gemini-vision detect --help`
Expected: usage text listing `--classes`, `--output`, `--json-out`, `--structured`, `--labels/--no-labels`, `--model`.

- [ ] **Step 3: Commit**

```bash
git add src/gemini_vision/cli.py
git commit -m "feat: add typer CLI with the detect command" \
           -m "Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Example scripts

Each script imports the public API and runs one detection — no duplicated logic. They make real API calls, so they are verified by compilation only.

**Files:**
- Create: `examples/01_single_class.py`
- Create: `examples/02_multi_class.py`
- Create: `examples/03_structured_output.py`

- [ ] **Step 1: Create `examples/01_single_class.py`**

```python
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
```

- [ ] **Step 2: Create `examples/02_multi_class.py`**

```python
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
```

- [ ] **Step 3: Create `examples/03_structured_output.py`**

```python
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
```

- [ ] **Step 4: Verify the scripts compile**

Run: `uv run python -m py_compile examples/01_single_class.py examples/02_multi_class.py examples/03_structured_output.py`
Expected: no output, exit code 0.

- [ ] **Step 5: Commit**

```bash
git add examples/01_single_class.py examples/02_multi_class.py \
        examples/03_structured_output.py
git commit -m "docs: add three example scripts for the public API" \
           -m "Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Notebook and image download script

**Files:**
- Move: `object_detection_with_gemini_3_5_flash.ipynb` → `notebooks/original_exploration.ipynb`
- Create: `scripts/download_examples.sh`

- [ ] **Step 1: Move the original notebook**

Run:
```bash
mkdir -p notebooks
mv object_detection_with_gemini_3_5_flash.ipynb \
   notebooks/original_exploration.ipynb
```
Expected: the notebook now lives at `notebooks/original_exploration.ipynb` and no `.ipynb` remains in the repo root.

- [ ] **Step 2: Create `scripts/download_examples.sh`**

```bash
#!/usr/bin/env bash
# Download the example images used by the notebook and the example scripts.
set -euo pipefail

DEST="$(cd "$(dirname "$0")/.." && pwd)/assets"
mkdir -p "$DEST"

BASE="https://storage.googleapis.com/com-roboflow-marketing/playground-examples"
IMAGES=(
  "pexels-vanessa-loring-5966631.jpg"
  "pexels-eyup-sayar-290427017-18373303.jpg"
  "pexels-mutecevvil-18013812.jpg"
  "pexels-shvets-production-7195054.jpg"
  "pexels-spencer-4353558.jpg"
  "top-shot-of-a-worker-scanning-boxes-using-a-bar-co-2026-01-11-09-59-09-utc.jpg"
  "warehouse-workers-inspecting-boxes-along-conveyor-2026-01-11-09-55-23-utc.jpg"
  "top-view-of-people-relaxing-in-the-pool-on-yellow-2026-03-24-21-54-59-utc.jpg"
  "aerial-drone-photograph-of-traffic-jam-in-metropol-2026-03-18-17-36-02-utc.jpg"
)

for img in "${IMAGES[@]}"; do
  echo "Downloading $img"
  wget -q "$BASE/$img" -O "$DEST/$img"
done

echo "Done. ${#IMAGES[@]} images saved to $DEST"
```

- [ ] **Step 3: Make the script executable**

Run: `chmod +x scripts/download_examples.sh`

- [ ] **Step 4: Verify the script is valid bash**

Run: `bash -n scripts/download_examples.sh`
Expected: no output, exit code 0.

- [ ] **Step 5: Commit**

```bash
git add notebooks/original_exploration.ipynb scripts/download_examples.sh
git commit -m "docs: add original notebook and image download script" \
           -m "Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: README

**Files:**
- Create: `README.md`

- [ ] **Step 1: Create `README.md`**

````markdown
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
````

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add project README" \
           -m "Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 10: Lint and final verification

**Files:** none created; this task fixes any lint or formatting issues.

- [ ] **Step 1: Auto-fix lint issues**

Run: `uv run ruff check --fix .`
Expected: reports issues fixed, or "All checks passed!".

- [ ] **Step 2: Format the code**

Run: `uv run ruff format .`
Expected: reports files reformatted or left unchanged.

- [ ] **Step 3: Confirm lint is clean**

Run: `uv run ruff check .`
Expected: `All checks passed!`. If any error remains, fix it manually and re-run.

- [ ] **Step 4: Confirm formatting is stable**

Run: `uv run ruff format --check .`
Expected: `N files already formatted`.

- [ ] **Step 5: Run the full test suite**

Run: `uv run pytest -v`
Expected: 11 passed.

- [ ] **Step 6: Commit any changes**

If Steps 1–2 changed files:

```bash
git add -A
git commit -m "style: apply ruff lint fixes and formatting" \
           -m "Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

If nothing changed, skip the commit.

---

## Self-Review

**Spec coverage:** Every spec section maps to a task — config/schemas/prompts/client/detector/annotator/cli (Tasks 2–6), examples (Task 7), notebook + download script (Task 8), README (Task 9), pyproject/tooling (Task 1), lint pass (Task 10). The five deviations and both risks from the spec are reflected (ruff-only, `[dependency-groups]`, `[tool.uv.sources]`, explicit `pillow`, no response-time field; README notes the `supervision` branch).

**Placeholder scan:** No `TBD`/`TODO`/"handle edge cases" steps. The `LICENSE` step uses a concrete `curl` command for a fixed external document.

**Type consistency:** `Detection`/`DetectionResult` fields, `ObjectDetector.__init__`/`detect`/`detect_and_annotate` signatures, `annotate(image, detections, with_labels, max_size)`, `get_client`, `get_settings`, and `build_detection_prompt` are used identically across the detector, CLI, tests, and examples.
