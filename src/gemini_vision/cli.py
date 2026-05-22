"""Command-line interface for the Gemini vision pipeline."""

from pathlib import Path

import typer

from gemini_vision.detector import ObjectDetector

app = typer.Typer(help="Zero-shot object detection with Google Gemini 3.5 Flash.")


@app.callback()
def main() -> None:
    """Run the Gemini vision pipeline from the command line.

    Declaring a callback keeps ``detect`` as an explicit subcommand; without
    it, Typer collapses a single-command app and drops the command name.
    """


@app.command()
def detect(
    image: Path = typer.Argument(..., help="Path to the input image."),
    classes: str = typer.Option(..., "--classes", help="Comma-separated class names."),
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
        result = detector.detect(image, class_list, structured_output=structured)

    typer.echo(f"Detected {len(result.detections)} object(s).")

    if json_out is not None:
        json_out.write_text(result.model_dump_json(indent=2))
        typer.echo(f"Detections saved to {json_out}")
