"""Tests for the command-line interface."""

from typer.testing import CliRunner

from gemini_vision.cli import app

runner = CliRunner()


def test_detect_is_a_named_subcommand():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "detect" in result.output


def test_detect_help_lists_options():
    result = runner.invoke(app, ["detect", "--help"])
    assert result.exit_code == 0
    assert "--classes" in result.output
    assert "--structured" in result.output


def test_bare_image_argument_is_rejected():
    # The image path must follow the `detect` subcommand, not stand alone.
    result = runner.invoke(app, ["some-image.jpg"])
    assert result.exit_code != 0
