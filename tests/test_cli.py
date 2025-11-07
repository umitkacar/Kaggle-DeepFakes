"""Tests for CLI module."""

import pytest
from typer.testing import CliRunner
from deepfake_detector.cli import app

runner = CliRunner()


def test_cli_version():
    """Test CLI version command."""
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "DeepFake Detector" in result.stdout


def test_cli_help():
    """Test CLI help command."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "deepfake-detector" in result.stdout.lower()


def test_config_show():
    """Test config show command."""
    result = runner.invoke(app, ["config", "--show"])
    assert result.exit_code == 0
