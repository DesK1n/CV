"""Unit tests for the command-line interface (CLIApp) in cli.py.

Tests command dispatcher, argument parsing, dataset preparation checks,
synthetic mock video creation, and video tracking pipeline execution.
"""
from pathlib import Path
from typing import Sequence

import pytest

from cli import CLIApp, main
from data_manager import create_mock_video


def test_cli_parser_construction() -> None:
    """Verify CLI argument parser defines expected subcommands and options.

    Returns:
        None
    """
    app = CLIApp()
    subparsers_action = next(
        action
        for action in app.parser._actions
        if action.__class__.__name__ == "_SubParsersAction"
    )
    assert "prepare" in subparsers_action.choices
    assert "train" in subparsers_action.choices
    assert "track" in subparsers_action.choices
    assert "eval" in subparsers_action.choices


def test_cli_prepare_verify_only() -> None:
    """Verify 'prepare --verify-only' executes successfully without error.

    Returns:
        None
    """
    ret = main(["prepare", "--verify-only"])
    assert ret == 0


def test_cli_prepare_create_mock_video(tmp_path: Path) -> None:
    """Verify 'prepare --create-mock-video <path>' generates valid video file.

    Args:
        tmp_path: Pytest temporary directory fixture.

    Returns:
        None
    """
    out_video = tmp_path / "cli_mock.mp4"
    ret = main(["prepare", "--create-mock-video", str(out_video), "--frames", "5"])
    assert ret == 0
    assert out_video.exists()
    assert out_video.stat().st_size > 0


def test_cli_track_on_mock_video(tmp_path: Path) -> None:
    """Verify 'track' subcommand executes on mock video and outputs telemetry.

    Args:
        tmp_path: Pytest temporary directory fixture.

    Returns:
        None
    """
    input_video = tmp_path / "input.mp4"
    create_mock_video(input_video, num_frames=5, width=320, height=240, fps=10)
    out_json = tmp_path / "telemetry.json"

    ret = main(
        [
            "track",
            "--source",
            str(input_video),
            "--tracker",
            "enhanced_sort",
            "--max-frames",
            "3",
            "--no-trails",
            "--out-json",
            str(out_json),
        ]
    )
    assert ret == 0
    assert out_json.exists()
