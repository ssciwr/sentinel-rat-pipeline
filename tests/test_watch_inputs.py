"""Tests for watch_inputs helper."""

import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from sentinel_rat_pipeline.scripts.watch_inputs import (
    ImageCreatedHandler,
    _is_image,
    list_images,
)


def test_list_images_empty(tmp_path: Path) -> None:
    assert list_images(str(tmp_path)) == []


def test_list_images_with_image(tmp_path: Path) -> None:
    (tmp_path / "photo.jpg").write_text("jpg")
    (tmp_path / "note.txt").write_text("txt")
    result = list_images(str(tmp_path))
    assert result == [str(tmp_path / "photo.jpg")]


def test_is_image(tmp_path: Path) -> None:
    assert _is_image(str(tmp_path / "a.jpg"))
    assert _is_image(str(tmp_path / "b.jpeg"))
    assert _is_image(str(tmp_path / "c.JPG"))
    assert not _is_image(str(tmp_path / "d.txt"))
    assert not _is_image(str(tmp_path / ""))  # empty suffix (Windows copy event)


def test_image_created_handler_filters_non_images(tmp_path: Path) -> None:
    received: list[str] = []
    handler = ImageCreatedHandler(received.append)

    # Non-image src_path is rejected immediately, no blocking
    event = type("Event", (), {"is_directory": False, "src_path": str(tmp_path / "doc.txt"), "dest_path": ""})()  # type: ignore[arg-type]
    handler.on_created(event)
    assert received == []


def test_image_created_handler_schedules_thread(tmp_path: Path) -> None:
    image = tmp_path / "photo.jpg"
    mock_callback = MagicMock()
    handler = ImageCreatedHandler(mock_callback)

    event = type("Event", (), {"is_directory": False, "src_path": str(image), "dest_path": ""})()  # type: ignore[arg-type]
    handler.on_created(event)
    # callback is invoked from a daemon thread; give it time to finish
    time.sleep(2.0)
    mock_callback.assert_called_once_with(str(image))
