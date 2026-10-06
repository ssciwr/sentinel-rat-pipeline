"""Tests for watch_inputs helper."""

import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from sentinel_rat_pipeline.scripts import watch_inputs
from sentinel_rat_pipeline.scripts.watch_inputs import (
    ImageCreatedHandler,
    _is_image,
    _wait_for_stable_file,
    list_images,
)


@pytest.fixture(autouse=True)
def fast_stability_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Speed up the stable-file polling and reset the dedup state per test."""
    monkeypatch.setattr(watch_inputs, "STABLE_INTERVAL", 0.01)
    monkeypatch.setattr(watch_inputs, "STABLE_TIMEOUT", 0.5)
    monkeypatch.setattr(watch_inputs, "_SEEN", {})


def _wait_until_called(mock: MagicMock, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not mock.called and time.monotonic() < deadline:
        time.sleep(0.01)


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
    image.write_bytes(b"jpg")
    mock_callback = MagicMock()
    handler = ImageCreatedHandler(mock_callback)

    event = type("Event", (), {"is_directory": False, "src_path": str(image), "dest_path": ""})()  # type: ignore[arg-type]
    handler.on_created(event)
    # callback is invoked from a daemon thread once the file is stable
    _wait_until_called(mock_callback)
    mock_callback.assert_called_once_with(str(image))


def test_wait_for_stable_file_skips_already_seen(tmp_path: Path) -> None:
    image = tmp_path / "photo.jpg"
    image.write_bytes(b"jpg")
    mock_callback = MagicMock()

    _wait_for_stable_file(str(image), mock_callback)
    _wait_for_stable_file(str(image), mock_callback)

    mock_callback.assert_called_once_with(str(image))


def test_wait_for_stable_file_gives_up_on_missing_file(tmp_path: Path) -> None:
    mock_callback = MagicMock()

    _wait_for_stable_file(str(tmp_path / "missing.jpg"), mock_callback)

    mock_callback.assert_not_called()
