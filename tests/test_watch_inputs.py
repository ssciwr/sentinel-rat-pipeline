"""Tests for watch_inputs helper."""

import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from sentinel_rat_pipeline.config import Settings
from sentinel_rat_pipeline.scripts import watch_inputs
from sentinel_rat_pipeline.scripts.watch_inputs import (
    ImageBatcher,
    ImageCreatedHandler,
    _claim,
    _is_image,
    _split_stable,
    _wait_for_stable_file,
    list_images,
    queue_backlog,
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


def test_list_images_includes_sub_folders(tmp_path: Path) -> None:
    (tmp_path / "top.jpg").write_text("jpg")
    (tmp_path / "cam01" / "2026-10").mkdir(parents=True)
    (tmp_path / "cam01" / "a.jpg").write_text("jpg")
    (tmp_path / "cam01" / "2026-10" / "b.png").write_text("png")
    (tmp_path / "cam01" / "dir.jpg").mkdir()  # a folder, not an image
    result = list_images(str(tmp_path))
    assert sorted(result) == [
        str(tmp_path / "cam01" / "2026-10" / "b.png"),
        str(tmp_path / "cam01" / "a.jpg"),
        str(tmp_path / "top.jpg"),
    ]


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


def test_batcher_collects_images_within_window() -> None:
    mock_callback = MagicMock()
    batcher = ImageBatcher(mock_callback, window=0.2)

    batcher.add("a.jpg")
    batcher.add("b.jpg")
    batcher.add("a.jpg")  # duplicates are ignored
    mock_callback.assert_not_called()  # window still open

    _wait_until_called(mock_callback)
    mock_callback.assert_called_once_with(["a.jpg", "b.jpg"])


def test_batcher_opens_new_window_after_flush() -> None:
    mock_callback = MagicMock()
    batcher = ImageBatcher(mock_callback, window=0.05)

    batcher.add("a.jpg")
    _wait_until_called(mock_callback)
    mock_callback.reset_mock()

    batcher.add("b.jpg")
    _wait_until_called(mock_callback)
    mock_callback.assert_called_once_with(["b.jpg"])


def test_batcher_discard_removes_pending_image() -> None:
    mock_callback = MagicMock()
    batcher = ImageBatcher(mock_callback, window=60)

    batcher.add("a.jpg")
    batcher.add("b.jpg")
    batcher.discard("a.jpg")
    batcher.discard("missing.jpg")
    batcher.flush()

    mock_callback.assert_called_once_with(["b.jpg"])


def test_batcher_flush_empty_does_not_call() -> None:
    mock_callback = MagicMock()
    ImageBatcher(mock_callback, window=60).flush()
    mock_callback.assert_not_called()


def test_batcher_survives_callback_error() -> None:
    mock_callback = MagicMock(side_effect=RuntimeError("boom"))
    batcher = ImageBatcher(mock_callback, window=60)

    batcher.add("a.jpg")
    batcher.flush()
    batcher.add("b.jpg")
    batcher.flush()

    assert mock_callback.call_count == 2


def test_claim_only_once(tmp_path: Path) -> None:
    image = tmp_path / "photo.jpg"
    image.write_bytes(b"jpg")

    assert _claim(str(image))
    assert not _claim(str(image))
    assert not _claim(str(tmp_path / "missing.jpg"))


def test_split_stable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stable = tmp_path / "stable.jpg"
    changing = tmp_path / "changing.jpg"
    removed = tmp_path / "removed.jpg"
    for image in (stable, changing, removed):
        image.write_bytes(b"jpg")

    def copy_in_progress(_: float) -> None:
        changing.write_bytes(b"jpg, more data")
        removed.unlink()

    monkeypatch.setattr(watch_inputs.time, "sleep", copy_in_progress)

    assert _split_stable([str(stable), str(changing), str(removed)]) == (
        [str(stable)],
        [str(changing)],
    )


def test_queue_backlog_processes_stable_images_at_once(tmp_path: Path) -> None:
    (tmp_path / "a.jpg").write_bytes(b"jpg")
    (tmp_path / "b.jpg").write_bytes(b"jpg")
    mock_callback = MagicMock()

    queue_backlog(ImageBatcher(mock_callback, window=60), watch_folder=str(tmp_path))

    # flushed right away, without waiting for the batch window
    mock_callback.assert_called_once()
    assert sorted(mock_callback.call_args.args[0]) == [
        str(tmp_path / "a.jpg"),
        str(tmp_path / "b.jpg"),
    ]


def test_queue_backlog_applies_filter(tmp_path: Path) -> None:
    (tmp_path / "stored.jpg").write_bytes(b"jpg")
    (tmp_path / "new.jpg").write_bytes(b"jpg")
    mock_callback = MagicMock()

    queue_backlog(
        ImageBatcher(mock_callback, window=60),
        backlog_filter=lambda paths: [p for p in paths if "new" in p],
        watch_folder=str(tmp_path),
    )

    mock_callback.assert_called_once_with([str(tmp_path / "new.jpg")])


def test_queue_backlog_skips_everything_if_filter_fails(tmp_path: Path) -> None:
    (tmp_path / "a.jpg").write_bytes(b"jpg")
    mock_callback = MagicMock()

    def failing_filter(_: list[str]) -> list[str]:
        raise ConnectionError("database down")

    queue_backlog(
        ImageBatcher(mock_callback, window=60),
        backlog_filter=failing_filter,
        watch_folder=str(tmp_path),
    )

    mock_callback.assert_not_called()


def test_queue_backlog_skips_images_claimed_by_watcher(tmp_path: Path) -> None:
    (tmp_path / "a.jpg").write_bytes(b"jpg")
    (tmp_path / "b.jpg").write_bytes(b"jpg")
    assert _claim(str(tmp_path / "a.jpg"))  # the watcher got there first
    mock_callback = MagicMock()

    queue_backlog(ImageBatcher(mock_callback, window=60), watch_folder=str(tmp_path))

    mock_callback.assert_called_once_with([str(tmp_path / "b.jpg")])


def test_watcher_skips_images_claimed_by_backlog(tmp_path: Path) -> None:
    image = tmp_path / "a.jpg"
    image.write_bytes(b"jpg")
    mock_callback = MagicMock()

    queue_backlog(ImageBatcher(mock_callback, window=60), watch_folder=str(tmp_path))
    _wait_for_stable_file(str(image), mock_callback)

    mock_callback.assert_called_once_with([str(image)])


def test_queue_backlog_waits_for_images_still_being_copied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = tmp_path / "a.jpg"
    image.write_bytes(b"jpg")
    monkeypatch.setattr(
        watch_inputs, "_split_stable", lambda paths: ([], list(paths))
    )
    mock_callback = MagicMock()
    batcher = ImageBatcher(mock_callback, window=0.05)

    queue_backlog(batcher, watch_folder=str(tmp_path))
    mock_callback.assert_not_called()  # nothing ready yet

    # picked up once the per-file check sees it stable
    _wait_until_called(mock_callback)
    mock_callback.assert_called_once_with([str(image)])


def test_start_watch_picks_up_images_in_new_sub_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        watch_inputs,
        "settings",
        Settings(watch_folder=str(tmp_path), batch_window_seconds=0.05),
    )
    received: list[list[str]] = []
    watcher = threading.Thread(
        target=watch_inputs.start_watch, args=(received.append,), daemon=True
    )
    watcher.start()
    time.sleep(0.3)  # let the observer start

    # a camera folder moved in at once, as well as a file in an existing one
    staged = tmp_path.parent / f"{tmp_path.name}-staged" / "cam02"
    staged.mkdir(parents=True)
    (staged / "a.jpg").write_bytes(b"jpg")
    staged.rename(tmp_path / "cam02")
    (tmp_path / "cam02" / "b.jpg").write_bytes(b"jpg")

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and sum(map(len, received)) < 2:
        time.sleep(0.05)
    assert sorted(p for batch in received for p in batch) == [
        str((tmp_path / "cam02" / "a.jpg").resolve()),
        str((tmp_path / "cam02" / "b.jpg").resolve()),
    ]
