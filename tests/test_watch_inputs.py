"""Tests for watch_inputs helper."""

import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, MagicMock

import pytest

from sentinel_rat_pipeline.config import Settings
from sentinel_rat_pipeline.scripts import watch_inputs
from sentinel_rat_pipeline.scripts.watch_inputs import (
    ImageBatcher,
    ImageCreatedHandler,
    ImageDeletedHandler,
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


def _event(src_path: str, is_directory: bool = False):
    return type(
        "Event",
        (),
        {"is_directory": is_directory, "src_path": src_path, "dest_path": ""},
    )()


class FakeObserver:
    """Stand-in for watchdog's Observer that records what start_watch does."""

    def __init__(self) -> None:
        self.handlers: list[tuple[object, str, bool]] = []
        self.started = self.stopped = self.joined = False

    def schedule(self, handler, path: str, recursive: bool = False) -> None:
        self.handlers.append((handler, path, recursive))

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def join(self) -> None:
        self.joined = True


@pytest.fixture
def fake_observer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeObserver:
    monkeypatch.setattr(
        watch_inputs, "settings", Settings(watch_folder=str(tmp_path))
    )
    observer = FakeObserver()
    monkeypatch.setattr(watch_inputs, "Observer", lambda: observer)
    return observer


def _stop_watch_on_sleep(
    monkeypatch: pytest.MonkeyPatch, before_stop=lambda: None
) -> None:
    """Make the watch loop's first sleep run ``before_stop`` and then stop the
    watcher as Ctrl+C would."""

    def sleep(_: float) -> None:
        before_stop()
        raise KeyboardInterrupt

    monkeypatch.setattr(
        watch_inputs, "time", SimpleNamespace(sleep=sleep, monotonic=time.monotonic)
    )


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


def test_image_created_handler_skips_already_claimed_image(tmp_path: Path) -> None:
    image = tmp_path / "photo.jpg"
    image.write_bytes(b"jpg")
    assert _claim(str(image))
    mock_callback = MagicMock()

    ImageCreatedHandler(mock_callback).on_created(_event(str(image)))

    time.sleep(0.1)  # a scheduled thread would have called back by now
    mock_callback.assert_not_called()


def test_image_deleted_handler_ignores_non_images(tmp_path: Path) -> None:
    mock_callback = MagicMock()
    handler = ImageDeletedHandler(mock_callback)

    handler.on_deleted(_event(""))
    handler.on_deleted(_event(str(tmp_path / "cam01"), is_directory=True))
    handler.on_deleted(_event(str(tmp_path / "doc.txt")))

    mock_callback.assert_not_called()


def test_image_deleted_handler_calls_back_and_forgets_image(tmp_path: Path) -> None:
    image = tmp_path / "photo.jpg"
    image.write_bytes(b"jpg")
    assert _claim(str(image))
    image.unlink()
    mock_callback = MagicMock()

    ImageDeletedHandler(mock_callback).on_deleted(_event(str(image)))

    mock_callback.assert_called_once_with(str(image))
    # if the image is put back, it is processed again
    assert str(image) not in watch_inputs._SEEN


def test_image_deleted_handler_survives_callback_error(tmp_path: Path) -> None:
    image = tmp_path / "photo.jpg"
    image.write_bytes(b"jpg")
    assert _claim(str(image))
    image.unlink()
    mock_callback = MagicMock(side_effect=RuntimeError("database down"))

    ImageDeletedHandler(mock_callback).on_deleted(_event(str(image)))

    mock_callback.assert_called_once_with(str(image))
    # the deletion was not handled, so the image stays claimed
    assert str(image) in watch_inputs._SEEN


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


def test_queue_backlog_with_empty_folder(tmp_path: Path) -> None:
    mock_callback = MagicMock()
    mock_filter = MagicMock()

    queue_backlog(
        ImageBatcher(mock_callback, window=60),
        backlog_filter=mock_filter,
        watch_folder=str(tmp_path),
    )

    mock_filter.assert_not_called()
    mock_callback.assert_not_called()


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


def test_start_watch_reports_deleted_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        watch_inputs,
        "settings",
        Settings(watch_folder=str(tmp_path), batch_window_seconds=0.05),
    )
    received: list[list[str]] = []
    deleted = MagicMock()
    watcher = threading.Thread(
        target=watch_inputs.start_watch,
        args=(received.append,),
        kwargs={"on_deleted": deleted},
        daemon=True,
    )
    watcher.start()
    time.sleep(0.3)  # let the observer start

    image = tmp_path / "cam01" / "a.jpg"
    image.parent.mkdir()
    image.write_bytes(b"jpg")
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not received:
        time.sleep(0.05)
    assert received == [[str(image.resolve())]]

    image.unlink()
    _wait_until_called(deleted)
    deleted.assert_called_once_with(str(image.resolve()))


def test_start_watch_drops_deleted_image_from_pending_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_observer: FakeObserver
) -> None:
    on_batch = MagicMock()
    on_deleted = MagicMock()
    folder = tmp_path.resolve()

    def create_and_delete() -> None:
        deleted_handler, created_handler = (h for h, _, _ in fake_observer.handlers)
        created_handler.callback(str(folder / "a.jpg"))
        created_handler.callback(str(folder / "b.jpg"))
        deleted_handler.on_deleted(_event(str(folder / "a.jpg")))

    _stop_watch_on_sleep(monkeypatch, create_and_delete)

    watch_inputs.start_watch(on_batch, on_deleted=on_deleted, batch_window=60)

    assert [(path, recursive) for _, path, recursive in fake_observer.handlers] == [
        (str(folder), True),
        (str(folder), True),
    ]
    on_deleted.assert_called_once_with(str(folder / "a.jpg"))
    # stopping flushes the pending batch, without the deleted image
    on_batch.assert_called_once_with([str(folder / "b.jpg")])
    assert fake_observer.started
    assert fake_observer.stopped
    assert fake_observer.joined


def test_start_watch_without_on_deleted_only_watches_new_images(
    monkeypatch: pytest.MonkeyPatch, fake_observer: FakeObserver
) -> None:
    on_batch = MagicMock()
    _stop_watch_on_sleep(monkeypatch)

    watch_inputs.start_watch(on_batch)

    assert [type(h) for h, _, _ in fake_observer.handlers] == [ImageCreatedHandler]
    on_batch.assert_not_called()  # nothing pending when stopped
    assert fake_observer.stopped
    assert fake_observer.joined


def test_start_watch_raises_if_observer_fails_to_start(
    fake_observer: FakeObserver, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail() -> None:
        raise OSError("inotify watch limit reached")

    monkeypatch.setattr(fake_observer, "start", fail)

    with pytest.raises(OSError, match="inotify"):
        watch_inputs.start_watch(MagicMock())


def test_start_watch_processes_backlog_after_observer_started(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_observer: FakeObserver
) -> None:
    def backlog(*_) -> None:
        # the observer must already run, so no image is missed
        assert fake_observer.started

    mock_backlog = MagicMock(side_effect=backlog)
    monkeypatch.setattr(watch_inputs, "queue_backlog", mock_backlog)
    _stop_watch_on_sleep(monkeypatch)
    backlog_filter = MagicMock()

    watch_inputs.start_watch(
        MagicMock(), process_backlog=True, backlog_filter=backlog_filter
    )

    mock_backlog.assert_called_once_with(
        ANY, backlog_filter, str(tmp_path.resolve())
    )
    assert isinstance(mock_backlog.call_args.args[0], ImageBatcher)
