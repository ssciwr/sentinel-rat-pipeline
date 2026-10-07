"""Resolve new images in the watch folder using watchdog and group them into
time-windowed batches."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Callable

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from sentinel_rat_pipeline.config import settings

logger = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

# A file counts as fully copied once its mtime is unchanged for STABLE_CHECKS
# polls, STABLE_INTERVAL seconds apart. Give up if the file stays missing for
# STABLE_TIMEOUT seconds (e.g. it was removed before it became stable).
STABLE_CHECKS = 3
STABLE_INTERVAL = 0.5
STABLE_TIMEOUT = 300.0

# Dedup: src_path -> mtime when it was claimed for processing. Shared by the
# watcher and the startup backlog, so each image is processed only once.
_SEEN: dict[str, float] = {}
_seen_lock = threading.Lock()


def _claim(src_path: str) -> bool:
    """Mark an image as taken; return False if it was already claimed."""
    with _seen_lock:
        if src_path in _SEEN:
            return False
        try:
            _SEEN[src_path] = Path(src_path).stat().st_mtime
        except OSError:
            return False  # removed in the meantime
        return True


def _is_image(src_path: str) -> bool:
    p = Path(src_path)
    return p.suffix.lower() in IMAGE_SUFFIXES and not p.is_dir()


def _wait_for_stable_file(src_path: str, callback: Callable[[str], None]) -> None:
    """Poll mtime until the file stops changing (copy is complete)."""

    deadline = time.monotonic() + STABLE_TIMEOUT
    last_mtime: float | None = None
    stable_count = 0
    while stable_count < STABLE_CHECKS:
        try:
            mtime = Path(src_path).stat().st_mtime
        except OSError:
            if time.monotonic() > deadline:
                logger.warning(
                    "watchdog: %s missing for %ss, skipping", src_path, STABLE_TIMEOUT
                )
                return
            stable_count = 0
            last_mtime = None
            time.sleep(STABLE_INTERVAL)
            continue

        deadline = time.monotonic() + STABLE_TIMEOUT
        if mtime == last_mtime:
            stable_count += 1
        else:
            stable_count = 0
            last_mtime = mtime
        time.sleep(STABLE_INTERVAL)

    # Dedup: skip if another handler or the backlog already took this file
    if _claim(src_path):
        logger.info("watchdog: invoking callback for: %s", src_path)
        try:
            callback(src_path)
        except Exception as exc:
            logger.exception("watchdog handler error for %s: %s", src_path, exc)


class ImageBatcher:
    """Collect stable images and hand them over as one batch.

    The first image added to an empty batch opens a time window of
    ``window`` seconds; every image arriving within that window joins the
    batch, which is passed to ``on_batch`` once the window closes.
    """

    def __init__(self, on_batch: Callable[[list[str]], None], window: float) -> None:
        self.callback = on_batch
        self.window = window
        self._pending: list[str] = []
        self._timer: threading.Timer | None = None
        self._lock = threading.Lock()

    def add(self, src_path: str) -> None:
        with self._lock:
            if src_path in self._pending:
                return
            self._pending.append(src_path)
            if self._timer is None:
                logger.info("batch: window of %ss opened by %s", self.window, src_path)
                self._timer = threading.Timer(self.window, self.flush)
                self._timer.daemon = True
                self._timer.start()

    def discard(self, src_path: str) -> None:
        """Drop an image that disappeared before its batch was processed."""
        with self._lock:
            if src_path in self._pending:
                self._pending.remove(src_path)

    def flush(self) -> None:
        """Process the pending batch now and close the current window."""
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
            batch, self._pending = self._pending, []

        if not batch:
            return
        logger.info("batch: invoking callback for %s image(s)", len(batch))
        try:
            self.callback(batch)
        except Exception as exc:
            logger.exception("batch handler error for %s: %s", batch, exc)


class ImageCreatedHandler(FileSystemEventHandler):
    """Queue file for processing once it is stable on disk."""

    def __init__(self, on_created: Callable[[str], None]) -> None:
        self.callback = on_created

    def on_created(self, event: FileSystemEvent) -> None:  # type: ignore[override]
        src = str(getattr(event, "src_path", ""))
        if not src or event.is_directory:
            return

        logger.info(
            "watchdog: src=%s dest=%s suffix=%s",
            src,
            getattr(event, "dest_path", ""),
            Path(src).suffix,
        )

        if not _is_image(src):
            logger.info("watchdog: ignored non-image suffix: %s", Path(src).suffix)
            return

        with _seen_lock:
            if src in _SEEN:
                logger.info("watchdog: already scheduled, skipping: %s", src)
                return

        logger.info("watchdog: queued for processing: %s", src)
        threading.Thread(
            target=_wait_for_stable_file,
            args=(src, self.callback),
            daemon=True,
        ).start()


class ImageDeletedHandler(FileSystemEventHandler):
    """Delete database rows when a watched image is removed."""

    def __init__(self, on_deleted: Callable[[str], None]) -> None:
        self.callback = on_deleted

    def on_deleted(self, event: FileSystemEvent) -> None:  # type: ignore[override]
        src = str(getattr(event, "src_path", ""))
        if not src or event.is_directory:
            return
        if not _is_image(src):
            return
        logger.info("watchdog: file deleted: %s", src)
        try:
            self.callback(src)
            # update _SEEN as well, so that if the file is recreated, it will be reprocessed
            with _seen_lock:
                if src in _SEEN:
                    del _SEEN[src]
        except Exception as exc:
            logger.exception("watchdog delete handler error for %s: %s", src, exc)


def list_images(watch_folder: str | None = None) -> list[str]:
    folder = Path(watch_folder or settings.watch_folder)
    return [str(p) for p in folder.glob("*") if p.suffix.lower() in IMAGE_SUFFIXES]


def _split_stable(image_paths: list[str]) -> tuple[list[str], list[str]]:
    """Split images into fully copied and still changing ones.

    Checks all files at once instead of polling each one: stat everything,
    wait as long as a per-file check would, and stat again. Files that
    disappeared in the meantime are dropped.
    """

    def snapshot() -> dict[str, tuple[float, int]]:
        stats = {}
        for image_path in image_paths:
            try:
                st = Path(image_path).stat()
            except OSError:
                continue
            stats[image_path] = (st.st_mtime, st.st_size)
        return stats

    before = snapshot()
    time.sleep(STABLE_CHECKS * STABLE_INTERVAL)
    after = snapshot()

    stable = [p for p in image_paths if p in after and before.get(p) == after[p]]
    changing = [p for p in image_paths if p in after and before.get(p) != after[p]]
    return stable, changing


def queue_backlog(
    batcher: ImageBatcher,
    backlog_filter: Callable[[list[str]], list[str]] | None = None,
    watch_folder: str | None = None,
) -> None:
    """Process images that were already in the watch folder.

    Call this only after the observer is running, so images created while the
    backlog is handled are not missed. ``backlog_filter`` drops images that
    must not be processed again (e.g. already stored in the database).
    Backlog and watcher claim images through the same ``_SEEN`` registry, so
    an image seen by both is processed once.
    """

    image_paths = list_images(watch_folder)
    if backlog_filter is not None and image_paths:
        try:
            image_paths = backlog_filter(image_paths)
        except Exception as exc:
            # better to skip the backlog than to store images twice
            logger.error("backlog: filter failed, skipping backlog: %s", exc)
            return
    if not image_paths:
        logger.info("backlog: no images to process")
        return

    stable, changing = _split_stable(image_paths)
    logger.info(
        "backlog: %s image(s) ready, %s still being copied", len(stable), len(changing)
    )
    for image_path in changing:
        threading.Thread(
            target=_wait_for_stable_file,
            args=(image_path, batcher.add),
            daemon=True,
        ).start()
    for image_path in stable:
        if _claim(image_path):
            batcher.add(image_path)
    # no need to wait for the batch window, the backlog is complete
    batcher.flush()


def start_watch(
    on_batch: Callable[[list[str]], None],
    on_deleted: Callable[[str], None] | None = None,
    poll_interval: float = 1.0,
    batch_window: float | None = None,
    process_backlog: bool = False,
    backlog_filter: Callable[[list[str]], list[str]] | None = None,
) -> None:
    """Watch the folder and pass new images to ``on_batch`` in batches
    collected over ``batch_window`` seconds (default from settings).

    With ``process_backlog``, images already in the folder are processed too,
    after the observer has started (see ``queue_backlog``).
    """

    folder = Path(settings.watch_folder).resolve()
    folder.mkdir(parents=True, exist_ok=True)

    batcher = ImageBatcher(
        on_batch,
        settings.batch_window_seconds if batch_window is None else batch_window,
    )
    created_handler = ImageCreatedHandler(batcher.add)
    observer = Observer()

    if on_deleted is not None:

        def handle_deleted(src_path: str) -> None:
            batcher.discard(src_path)
            on_deleted(src_path)

        deleted_handler = ImageDeletedHandler(handle_deleted)
        observer.schedule(deleted_handler, str(folder), recursive=False)

    try:
        observer.schedule(created_handler, str(folder), recursive=False)
        observer.start()
    except Exception as exc:
        logger.error("Failed to start watchdog observer: %s", exc)
        raise

    logger.info("Watching %s", folder)
    if process_backlog:
        queue_backlog(batcher, backlog_filter, str(folder))
    try:
        while True:
            time.sleep(poll_interval)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
    batcher.flush()
