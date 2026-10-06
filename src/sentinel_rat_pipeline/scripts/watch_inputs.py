"""Resolve new images in the watch folder using watchdog."""

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

# Dedup: src_path -> mtime when it was first queued
_SEEN: dict[str, float] = {}
_seen_lock = threading.Lock()


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

    # Dedup: skip if another handler already processed this file
    with _seen_lock:
        if src_path not in _SEEN:
            _SEEN[src_path] = Path(src_path).stat().st_mtime
            should_process = True
        else:
            should_process = False

    if should_process:
        logger.info("watchdog: invoking callback for: %s", src_path)
        try:
            callback(src_path)
        except Exception as exc:
            logger.exception("watchdog handler error for %s: %s", src_path, exc)


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


def start_watch(
    on_created: Callable[[str], None],
    on_deleted: Callable[[str], None] | None = None,
    poll_interval: float = 1.0,
) -> None:
    folder = Path(settings.watch_folder).resolve()
    folder.mkdir(parents=True, exist_ok=True)

    created_handler = ImageCreatedHandler(on_created)
    observer = Observer()

    if on_deleted is not None:
        deleted_handler = ImageDeletedHandler(on_deleted)
        observer.schedule(deleted_handler, str(folder), recursive=False)

    try:
        observer.schedule(created_handler, str(folder), recursive=False)
        observer.start()
    except Exception as exc:
        logger.error("Failed to start watchdog observer: %s", exc)
        raise

    logger.info("Watching %s", folder)
    try:
        while True:
            time.sleep(poll_interval)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
