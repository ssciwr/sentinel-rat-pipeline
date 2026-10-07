"""Entrypoint for the Sentinel Rat Pipeline."""

from __future__ import annotations

import logging

from pathlib import Path

from sentinel_rat_pipeline.config import settings
from sentinel_rat_pipeline.scripts.persist_results import (
    delete_result,
    existing_image_paths,
    persist_results,
)
from sentinel_rat_pipeline.scripts.run_analysis import run_analysis_remote
from sentinel_rat_pipeline.scripts.watch_inputs import start_watch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def on_new_images(image_paths: list[str]) -> None:
    """Analyze a batch of newly detected images and persist each result.

    Convert the container-internal paths to paths relative to the host's
    watch folder before passing them downstream so the stored ``image_path``
    matches what an operator sees on the host filesystem. Batches larger than
    ``settings.max_batch_size`` are split and sent to the ML service in chunks.
    """
    host_paths = [_host_path(image_path) for image_path in image_paths]

    logger.info("*** on_new_images callback entered: %s image(s) ***", len(host_paths))
    size = settings.max_batch_size
    for start in range(0, len(host_paths), size):
        _analyze_and_persist(host_paths[start : start + size])


def _analyze_and_persist(host_paths: list[str]) -> None:
    """Send one chunk to the ML service and persist each of its results."""
    try:
        results = run_analysis_remote(host_paths)
        logger.info("Analysis results: %s", results)
    except Exception as exc:
        logger.error("Analysis failed for %s: %s", host_paths, exc)
        return

    results_by_path = {result.get("image_path"): result for result in results}
    for host_path in host_paths:
        result = results_by_path.get(host_path)
        if result is None:
            logger.error("ML service returned no result for %s", host_path)
            continue
        # one transaction per image, so a failing image doesn't drop the batch
        try:
            persist_results(host_path, result)
            logger.info("Persisted result for %s", host_path)
        except Exception as exc:
            logger.error("Persistence failed for %s: %s", host_path, exc)


def _host_path(image_path: str) -> str:
    """Host-relative path of an image, or the full path if it is outside the
    watch folder."""
    try:
        return _to_host_relative(image_path)
    except ValueError:
        logger.warning(
            "Could not make host-relative path for %s (watch_folder=%s), "
            "storing full container path",
            image_path,
            settings.watch_folder,
        )
        return image_path


def _unprocessed_images(image_paths: list[str]) -> list[str]:
    """Drop images that are already stored in the database."""
    host_paths = {image_path: _host_path(image_path) for image_path in image_paths}
    stored = existing_image_paths(list(host_paths.values()))
    if stored:
        logger.info("Skipping %s image(s) already in the database", len(stored))
    return [path for path, host in host_paths.items() if host not in stored]


def _to_host_relative(container_path: str) -> str:
    """Strip the container watch folder prefix to yield a host-relative path."""
    from pathlib import PurePosixPath

    watch = PurePosixPath(settings.watch_folder)
    path = PurePosixPath(container_path)
    try:
        return str(path.relative_to(watch))
    except ValueError:
        # the watcher reports paths under the resolved watch folder
        return str(path.relative_to(Path(settings.watch_folder).resolve()))


def on_image_deleted(image_path: str) -> None:
    logger.info("*** on_image_deleted callback entered: %s ***", image_path)
    host_relative = _to_host_relative(image_path)
    try:
        delete_result(host_relative)
        logger.info("Delete issued for %s", host_relative)
    except Exception as exc:
        logger.error("Delete failed for %s: %s", host_relative, exc)


def main() -> None:
    logger.info("Starting Sentinel Rat Pipeline")

    logger.info(
        "Loaded settings: watch_folder=%s database_url=%s ml_pipeline_url=%s",
        settings.watch_folder,
        settings.database_url,
        settings.ml_pipeline_url,
    )

    watch_path = Path(settings.watch_folder)
    if not watch_path.exists() or not watch_path.is_dir():
        logger.error(
            "Watch folder does not exist or is not a directory: %s", watch_path
        )
        return

    if not settings.database_url:
        logger.error("DATABASE_URL is not set")
        return

    logger.info("Watch folder: %s", watch_path)
    logger.info("ML service: %s", settings.ml_pipeline_url)
    logger.info("Database: %s", settings.database_url)

    if not settings.process_backlog:
        logger.info("Skipping existing images (PROCESS_BACKLOG is disabled)")

    # the backlog is handled inside start_watch, after the observer is running,
    # so images added meanwhile are not missed
    logger.info(
        "Entering watchdog loop (batch window %ss)...", settings.batch_window_seconds
    )
    start_watch(
        on_new_images,
        on_deleted=on_image_deleted,
        process_backlog=settings.process_backlog,
        backlog_filter=_unprocessed_images,
    )


if __name__ == "__main__":
    main()
