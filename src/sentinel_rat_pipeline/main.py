"""Entrypoint for the Sentinel Rat Pipeline."""

from __future__ import annotations

import logging

from pathlib import Path

from sentinel_rat_pipeline.config import settings
from sentinel_rat_pipeline.scripts.persist_results import delete_result, persist_results
from sentinel_rat_pipeline.scripts.run_analysis import run_analysis_remote
from sentinel_rat_pipeline.scripts.watch_inputs import start_watch

import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def on_new_image(image_path: str) -> None:
    """Process a newly detected image.

    Convert the container-internal path to a path relative to the host's
    watch folder before passing it downstream so the stored ``image_path``
    matches what an operator sees on the host filesystem.
    """
    try:
        host_relative = _to_host_relative(image_path)
    except ValueError:
        logger.warning(
            "Could not make host-relative path for %s (watch_folder=%s), "
            "storing full container path",
            image_path,
            settings.watch_folder,
        )
        host_relative = image_path

    logger.info("*** on_new_image callback entered: %s ***", host_relative)
    try:
        result = run_analysis_remote(host_relative)
        logger.info("Analysis result: %s", result)
    except Exception as exc:
        logger.error("Analysis failed for %s: %s", host_relative, exc)
        return
    try:
        time_id = time.strftime("%Y%m%d%H%M%S")
        persist_results(result, detection_id=time_id)
        logger.info("Persisted result for %s", host_relative)
    except Exception as exc:
        logger.error("Persistence failed for %s: %s", host_relative, exc)


def _to_host_relative(container_path: str) -> str:
    """Strip the container watch folder prefix to yield a host-relative path."""
    from pathlib import PurePosixPath

    watch = PurePosixPath(settings.watch_folder)
    path = PurePosixPath(container_path)
    return str(path.relative_to(watch))


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

    from sentinel_rat_pipeline.scripts.watch_inputs import list_images

    for existing in list_images():
        logger.info("Processing existing image: %s", existing)
        on_new_image(existing)

    logger.info("Entering watchdog loop...")
    start_watch(on_new_image, on_deleted=on_image_deleted)


if __name__ == "__main__":
    main()
