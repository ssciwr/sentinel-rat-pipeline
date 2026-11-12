"""Health file for the Docker healthchecks of the services of the pipeline image.

A service reports that it is healthy by keeping the file ``HEALTH_FILE``, and
removes it when it is not; the healthcheck in docker-compose.yml tests that the
file exists. Without ``HEALTH_FILE`` (e.g. in tests or when run locally),
nothing is written.
"""

from __future__ import annotations

import logging

from sentinel_rat_pipeline.config import settings

logger = logging.getLogger(__name__)

_warned = False


def report_healthy(healthy: bool) -> None:
    """Create the health file if ``healthy``, else remove it."""

    global _warned
    path = settings.health_file
    if path is None:
        return
    try:
        if healthy:
            path.touch()
        else:
            path.unlink(missing_ok=True)
    except OSError as exc:
        # called every second by the watcher, so warn only once
        if not _warned:
            logger.warning("Could not update the health file %s: %s", path, exc)
            _warned = True
