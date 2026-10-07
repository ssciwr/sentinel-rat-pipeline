"""Call the ML pipeline for a batch of images."""

from __future__ import annotations

import logging
import time

import httpx

from sentinel_rat_pipeline.config import settings

logger = logging.getLogger(__name__)


def run_analysis_remote(
    image_paths: list[str], retries: int = 5, delay: float = 1.0
) -> list[dict]:
    """Analyze several images in one request.

    The ML service receives ``{"image_paths": [...]}`` and answers with
    ``{"results": [...]}``, one result per image carrying its ``image_path``.
    """

    url = f"{settings.ml_pipeline_url.rstrip('/')}/api/v1/analyze"
    payload = {"image_paths": image_paths}
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            # a batch can take a while to analyze, so allow more than a single image
            with httpx.Client(timeout=300.0) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()
                return response.json()["results"]
        except Exception as exc:
            last_exc = exc
            logger.warning("ML service attempt %s/%s failed: %s", attempt, retries, exc)
            time.sleep(delay * attempt)
    raise RuntimeError(f"ML service unavailable after {retries} attempts") from last_exc
