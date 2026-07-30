"""Call the ML pipeline for a single image."""

from __future__ import annotations

import logging
import time

import httpx

from sentinel_rat_pipeline.config import settings

logger = logging.getLogger(__name__)


def run_analysis(image_path: str) -> dict:
    # TODO: replace with real ML pipeline call
    logger.info("Analyzing %s", image_path)
    return {
        "image_path": image_path,
        "animals_detected": 0,
        "species": {},
        "confidence": 0.0,
    }


def run_analysis_remote(image_path: str, retries: int = 5, delay: float = 1.0) -> dict:
    url = f"{settings.ml_pipeline_url.rstrip('/')}/api/v1/analyze"
    payload = {"image_path": image_path}
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with httpx.Client(timeout=60.0) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()
                return response.json()
        except Exception as exc:
            last_exc = exc
            logger.warning("ML service attempt %s/%s failed: %s", attempt, retries, exc)
            time.sleep(delay * attempt)
    raise RuntimeError(f"ML service unavailable after {retries} attempts") from last_exc
