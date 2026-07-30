"""Save analysis results to PostgreSQL."""

from __future__ import annotations

import json
import logging

from sqlalchemy import create_engine, text  # type: ignore[import-untyped]

from sentinel_rat_pipeline.config import settings

logger = logging.getLogger(__name__)

def persist_results(result: dict, detection_id: str | None = None) -> None:
    # TODO: switch to ORM once schema is defined
    engine = create_engine(settings.database_url)
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO analysis_results (detection_id, image_path, animals_detected, species, confidence)
                VALUES (:detection_id, :image_path, :animals_detected, :species, :confidence)
                """
            ),
            {
                "detection_id": detection_id or result.get("image_path"),
                "image_path": result.get("image_path"),
                "animals_detected": result.get("animals_detected", 0),
                "species": json.dumps(result.get("species", {})),
                "confidence": result.get("confidence", 0.0),
            },
        )
        logger.info("Persisted result for %s", result.get("image_path"))


def delete_result(image_path: str) -> None:
    engine = create_engine(settings.database_url)
    with engine.begin() as conn:
        result = conn.execute(
            text("DELETE FROM analysis_results WHERE image_path = :image_path"),
            {"image_path": image_path},
        )
        logger.info(
            "Deleted %s row(s) for %s", result.rowcount, image_path
        )
