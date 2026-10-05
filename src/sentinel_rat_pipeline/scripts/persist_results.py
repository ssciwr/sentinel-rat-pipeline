"""Save analysis results to PostgreSQL using the dashboard's data model.

Expected ML service response (one entry in ``detections`` per detected object,
each with zero or more species classifications)::

    {
        "image_path": "CAM01_20260730_115638.jpg",
        "detection_model": {"name": "megadetector", "version": "v5a"},
        "classification_model": {"name": "rodent-classifier", "version": "0.1"},
        "detections": [
            {
                "detected_class": "animal",
                "confidence": 0.92,
                "bbox": {"x_min": 0.1, "y_min": 0.2, "x_max": 0.4, "y_max": 0.5},
                "classifications": [
                    {
                        "genus": "Rattus",
                        "species": "rattus",
                        "common_name": "black rat",
                        "confidence": 0.87,
                    }
                ],
            }
        ],
    }
"""

from __future__ import annotations

import logging
from pathlib import Path

from dashboard.db.data_model import (
    Camera,
    ImageCapture,
    MLModel,
    ObjectDetection,
    SpeciesClassification,
    Taxonomy,
)
from sqlalchemy import create_engine, select  # type: ignore[import-untyped]
from sqlalchemy.orm import Session  # type: ignore[import-untyped]

from sentinel_rat_pipeline.config import settings

logger = logging.getLogger(__name__)

# image files are named "<camera_name>_<anything>.<ext>", e.g. "CAM01_20260730_115638.jpg"
CAMERA_NAME_SEPARATOR = "_"

TAXONOMY_FIELDS = (
    "kingdom",
    "phylum",
    "class_name",
    "order",
    "family",
    "genus",
    "species",
    "common_name",
)


def camera_name_from_path(image_path: str) -> str:
    """Extract the camera name from the image file name."""

    camera_name = Path(image_path).stem.split(CAMERA_NAME_SEPARATOR, 1)[0]
    if not camera_name:
        raise ValueError(f"Cannot extract camera name from {image_path}")
    return camera_name


def _get_camera(session: Session, camera_name: str) -> Camera:
    cameras = session.scalars(select(Camera).where(Camera.name == camera_name)).all()
    if not cameras:
        raise LookupError(f"No camera named {camera_name!r} in camera table")
    if len(cameras) > 1:
        raise ValueError(f"Multiple cameras named {camera_name!r} in camera table")
    return cameras[0]


def _get_or_create_ml_model(session: Session, model_info: dict, task: str) -> MLModel:
    name = model_info["name"]
    version = model_info.get("version")
    ml_model = session.scalars(
        select(MLModel).where(
            MLModel.name == name,
            MLModel.version.is_(None) if version is None else MLModel.version == version,
            MLModel.task == task,
        )
    ).first()
    if ml_model is None:
        ml_model = MLModel(
            name=name,
            version=version,
            task=task,
            description=model_info.get("description"),
        )
        session.add(ml_model)
    return ml_model


def _get_or_create_taxonomy(session: Session, classification: dict) -> Taxonomy:
    genus = classification["genus"]
    species = classification.get("species")
    taxonomy = session.scalars(
        select(Taxonomy).where(
            Taxonomy.genus == genus,
            Taxonomy.species.is_(None) if species is None else Taxonomy.species == species,
        )
    ).first()
    if taxonomy is None:
        taxonomy = Taxonomy(
            **{field: classification.get(field) for field in TAXONOMY_FIELDS}
        )
        session.add(taxonomy)
    return taxonomy


def persist_results(image_path: str, result: dict) -> int:
    """Store an image and its detections/classifications in one transaction.

    Returns the id of the new image_capture row.
    """

    engine = create_engine(settings.database_url)
    with Session(engine) as session, session.begin():
        camera = _get_camera(session, camera_name_from_path(image_path))

        image_capture = ImageCapture(
            camera_id=camera.id,
            image_path=image_path,
            location=camera.location,  # current location of the camera
        )
        session.add(image_capture)

        detections = result.get("detections", [])
        det_model = clas_model = None
        if detections:
            det_model = _get_or_create_ml_model(
                session, result["detection_model"], "detection"
            )
        if any(det.get("classifications") for det in detections):
            clas_model = _get_or_create_ml_model(
                session, result["classification_model"], "classification"
            )

        for det in detections:
            object_detection = ObjectDetection(
                image_capture=image_capture,
                ml_model=det_model,
                confidence=det["confidence"],
                bbox=det["bbox"],
                detected_class=det["detected_class"],
            )
            for clas in det.get("classifications", []):
                object_detection.species_classifications.append(
                    SpeciesClassification(
                        taxonomy=_get_or_create_taxonomy(session, clas),
                        ml_model=clas_model,
                        confidence=clas["confidence"],
                    )
                )

        session.flush()
        logger.info(
            "Persisted image %s (camera %s) with %s detection(s)",
            image_path,
            camera.name,
            len(detections),
        )
        return image_capture.id


def delete_result(image_path: str) -> None:
    """Delete image_capture rows for an image along with their detections,
    classifications, and corrections (via ORM cascades)."""

    engine = create_engine(settings.database_url)
    with Session(engine) as session, session.begin():
        images = session.scalars(
            select(ImageCapture).where(ImageCapture.image_path == image_path)
        ).all()
        for image in images:
            session.delete(image)
        logger.info("Deleted %s image_capture row(s) for %s", len(images), image_path)
