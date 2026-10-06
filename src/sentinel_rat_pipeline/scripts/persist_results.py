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
import re
from pathlib import Path

from dashboard.db.data_model import (
    Camera,
    ImageCapture,
    MLModel,
    ObjectDetection,
    SpeciesClassification,
    Taxonomy,
)
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from datetime import datetime, timezone

import exifread

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
            (
                MLModel.version.is_(None)
                if version is None
                else MLModel.version == version
            ),
            MLModel.task == task,
        )
    ).first()
    if ml_model is None:
        ml_model = MLModel(
            name=name,
            version=version,
            task=task,
            description=model_info.get("description", "Added by sentinel-rat-pipeline"),
        )
        session.add(ml_model)
    return ml_model


def _get_or_create_taxonomy(session: Session, classification: dict) -> Taxonomy:
    genus = classification["genus"]
    species = classification.get("species")
    taxonomy = session.scalars(
        select(Taxonomy).where(
            Taxonomy.genus == genus,
            (
                Taxonomy.species.is_(None)
                if species is None
                else Taxonomy.species == species
            ),
        )
    ).first()
    if taxonomy is None:
        taxonomy = Taxonomy(
            **{field: classification.get(field) for field in TAXONOMY_FIELDS}
        )
        session.add(taxonomy)
    return taxonomy


def _get_captured_time(image_path: str) -> datetime | None:
    """Extract the capture time from the image file name, if present.
    Otherwise, try to read the EXIF data from the image file.

    Expected filename format: [<camera_name>_]image_<YYYYMMDD>T<HHMMSS>Z.<ext>
    e.g. hdcamera_image_20260728T170840Z.jpg or image_20260728T170840Z.jpg
    """

    stem = Path(image_path).stem
    match = re.search(r"(?:^|_)image_(\d{8}T\d{6}Z)$", stem)
    if match:
        try:
            captured_time = datetime.strptime(match.group(1), "%Y%m%dT%H%M%SZ")
            return captured_time.replace(tzinfo=timezone.utc)
        except ValueError:
            pass  # invalid date values, fall back to EXIF

    try:
        with open(image_path, "rb") as f:
            tags = exifread.process_file(f, details=False)
    except Exception as e:
        logger.warning("Failed to read EXIF data from %s: %s", image_path, e)
        return None

    raw = tags.get("EXIF DateTimeOriginal") or tags.get("Image DateTime")
    if raw is None:
        logger.warning(
            "No EXIF DateTimeOriginal or Image DateTime found in %s", image_path
        )
        return None

    try:
        captured_time = datetime.strptime(str(raw), "%Y:%m:%d %H:%M:%S")
    except ValueError:
        logger.warning("Invalid EXIF date format in %s: %s", image_path, raw)
        return None

    # add offset time if available
    offset_raw = tags.get("EXIF OffsetTimeOriginal")
    if offset_raw is not None:
        try:
            captured_time = datetime.strptime(
                f"{captured_time:%Y-%m-%d %H:%M:%S}{str(offset_raw).strip()}",
                "%Y-%m-%d %H:%M:%S%z",
            ).astimezone(timezone.utc)
        except ValueError:
            pass

    # no (valid) offset: assume the camera clock is set to UTC
    if captured_time.tzinfo is None:
        captured_time = captured_time.replace(tzinfo=timezone.utc)
    return captured_time


def persist_results(image_path: str, result: dict) -> int:
    """Store an image and its detections/classifications in one transaction.

    Returns the id of the new image_capture row.
    """

    engine = create_engine(settings.database_url)
    with Session(engine) as session, session.begin():
        camera = _get_camera(session, camera_name_from_path(image_path))

        image_captured_at = _get_captured_time(image_path)
        if image_captured_at is None:
            # use the current time if we can't determine the capture time
            image_captured_at = datetime.now(timezone.utc)

        image_capture = ImageCapture(
            camera_id=camera.id,
            image_path=image_path,
            captured_at=image_captured_at,
            uploaded_at=datetime.now(timezone.utc),
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
