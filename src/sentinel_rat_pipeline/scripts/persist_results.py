"""Save analysis results to PostgreSQL using the dashboard's data model.

The ML service answers a batch request with ``{"results": [...]}``; each
entry is the result for one image and is persisted on its own. Expected
per-image result (one entry in ``detections`` per detected object, each with
zero or more species classifications)::

    {
        "image_path": "hd-zoo-01/image_20260730T115638Z.jpg",
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
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path, PurePosixPath

from dashboard.db.data_model import (
    Camera,
    ImageCapture,
    MLModel,
    ObjectDetection,
    SpeciesClassification,
    Taxonomy,
)
from sqlalchemy import Engine, create_engine, select
from sqlalchemy.orm import Session

from datetime import datetime, timezone

import exifread

from sentinel_rat_pipeline.config import settings

logger = logging.getLogger(__name__)

# stored for ML models whose response carries no description
DEFAULT_MODEL_DESCRIPTION = "Added by sentinel-rat-pipeline"

# max number of paths per "IN (...)" query when looking up stored images
LOOKUP_CHUNK_SIZE = 1000

# EXIF ImageDescription carries the camera name as "device_id", e.g.
# '{"device_id": "hd-zoo-02", ...}', "device_id=hd-zoo-02" or "device_id: hd-zoo-02"
DEVICE_ID_PATTERN = re.compile(r"device_id[\"']?\s*[:=]\s*[\"']?([^\s\"',;}]+)")

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


@lru_cache
def _engine(database_url: str) -> Engine:
    """One engine (and connection pool) per database, shared by all calls."""
    # pre-ping replaces pooled connections that died, e.g. after a DB restart
    return create_engine(database_url, pool_pre_ping=True)


def camera_folder(image_path: str) -> str | None:
    """Name of the sub-folder of the watch folder that holds the image, or None
    if the image lies directly in the watch folder.

    ``image_path`` is relative to the watch folder; for nested folders the
    top-level one names the camera, e.g. "hd-zoo-02/2026-10/x.jpg" -> "hd-zoo-02".
    An absolute path lies outside the watch folder and raises ValueError.
    """

    path = PurePosixPath(image_path)
    if path.is_absolute():
        raise ValueError(f"{image_path} is outside the watch folder")
    if len(path.parts) < 2:
        return None
    return path.parts[0]


def _read_exif(file_path: str) -> dict | None:
    try:
        with open(file_path, "rb") as f:
            return exifread.process_file(f, details=False)
    except Exception as e:
        logger.warning("Failed to read EXIF data from %s: %s", file_path, e)
        return None


def device_id_from_metadata(file_path: str) -> str | None:
    """Extract the device_id from the EXIF ImageDescription of the image."""

    tags = _read_exif(file_path)
    description = tags.get("Image ImageDescription") if tags else None
    if description is None:
        return None
    match = DEVICE_ID_PATTERN.search(str(description))
    return match.group(1) if match else None


def _find_camera(session: Session, camera_name: str) -> Camera | None:
    cameras = session.scalars(select(Camera).where(Camera.name == camera_name)).all()
    if len(cameras) > 1:
        raise ValueError(f"Multiple cameras named {camera_name!r} in camera table")
    return cameras[0] if cameras else None


def _camera_candidates(folder: str | None, file_path: str) -> Iterator[str]:
    """Camera names to try, in order of priority. Lazy, so the metadata is
    only read if the folder names no known camera."""

    if folder is not None:
        yield folder
    device_id = device_id_from_metadata(file_path)
    if device_id:
        yield device_id
    if folder is None:
        yield settings.default_camera_name


def _resolve_camera(session: Session, image_path: str, file_path: str) -> Camera:
    """Find the camera of an image.

    An image in a sub-folder belongs to the camera named after the folder,
    else to the camera of the device_id in its metadata. An image directly in
    the watch folder belongs to the camera of the device_id in its metadata,
    else to ``settings.default_camera_name``.
    """

    tried = []
    for name in _camera_candidates(camera_folder(image_path), file_path):
        tried.append(name)
        camera = _find_camera(session, name)
        if camera is not None:
            return camera
    raise LookupError(
        f"No camera for {image_path} in camera table (tried {tried or 'nothing'})"
    )


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
            # the ML service may send "description": null, so fall back on falsy too
            description=model_info.get("description") or DEFAULT_MODEL_DESCRIPTION,
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

    tags = _read_exif(image_path)
    if tags is None:
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


def existing_image_paths(image_paths: list[str]) -> set[str]:
    """Return the subset of ``image_paths`` already stored in image_capture."""

    found: set[str] = set()
    with Session(_engine(settings.database_url)) as session:
        for start in range(0, len(image_paths), LOOKUP_CHUNK_SIZE):
            chunk = image_paths[start : start + LOOKUP_CHUNK_SIZE]
            found.update(
                session.scalars(
                    select(ImageCapture.image_path).where(
                        ImageCapture.image_path.in_(chunk)
                    )
                ).all()
            )
    return found


def persist_results(image_path: str, result: dict, file_path: str) -> int:
    """Store an image and its detections/classifications in one transaction.

    ``image_path`` is the path stored in the database, relative to the watch
    folder; ``file_path`` is where the file can be read (e.g. its path in the
    container), for its camera and capture time metadata.

    Returns the id of the new image_capture row. If the image is already
    stored, nothing is written and the id of the existing row is returned.
    """

    with Session(_engine(settings.database_url)) as session, session.begin():
        existing_id = session.scalar(
            select(ImageCapture.id).where(ImageCapture.image_path == image_path)
        )
        if existing_id is not None:
            logger.warning("Image %s is already stored, skipping", image_path)
            return existing_id

        camera = _resolve_camera(session, image_path, file_path)

        image_captured_at = _get_captured_time(file_path)
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
            # SQLAlchemy 2.0 doesn't cascade into the session via backrefs,
            # so setting image_capture= alone won't persist the detection
            session.add(object_detection)
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
    """Handle the removal of an image file from the watch folder.

    By default, image_capture rows marked with ``tobe_deleted`` are flagged
    as ``is_moved``; the rows themselves are kept. If
    ``settings.force_delete_image_in_db`` is set, all rows for the image are
    deleted regardless of ``tobe_deleted``, along with their detections,
    classifications, and corrections (via ORM cascades).
    """

    with Session(_engine(settings.database_url)) as session, session.begin():
        statement = select(ImageCapture).where(ImageCapture.image_path == image_path)
        if settings.force_delete_image_in_db:
            images = session.scalars(statement).all()
            for image in images:
                session.delete(image)
            logger.info(
                "Deleted %s image_capture row(s) for %s", len(images), image_path
            )
            return

        images = session.scalars(
            statement.where(ImageCapture.tobe_deleted.is_(True))
        ).all()
        for image in images:
            image.is_moved = True
        logger.info(
            "Marked %s image_capture row(s) as moved for %s", len(images), image_path
        )
