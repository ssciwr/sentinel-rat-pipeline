"""Tests for persist_results helper."""

from unittest.mock import MagicMock, patch

import pytest
from dashboard.db.data_model import (
    Base,
    Camera,
    ImageCapture,
    MLModel,
    ObjectDetection,
    SpeciesClassification,
    Taxonomy,
)
from sqlalchemy import create_engine, func, select, text
from sqlalchemy.orm import Session

from sentinel_rat_pipeline.config import Settings
from sentinel_rat_pipeline.scripts.persist_results import (
    camera_name_from_path,
    delete_result,
    persist_results,
)

RESULT = {
    "image_path": "CAM01_20260730_115638.jpg",
    "detection_model": {"name": "megadetector", "version": "v5a"},
    "classification_model": {"name": "rodent-classifier", "version": "0.1"},
    "detections": [
        {
            "detected_class": "animal",
            "confidence": 0.92,
            "bbox": {"x_min": 0.1, "y_min": 0.2, "x_max": 0.4, "y_max": 0.5},
            "classifications": [
                {"genus": "Rattus", "species": "rattus", "confidence": 0.87},
                {"genus": "Rattus", "species": "norvegicus", "confidence": 0.1},
            ],
        },
        {
            "detected_class": "animal",
            "confidence": 0.8,
            "bbox": {"x_min": 0.5, "y_min": 0.5, "x_max": 0.7, "y_max": 0.9},
            "classifications": [
                {"genus": "Rattus", "species": "rattus", "confidence": 0.6},
            ],
        },
    ],
}


@pytest.mark.parametrize(
    ("image_path", "expected"),
    [
        ("CAM01_20260730_115638.jpg", "CAM01"),
        ("sub/dir/cam-2_x.png", "cam-2"),
        ("CAM03.jpg", "CAM03"),
    ],
)
def test_camera_name_from_path(image_path: str, expected: str) -> None:
    assert camera_name_from_path(image_path) == expected


def test_camera_name_from_path_empty() -> None:
    with pytest.raises(ValueError):
        camera_name_from_path("_20260730.jpg")


def test_persist_results_unknown_camera(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sentinel_rat_pipeline.scripts.persist_results.settings",
        Settings(database_url="postgresql://fake:5432/db"),
    )
    mock_session = MagicMock()
    mock_session.__enter__.return_value = mock_session
    mock_session.scalars.return_value.all.return_value = []

    with (
        patch("sentinel_rat_pipeline.scripts.persist_results.create_engine"),
        patch(
            "sentinel_rat_pipeline.scripts.persist_results.Session",
            return_value=mock_session,
        ),
        pytest.raises(LookupError, match="CAM01"),
    ):
        persist_results("CAM01_20260730_115638.jpg", RESULT)

    mock_session.add.assert_not_called()


# --- integration tests against a real PostGIS database ---


@pytest.fixture(scope="module")
def db_url():
    testcontainers = pytest.importorskip("testcontainers.community.postgres")
    try:
        container = testcontainers.PostgresContainer("postgis/postgis:17-3.5", driver="psycopg")
        container.start()
    except Exception as exc:
        pytest.skip(f"Docker not available: {exc}")

    url = container.get_connection_url()
    engine = create_engine(url)
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis"))
    Base.metadata.create_all(engine)
    with Session(engine) as session, session.begin():
        session.add(Camera(name="CAM01", location="SRID=4326;POINT(80.6 7.3)"))
    engine.dispose()

    yield url
    container.stop()


@pytest.fixture
def db_settings(db_url, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "sentinel_rat_pipeline.scripts.persist_results.settings",
        Settings(database_url=db_url),
    )
    engine = create_engine(db_url)
    yield engine
    with Session(engine) as session, session.begin():
        for table in (ImageCapture, Taxonomy, MLModel):
            for row in session.scalars(select(table)).all():
                session.delete(row)
    engine.dispose()


def test_persist_results_writes_all_tables(db_settings) -> None:
    image_id = persist_results("CAM01_20260730_115638.jpg", RESULT)

    with Session(db_settings) as session:
        image = session.get(ImageCapture, image_id)
        assert image.image_path == "CAM01_20260730_115638.jpg"
        assert image.camera.name == "CAM01"
        assert (
            session.scalar(select(func.ST_AsText(ImageCapture.location)))
            == "POINT(80.6 7.3)"
        )

        detections = image.object_detections
        assert len(detections) == 2
        assert {d.confidence for d in detections} == {0.92, 0.8}
        assert detections[0].bbox == RESULT["detections"][0]["bbox"]
        assert detections[0].ml_model.task == "detection"

        classifications = session.scalars(select(SpeciesClassification)).all()
        assert len(classifications) == 3
        assert all(c.ml_model.task == "classification" for c in classifications)

        # taxonomy and ml_model rows are reused, not duplicated
        assert session.scalar(select(func.count()).select_from(Taxonomy)) == 2
        assert session.scalar(select(func.count()).select_from(MLModel)) == 2

    persist_results("CAM01_20260730_120000.jpg", RESULT)
    with Session(db_settings) as session:
        assert session.scalar(select(func.count()).select_from(Taxonomy)) == 2
        assert session.scalar(select(func.count()).select_from(MLModel)) == 2


def test_persist_results_no_detections(db_settings) -> None:
    image_id = persist_results("CAM01_empty.jpg", {"detections": []})

    with Session(db_settings) as session:
        assert session.get(ImageCapture, image_id).object_detections == []


def test_persist_results_unknown_camera_rolls_back(db_settings) -> None:
    with pytest.raises(LookupError):
        persist_results("CAM99_20260730_115638.jpg", RESULT)

    with Session(db_settings) as session:
        assert session.scalar(select(func.count()).select_from(ImageCapture)) == 0


def test_delete_result_cascades(db_settings) -> None:
    persist_results("CAM01_20260730_115638.jpg", RESULT)
    delete_result("CAM01_20260730_115638.jpg")

    with Session(db_settings) as session:
        for table in (ImageCapture, ObjectDetection, SpeciesClassification):
            assert session.scalar(select(func.count()).select_from(table)) == 0
