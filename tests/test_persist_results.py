"""Tests for persist_results helper."""

import struct
from datetime import datetime, timezone
from pathlib import Path
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
    DEFAULT_MODEL_DESCRIPTION,
    _get_captured_time,
    camera_name_from_path,
    delete_result,
    existing_image_paths,
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


def _pack_ifd(entries: list[tuple[int, int, int, bytes]], offset: int) -> bytes:
    """Pack a little-endian TIFF IFD that starts at ``offset`` in the TIFF block.

    Each entry is (tag, type, count, value bytes); values over 4 bytes are
    stored after the IFD and referenced by offset.
    """
    # 2 bytes for entry count, 12 bytes per entry, 4 bytes for next IFD pointer
    data_offset = offset + 2 + 12 * len(entries) + 4
    ifd = struct.pack("<H", len(entries))
    data = b""
    for tag, type_, count, value in sorted(entries):
        if len(value) <= 4:
            ifd += struct.pack("<HHI", tag, type_, count) + value.ljust(4, b"\0")
        else:
            ifd += struct.pack("<HHII", tag, type_, count, data_offset + len(data))
            data += value
    return ifd + struct.pack("<I", 0) + data


def _write_exif_jpeg(
    path: Path,
    *,
    datetime_original: str | None = None,
    offset_time_original: str | None = None,
    image_datetime: str | None = None,
) -> Path:
    """Write a minimal JPEG whose only content is an EXIF APP1 segment."""

    def ascii_entry(tag: int, text: str) -> tuple[int, int, int, bytes]:
        value = text.encode() + b"\0"
        return (tag, 2, len(value), value)

    exif_entries = []
    if datetime_original:
        exif_entries.append(ascii_entry(0x9003, datetime_original))
    if offset_time_original:
        exif_entries.append(ascii_entry(0x9011, offset_time_original))

    ifd0_entries = []
    if image_datetime:
        ifd0_entries.append(ascii_entry(0x0132, image_datetime))
    if exif_entries:
        ifd0_entries.append((0x8769, 4, 1, b"\0\0\0\0"))  # placeholder pointer

    ifd0 = _pack_ifd(ifd0_entries, 8)
    if exif_entries:
        exif_offset = 8 + len(ifd0)
        ifd0_entries[-1] = (0x8769, 4, 1, struct.pack("<I", exif_offset))
        ifd0 = _pack_ifd(ifd0_entries, 8) + _pack_ifd(exif_entries, exif_offset)

    app1 = b"Exif\0\0" + b"II*\0" + struct.pack("<I", 8) + ifd0
    path.write_bytes(
        b"\xff\xd8\xff\xe1" + struct.pack(">H", len(app1) + 2) + app1 + b"\xff\xd9"
    )
    return path


@pytest.mark.parametrize(
    "image_path",
    [
        "image_20260728T170840Z.jpg",
        "hdcamera_image_20260728T170840Z.jpg",
        "/data/samples/image_20260728T170840Z.jpg",
    ],
)
def test_get_captured_time_from_filename(image_path: str) -> None:
    assert _get_captured_time(image_path) == datetime(
        2026, 7, 28, 17, 8, 40, tzinfo=timezone.utc
    )


@pytest.mark.parametrize(
    ("exif", "expected"),
    [
        (
            {
                "datetime_original": "2026:07:28 19:08:40",
                "offset_time_original": "+02:00",
            },
            datetime(2026, 7, 28, 17, 8, 40, tzinfo=timezone.utc),
        ),
        (
            {"datetime_original": "2026:07:28 17:08:40"},
            datetime(2026, 7, 28, 17, 8, 40, tzinfo=timezone.utc),
        ),
        (
            {"datetime_original": "2026:07:28 17:08:40", "offset_time_original": "bad"},
            datetime(2026, 7, 28, 17, 8, 40, tzinfo=timezone.utc),
        ),
        (
            {"image_datetime": "2026:07:28 17:08:40"},
            datetime(2026, 7, 28, 17, 8, 40, tzinfo=timezone.utc),
        ),
        (
            {
                "datetime_original": "2026:07:28 17:08:40",
                "image_datetime": "2026:01:01 00:00:00",
            },
            datetime(2026, 7, 28, 17, 8, 40, tzinfo=timezone.utc),
        ),
        ({"datetime_original": "2026-07-28T17:08:40"}, None),
        ({}, None),
    ],
    ids=[
        "with-offset",
        "no-offset",
        "invalid-offset",
        "image-datetime-fallback",
        "original-preferred",
        "invalid-format",
        "no-dates",
    ],
)
def test_get_captured_time_from_exif(
    tmp_path: Path, exif: dict, expected: datetime | None
) -> None:
    image = _write_exif_jpeg(tmp_path / "CAM01.jpg", **exif)
    assert _get_captured_time(str(image)) == expected


def test_get_captured_time_invalid_filename_date_falls_back_to_exif(
    tmp_path: Path,
) -> None:
    image = _write_exif_jpeg(
        tmp_path / "image_20261399T170840Z.jpg", datetime_original="2026:07:28 17:08:40"
    )
    assert _get_captured_time(str(image)) == datetime(
        2026, 7, 28, 17, 8, 40, tzinfo=timezone.utc
    )


def test_get_captured_time_missing_file() -> None:
    assert _get_captured_time("does/not/exist.jpg") is None


def test_persist_results_unknown_camera(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sentinel_rat_pipeline.scripts.persist_results.settings",
        Settings(database_url="postgresql://fake:5432/db"),
    )
    mock_session = MagicMock()
    mock_session.__enter__.return_value = mock_session
    mock_session.scalars.return_value.all.return_value = []
    mock_session.scalar.return_value = None  # image not stored yet

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
        container = testcontainers.PostgresContainer(
            "postgis/postgis:17-3.5", driver="psycopg"
        )
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


def _mark_moved(engine, image_id: int) -> None:
    with Session(engine) as session, session.begin():
        image = session.get(ImageCapture, image_id)
        image.tobe_deleted = True
        image.is_moved = True


def test_delete_result_cascades(db_settings) -> None:
    image_id = persist_results("CAM01_20260730_115638.jpg", RESULT)
    _mark_moved(db_settings, image_id)
    delete_result("CAM01_20260730_115638.jpg")

    with Session(db_settings) as session:
        for table in (ImageCapture, ObjectDetection, SpeciesClassification):
            assert session.scalar(select(func.count()).select_from(table)) == 0


def test_delete_result_keeps_image_not_moved(db_settings) -> None:
    persist_results("CAM01_20260730_115638.jpg", RESULT)
    delete_result("CAM01_20260730_115638.jpg")

    with Session(db_settings) as session:
        assert session.scalar(select(func.count()).select_from(ImageCapture)) == 1
        assert session.scalar(select(func.count()).select_from(ObjectDetection)) == 2


def test_delete_result_force_deletes_image_not_moved(
    db_settings, db_url, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "sentinel_rat_pipeline.scripts.persist_results.settings",
        Settings(database_url=db_url, force_delete_image=True),
    )
    persist_results("CAM01_20260730_115638.jpg", RESULT)
    delete_result("CAM01_20260730_115638.jpg")

    with Session(db_settings) as session:
        for table in (ImageCapture, ObjectDetection, SpeciesClassification):
            assert session.scalar(select(func.count()).select_from(table)) == 0


def test_persist_results_skips_already_stored_image(db_settings) -> None:
    image_id = persist_results("CAM01_20260730_115638.jpg", RESULT)

    assert persist_results("CAM01_20260730_115638.jpg", RESULT) == image_id
    with Session(db_settings) as session:
        assert session.scalar(select(func.count()).select_from(ImageCapture)) == 1
        assert session.scalar(select(func.count()).select_from(ObjectDetection)) == 2


def test_existing_image_paths(db_settings, monkeypatch: pytest.MonkeyPatch) -> None:
    # force several lookup queries
    monkeypatch.setattr(
        "sentinel_rat_pipeline.scripts.persist_results.LOOKUP_CHUNK_SIZE", 1
    )
    persist_results("CAM01_a.jpg", {"detections": []})
    persist_results("CAM01_b.jpg", {"detections": []})

    assert existing_image_paths(["CAM01_a.jpg", "CAM01_b.jpg", "CAM01_c.jpg"]) == {
        "CAM01_a.jpg",
        "CAM01_b.jpg",
    }
    assert existing_image_paths([]) == set()


@pytest.mark.parametrize(
    ("model_info", "expected"),
    [
        ({"name": "m", "version": "1"}, DEFAULT_MODEL_DESCRIPTION),
        ({"name": "m", "version": "1", "description": None}, DEFAULT_MODEL_DESCRIPTION),
        ({"name": "m", "version": "1", "description": "custom"}, "custom"),
    ],
    ids=["missing", "null", "given"],
)
def test_persist_results_ml_model_description(
    db_settings, model_info: dict, expected: str
) -> None:
    result = {
        "detection_model": model_info,
        "detections": [{**RESULT["detections"][0], "classifications": []}],
    }
    persist_results("CAM01_20260730_115638.jpg", result)

    with Session(db_settings) as session:
        assert session.scalars(select(MLModel.description)).one() == expected
