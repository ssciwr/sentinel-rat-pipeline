"""Tests for the daily analysis scheduler."""

import threading
from datetime import date, datetime, time
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pytest
from dashboard.db.data_model import (
    Base,
    Camera,
    DailyAnalysisResult,
    ImageCapture,
    MLModel,
    ObjectDetection,
    SpeciesClassification,
    Taxonomy,
)
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import Session

from sentinel_rat_pipeline import daily_analysis
from sentinel_rat_pipeline.config import Settings

TZ = "Asia/Colombo"


def _local(day: date, hour: int) -> datetime:
    return datetime.combine(day, time(hour), tzinfo=ZoneInfo(TZ))


@pytest.mark.parametrize(
    ("now", "expected"),
    [
        (_local(date(2026, 10, 1), 10), _local(date(2026, 10, 2), 0)),
        (_local(date(2026, 10, 1), 0), _local(date(2026, 10, 2), 0)),
    ],
)
def test_next_run_is_the_next_midnight(now: datetime, expected: datetime) -> None:
    assert daily_analysis.next_run(now, time(0, 0)) == expected


def test_next_run_later_today() -> None:
    now = _local(date(2026, 10, 1), 0)
    assert daily_analysis.next_run(now, time(1, 30)) == datetime(
        2026, 10, 1, 1, 30, tzinfo=ZoneInfo(TZ)
    )


def test_run_scheduler_catches_up_then_stops(monkeypatch: pytest.MonkeyPatch) -> None:
    run_pending = MagicMock()
    monkeypatch.setattr(daily_analysis, "run_pending", run_pending)
    stop = threading.Event()
    stop.set()

    daily_analysis.run_scheduler(MagicMock(), TZ, time(0, 0), stop)

    run_pending.assert_called_once()


def test_run_day_reports_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    crud = MagicMock()
    crud.compute_daily_results.side_effect = ValueError("several detection models")
    monkeypatch.setattr(daily_analysis, "daily_analysis_result_crud", crud)
    monkeypatch.setattr(daily_analysis, "Session", MagicMock())

    assert daily_analysis.run_day(MagicMock(), date(2026, 10, 1), TZ) is False


@pytest.mark.parametrize("recompute", [False, True])
def test_run_day_passes_configured_thresholds(
    monkeypatch: pytest.MonkeyPatch, recompute: bool
) -> None:
    crud = MagicMock()
    crud.compute_daily_results.return_value = []
    monkeypatch.setattr(daily_analysis, "daily_analysis_result_crud", crud)
    monkeypatch.setattr(daily_analysis, "Session", MagicMock())
    monkeypatch.setattr(
        daily_analysis,
        "settings",
        Settings(det_conf_threshold=0.4, clas_conf_threshold=0.6),
    )

    assert daily_analysis.run_day(
        MagicMock(), date(2026, 10, 1), TZ, recompute=recompute
    )

    kwargs = crud.compute_daily_results.call_args.kwargs
    assert kwargs == {
        "recompute": recompute,
        "det_threshold": 0.4,
        "clas_threshold": 0.6,
    }


@pytest.fixture
def cli(monkeypatch: pytest.MonkeyPatch) -> tuple[MagicMock, MagicMock]:
    monkeypatch.setattr(
        daily_analysis, "settings", Settings(database_url="postgresql://fake/db")
    )
    monkeypatch.setattr(daily_analysis, "create_engine", MagicMock())
    run_day = MagicMock(return_value=True)
    run_pending = MagicMock(return_value=True)
    monkeypatch.setattr(daily_analysis, "run_day", run_day)
    monkeypatch.setattr(daily_analysis, "run_pending", run_pending)
    return run_day, run_pending


def test_main_date_runs_one_day(cli) -> None:
    run_day, run_pending = cli

    assert daily_analysis.main(["--date", "2026-10-01", "--recompute"]) == 0

    run_day.assert_called_once()
    assert run_day.call_args.args[1:] == (date(2026, 10, 1), TZ)
    assert run_day.call_args.kwargs == {"recompute": True}
    run_pending.assert_not_called()


@pytest.mark.parametrize(("succeeded", "exit_code"), [(True, 0), (False, 1)])
def test_main_recompute_all(
    cli, monkeypatch: pytest.MonkeyPatch, succeeded: bool, exit_code: int
) -> None:
    run_day, run_pending = cli
    run_recompute_all = MagicMock(return_value=succeeded)
    monkeypatch.setattr(daily_analysis, "run_recompute_all", run_recompute_all)

    assert daily_analysis.main(["--recompute-all"]) == exit_code

    run_recompute_all.assert_called_once()
    assert run_recompute_all.call_args.args[1:] == (TZ,)
    run_day.assert_not_called()
    run_pending.assert_not_called()


@pytest.mark.parametrize(
    "args",
    [
        ["--recompute-all", "--once"],
        ["--recompute-all", "--date", "2026-10-01"],
        ["--recompute-all", "--recompute"],
    ],
)
def test_main_recompute_all_excludes_other_modes(cli, args: list[str]) -> None:
    with pytest.raises(SystemExit):
        daily_analysis.main(args)


def test_run_recompute_all_passes_configured_thresholds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    crud = MagicMock()
    crud.recompute_all.return_value = MagicMock(recomputed=[], skipped=[])
    monkeypatch.setattr(daily_analysis, "daily_analysis_result_crud", crud)
    monkeypatch.setattr(daily_analysis, "Session", MagicMock())
    monkeypatch.setattr(
        daily_analysis,
        "settings",
        Settings(det_conf_threshold=0.4, clas_conf_threshold=0.6),
    )
    today = date(2026, 10, 2)

    assert daily_analysis.run_recompute_all(MagicMock(), TZ, today=today) is True

    assert crud.recompute_all.call_args.args[1:] == (today, TZ)
    assert crud.recompute_all.call_args.kwargs == {
        "det_threshold": 0.4,
        "clas_threshold": 0.6,
    }


def test_run_recompute_all_logs_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    crud = MagicMock()
    crud.recompute_all.side_effect = ValueError("database is gone")
    monkeypatch.setattr(daily_analysis, "daily_analysis_result_crud", crud)
    monkeypatch.setattr(daily_analysis, "Session", MagicMock())

    assert daily_analysis.run_recompute_all(MagicMock(), TZ) is False


def test_main_once_runs_pending_days(cli) -> None:
    run_day, run_pending = cli
    run_pending.return_value = False

    assert daily_analysis.main(["--once"]) == 1

    run_pending.assert_called_once()
    run_day.assert_not_called()


def test_main_recompute_requires_date(cli) -> None:
    with pytest.raises(SystemExit):
        daily_analysis.main(["--once", "--recompute"])


def test_main_requires_database_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(daily_analysis, "settings", Settings(database_url=""))

    assert daily_analysis.main(["--once"]) == 1


# --- integration test against a real PostGIS database ---


@pytest.fixture(scope="module")
def engine():
    testcontainers = pytest.importorskip("testcontainers.community.postgres")
    try:
        container = testcontainers.PostgresContainer(
            "postgis/postgis:17-3.5", driver="psycopg"
        )
        container.start()
    except Exception as exc:
        pytest.skip(f"Docker not available: {exc}")

    engine = create_engine(container.get_connection_url())
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()
    container.stop()


def _add_image(session: Session, captured_at: datetime, path: str) -> None:
    camera = session.scalar(select(Camera))
    rat = session.scalar(select(Taxonomy))
    det_model, clas_model = session.scalars(select(MLModel).order_by(MLModel.id))
    image = ImageCapture(
        camera=camera,
        image_path=path,
        location="SRID=4326;POINT(80.6 7.3)",
        captured_at=captured_at,
    )
    detection = ObjectDetection(
        image_capture=image,
        ml_model=det_model,
        confidence=0.9,
        bbox={},
        detected_class="animal",
    )
    session.add(
        SpeciesClassification(
            object_detection=detection,
            taxonomy=rat,
            ml_model=clas_model,
            confidence=0.8,
        )
    )


def test_run_pending_aggregates_past_days_once(engine) -> None:
    yesterday, today = date(2026, 10, 1), date(2026, 10, 2)
    with Session(engine) as session, session.begin():
        session.add_all(
            [
                Camera(name="CAM01", location="SRID=4326;POINT(80.6 7.3)"),
                Taxonomy(genus="Rattus", species="rattus"),
                MLModel(name="megadetector", task="detection"),
                MLModel(name="rodent-classifier", task="classification"),
            ]
        )
    with Session(engine) as session, session.begin():
        _add_image(session, _local(yesterday, 10), "CAM01_a.jpg")
        _add_image(session, _local(today, 10), "CAM01_b.jpg")

    assert daily_analysis.run_pending(engine, TZ, today=today) is True
    # nothing left to aggregate before today
    assert daily_analysis.run_pending(engine, TZ, today=today) is True

    with Session(engine) as session:
        rows = session.scalars(select(DailyAnalysisResult)).all()
        assert [(row.start_time, row.taxonomy_count) for row in rows] == [
            (_local(yesterday, 0), 1)
        ]
        tobe_deleted = dict(
            session.execute(
                select(ImageCapture.image_path, ImageCapture.tobe_deleted)
            ).all()
        )
        assert tobe_deleted == {"CAM01_a.jpg": True, "CAM01_b.jpg": False}


def _seed(engine) -> None:
    with Session(engine) as session, session.begin():
        session.execute(
            text(
                "TRUNCATE camera, taxonomy, ml_model, image_capture, "
                "daily_analysis_result RESTART IDENTITY CASCADE"
            )
        )
        session.add_all(
            [
                Camera(name="CAM01", location="SRID=4326;POINT(80.6 7.3)"),
                Taxonomy(genus="Rattus", species="rattus"),
                MLModel(name="megadetector", task="detection"),
                MLModel(name="rodent-classifier", task="classification"),
            ]
        )


def _use_thresholds(monkeypatch: pytest.MonkeyPatch, clas_threshold: float) -> None:
    monkeypatch.setattr(
        daily_analysis, "settings", Settings(clas_conf_threshold=clas_threshold)
    )


def _rows(engine) -> list[tuple[int, float]]:
    with Session(engine) as session:
        return [
            (row.taxonomy_count, row.clas_threshold)
            for row in session.scalars(select(DailyAnalysisResult))
        ]


def test_run_recompute_all_applies_new_thresholds(
    engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    yesterday, today = date(2026, 10, 1), date(2026, 10, 2)
    _seed(engine)
    with Session(engine) as session, session.begin():
        _add_image(session, _local(yesterday, 10), "CAM01_a.jpg")  # confidence 0.8

    _use_thresholds(monkeypatch, 0.5)
    assert daily_analysis.run_pending(engine, TZ, today=today) is True
    assert _rows(engine) == [(1, 0.5)]

    # a higher threshold filters the result out, a lower one brings it back
    _use_thresholds(monkeypatch, 0.9)
    assert daily_analysis.run_recompute_all(engine, TZ, today=today) is True
    assert _rows(engine) == []

    _use_thresholds(monkeypatch, 0.3)
    assert daily_analysis.run_recompute_all(engine, TZ, today=today) is True
    assert _rows(engine) == [(1, 0.3)]


def test_run_recompute_all_keeps_results_of_deleted_images(
    engine, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    yesterday, today = date(2026, 10, 1), date(2026, 10, 2)
    _seed(engine)
    with Session(engine) as session, session.begin():
        _add_image(session, _local(yesterday, 10), "CAM01_a.jpg")
        _add_image(session, _local(yesterday, 11), "CAM01_b.jpg")

    _use_thresholds(monkeypatch, 0.5)
    assert daily_analysis.run_pending(engine, TZ, today=today) is True
    with Session(engine) as session, session.begin():
        session.delete(
            session.scalar(
                select(ImageCapture).where(ImageCapture.image_path == "CAM01_a.jpg")
            )
        )

    _use_thresholds(monkeypatch, 0.3)
    # a skipped camera and day is not a failure
    assert daily_analysis.run_recompute_all(engine, TZ, today=today) is True

    assert _rows(engine) == [(2, 0.5)]
    assert "some of its images were deleted" in caplog.text
