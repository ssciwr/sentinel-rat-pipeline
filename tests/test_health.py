from pathlib import Path

import pytest

from sentinel_rat_pipeline import health
from sentinel_rat_pipeline.config import Settings


def test_report_healthy_without_health_file_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(health, "settings", Settings())
    monkeypatch.chdir(tmp_path)

    health.report_healthy(True)

    assert list(tmp_path.iterdir()) == []


def test_report_healthy_creates_and_removes_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    health_file = tmp_path / "healthy"
    monkeypatch.setattr(health, "settings", Settings(health_file=health_file))

    health.report_healthy(True)
    assert health_file.exists()
    health.report_healthy(True)  # already there
    assert health_file.exists()
    health.report_healthy(False)
    assert not health_file.exists()
    health.report_healthy(False)  # already gone
    assert not health_file.exists()


def test_report_healthy_warns_once_if_the_file_cannot_be_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    health_file = tmp_path / "missing-folder" / "healthy"
    monkeypatch.setattr(health, "settings", Settings(health_file=health_file))
    monkeypatch.setattr(health, "_warned", False)

    health.report_healthy(True)
    health.report_healthy(True)

    warnings = [r for r in caplog.records if "health file" in r.getMessage()]
    assert len(warnings) == 1
