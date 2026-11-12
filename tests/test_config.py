"""Tests for application settings."""

from datetime import time

import pytest
from pydantic import ValidationError

from sentinel_rat_pipeline.config import Settings


def test_batch_defaults() -> None:
    settings = Settings()
    assert settings.batch_window_seconds == 60.0
    assert settings.max_batch_size == 50
    assert settings.process_backlog is True
    assert settings.force_delete_image_in_db is False
    assert settings.analysis_tz == "Asia/Colombo"
    assert settings.analysis_time == time(0, 0)
    assert settings.det_conf_threshold == 0.0
    assert settings.clas_conf_threshold == 0.0


def test_force_delete_image_in_db_defaults_to_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert Settings().force_delete_image_in_db is False
    monkeypatch.setenv("FORCE_DELETE_IMAGE_IN_DB", "true")
    assert Settings().force_delete_image_in_db is True


def test_batch_settings_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAX_BATCH_SIZE", "10")
    monkeypatch.setenv("PROCESS_BACKLOG", "false")
    settings = Settings()
    assert settings.max_batch_size == 10
    assert settings.process_backlog is False


def test_analysis_defaults() -> None:
    settings = Settings()
    assert settings.analysis_tz == "Asia/Colombo"
    assert settings.analysis_time == time(0, 0)


def test_analysis_settings_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANALYSIS_TZ", "Europe/Berlin")
    monkeypatch.setenv("ANALYSIS_TIME", "00:30")
    settings = Settings()
    assert settings.analysis_tz == "Europe/Berlin"
    assert settings.analysis_time == time(0, 30)


def test_thresholds_default_to_zero() -> None:
    settings = Settings()
    assert settings.det_conf_threshold == 0.0
    assert settings.clas_conf_threshold == 0.0


def test_thresholds_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DET_CONF_THRESHOLD", "0.4")
    monkeypatch.setenv("CLAS_CONF_THRESHOLD", "1")
    settings = Settings()
    assert settings.det_conf_threshold == 0.4
    assert settings.clas_conf_threshold == 1.0


@pytest.mark.parametrize("name", ["DET_CONF_THRESHOLD", "CLAS_CONF_THRESHOLD"])
@pytest.mark.parametrize("value", ["high", "1.5", "-0.1", "nan"])
def test_threshold_must_be_between_zero_and_one(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    monkeypatch.setenv(name, value)
    with pytest.raises(ValidationError):
        Settings()


def test_analysis_tz_must_be_known() -> None:
    with pytest.raises(ValidationError):
        Settings(analysis_tz="Mars/Olympus")


def test_max_batch_size_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        Settings(max_batch_size=0)
