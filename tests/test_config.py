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


def test_analysis_tz_must_be_known() -> None:
    with pytest.raises(ValidationError):
        Settings(analysis_tz="Mars/Olympus")


def test_max_batch_size_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        Settings(max_batch_size=0)
