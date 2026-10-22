"""Tests for application settings."""

import pytest
from pydantic import ValidationError

from sentinel_rat_pipeline.config import Settings


def test_batch_defaults() -> None:
    settings = Settings()
    assert settings.batch_window_seconds == 60.0
    assert settings.max_batch_size == 50
    assert settings.process_backlog is True


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


def test_max_batch_size_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        Settings(max_batch_size=0)
