"""Tests for persist_results helper."""

from unittest.mock import MagicMock, patch

import pytest

from sentinel_rat_pipeline.config import Settings
from sentinel_rat_pipeline.scripts.persist_results import persist_results


def test_persist_results(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(database_url="postgresql://fake:5432/db")
    monkeypatch.setattr(
        "sentinel_rat_pipeline.scripts.persist_results.settings", settings
    )

    mock_engine = MagicMock()
    mock_conn = MagicMock()

    with patch(
        "sentinel_rat_pipeline.scripts.persist_results.create_engine",
        return_value=mock_engine,
    ) as mock_create:
        mock_engine.begin.return_value.__enter__.return_value = mock_conn

        persist_results(
            {
                "image_path": "img1.jpg",
                "animals_detected": 1,
                "species": {"deer": 1},
                "confidence": 0.9,
            },
            detection_id="0001",
        )

    mock_create.assert_called_once_with("postgresql://fake:5432/db")
    mock_conn.execute.assert_called_once()
    call_kwargs = mock_conn.execute.call_args[0][0].bindparams
    assert call_kwargs["detection_id"].value == "0001"
    assert call_kwargs["species"].value == '{"deer": 1}'
