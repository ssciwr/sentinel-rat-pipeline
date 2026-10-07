"""Tests for run_analysis helper."""

from unittest.mock import patch

import pytest

from sentinel_rat_pipeline.config import Settings
from sentinel_rat_pipeline.scripts.run_analysis import run_analysis_remote


@pytest.fixture(autouse=True)
def ml_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sentinel_rat_pipeline.scripts.run_analysis.settings",
        Settings(ml_pipeline_url="http://ml-service/"),
    )


def test_run_analysis_remote() -> None:
    results = [
        {"image_path": "a.jpg", "detections": []},
        {"image_path": "b.jpg", "detections": []},
    ]

    with patch("sentinel_rat_pipeline.scripts.run_analysis.httpx.Client") as MockClient:
        mock_post = MockClient.return_value.__enter__.return_value.post
        mock_post.return_value.raise_for_status.return_value = None
        mock_post.return_value.json.return_value = {"results": results}

        assert run_analysis_remote(["a.jpg", "b.jpg"]) == results

    mock_post.assert_called_once_with(
        "http://ml-service/api/v1/analyze", json={"image_paths": ["a.jpg", "b.jpg"]}
    )


def test_run_analysis_remote_retries_then_fails() -> None:
    with (
        patch("sentinel_rat_pipeline.scripts.run_analysis.httpx.Client") as MockClient,
        patch("sentinel_rat_pipeline.scripts.run_analysis.time.sleep"),
    ):
        mock_post = MockClient.return_value.__enter__.return_value.post
        mock_post.side_effect = ConnectionError("down")

        with pytest.raises(RuntimeError, match="after 3 attempts"):
            run_analysis_remote(["a.jpg"], retries=3)

    assert mock_post.call_count == 3
