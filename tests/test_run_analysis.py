"""Tests for run_analysis helper."""

from unittest.mock import patch

import pytest

from sentinel_rat_pipeline.config import Settings
from sentinel_rat_pipeline.scripts.run_analysis import run_analysis_remote


def test_run_analysis_remote(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sentinel_rat_pipeline.scripts.run_analysis.settings",
        Settings(ml_pipeline_url="http://ml-service/"),
    )

    with patch("sentinel_rat_pipeline.scripts.run_analysis.httpx.Client") as MockClient:
        mock_post = MockClient.return_value.__enter__.return_value.post
        mock_post.return_value.raise_for_status.return_value = None
        mock_post.return_value.json.return_value = {"species": {"deer": 1}}

        result = run_analysis_remote("/fake/path.jpg")

    assert result["species"] == {"deer": 1}
