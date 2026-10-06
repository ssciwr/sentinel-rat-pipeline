"""Application configuration for Sentinel Rat Pipeline."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(populate_by_name=True)

    watch_folder: str = Field(default="/data/images", alias="WATCH_FOLDER_DOCKER")
    database_url: str = Field(default="", alias="DATABASE_URL")
    ml_pipeline_url: str = Field(
        default="http://sentinel-rat-ml-pipeline:8000", alias="ML_PIPELINE_URL"
    )


settings = Settings()
