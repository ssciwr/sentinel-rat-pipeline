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
    # new images are collected for this many seconds, then analyzed as one batch
    batch_window_seconds: float = Field(default=60.0, alias="BATCH_WINDOW_SECONDS")
    # larger batches are split into chunks of at most this many images
    max_batch_size: int = Field(default=50, gt=0, alias="MAX_BATCH_SIZE")
    # analyze images already in the watch folder at startup
    process_backlog: bool = Field(default=True, alias="PROCESS_BACKLOG")
    # delete DB rows of removed images even if they were not marked tobe_deleted
    force_delete_image_in_db: bool = Field(
        default=False, alias="FORCE_DELETE_IMAGE_IN_DB"
    )


settings = Settings()
