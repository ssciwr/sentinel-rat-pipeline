"""Application configuration for Sentinel Rat Pipeline."""

from __future__ import annotations

from datetime import time
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import Field, field_validator
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
    # the daily analysis aggregates a day from midnight to midnight in this timezone
    analysis_tz: str = Field(default="Asia/Colombo", alias="ANALYSIS_TZ")
    # local time (in ANALYSIS_TZ) at which the scheduler runs the daily analysis
    analysis_time: time = Field(default=time(0, 0), alias="ANALYSIS_TIME")

    @field_validator("analysis_tz")
    @classmethod
    def _known_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(f"Unknown timezone: {value}") from exc
        return value


settings = Settings()
