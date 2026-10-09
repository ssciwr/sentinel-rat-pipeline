"""Daily analysis of the stored detections.

Aggregates the images of each past day into the ``daily_analysis_result`` table,
using ``compute_daily_results`` of the sentinel-rat-dashboard data model.

Run the scheduler, which aggregates every night at ANALYSIS_TIME (in ANALYSIS_TZ)
all past days that have images not aggregated yet::

    python -m sentinel_rat_pipeline.daily_analysis

Or run it once and exit, e.g. with ``docker run``::

    python -m sentinel_rat_pipeline.daily_analysis --once
    python -m sentinel_rat_pipeline.daily_analysis --date 2026-10-01 [--recompute]
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from dashboard.db.analysis_result import daily_analysis_result_crud
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session

from sentinel_rat_pipeline.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_day(engine: Engine, day: date, tz: str, recompute: bool = False) -> bool:
    """Aggregate one day. Return False if it failed."""
    try:
        with Session(engine) as session:
            rows = daily_analysis_result_crud.compute_daily_results(
                session, day, tz, recompute=recompute
            )
            logger.info("Daily analysis of %s: %s result row(s)", day, len(rows))
    except Exception:
        logger.exception("Daily analysis of %s failed", day)
        return False
    return True


def run_pending(engine: Engine, tz: str, today: date | None = None) -> bool:
    """Aggregate every day before today (in tz) that has images not aggregated
    yet, e.g. yesterday, days missed while not running, or days with late images.
    Return False if a day failed."""
    today = today or datetime.now(ZoneInfo(tz)).date()
    try:
        with Session(engine) as session:
            days = daily_analysis_result_crud.pending_days(session, before=today, tz=tz)
    except Exception:
        logger.exception("Could not look up the days to aggregate")
        return False

    if not days:
        logger.info("Daily analysis: no days to aggregate")
    # aggregate every day, even if an earlier one failed
    results = [run_day(engine, day, tz) for day in days]
    return all(results)


def next_run(now: datetime, at: time) -> datetime:
    """Return the first time `at` after `now`, in the timezone of `now`."""
    run = datetime.combine(now.date(), at, tzinfo=now.tzinfo)
    if run <= now:
        run = datetime.combine(now.date() + timedelta(days=1), at, tzinfo=now.tzinfo)
    return run


def run_scheduler(
    engine: Engine, tz: str, at: time, stop: threading.Event | None = None
) -> None:
    """Catch up on pending days, then aggregate every day at `at` (in tz)
    until `stop` is set."""
    stop = stop or threading.Event()
    zone = ZoneInfo(tz)

    run_pending(engine, tz)
    while True:
        now = datetime.now(zone)
        run = next_run(now, at)
        logger.info("Next daily analysis at %s", run.isoformat())
        if stop.wait((run - now).total_seconds()):
            logger.info("Daily analysis scheduler stopped")
            return
        run_pending(engine, tz)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m sentinel_rat_pipeline.daily_analysis",
        description=(
            "Aggregate the stored detections per day. Without options, run the "
            "scheduler, which aggregates every night at ANALYSIS_TIME."
        ),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--once",
        action="store_true",
        help="aggregate all past days with images not aggregated yet, then exit",
    )
    mode.add_argument(
        "--date",
        type=date.fromisoformat,
        help="aggregate one day (YYYY-MM-DD, in ANALYSIS_TZ), then exit",
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="with --date: rebuild the day from all its images",
    )
    args = parser.parse_args(argv)
    if args.recompute and args.date is None:
        parser.error("--recompute requires --date")

    if not settings.database_url:
        logger.error("DATABASE_URL is not set")
        return 1

    engine = create_engine(settings.database_url)
    tz = settings.analysis_tz
    logger.info("Daily analysis in timezone %s", tz)

    if args.date is not None:
        return 0 if run_day(engine, args.date, tz, recompute=args.recompute) else 1
    if args.once:
        return 0 if run_pending(engine, tz) else 1

    # stop cleanly on `docker stop`
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    run_scheduler(engine, tz, settings.analysis_time, stop)
    return 0


if __name__ == "__main__":
    sys.exit(main())
