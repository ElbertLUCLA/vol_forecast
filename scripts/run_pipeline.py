"""Run the full data pipeline end to end: ingest -> calendar -> clean -> measures -> features -> splits -> qc.

Usage:
    uv run python scripts/run_pipeline.py                 # full pipeline, full history
    uv run python scripts/run_pipeline.py --test-month 2020-03   # quick smoke test, one month only
    uv run python scripts/run_pipeline.py --skip-ingest    # reuse already-downloaded raw data
    uv run python scripts/run_pipeline.py --force-ingest    # re-download even if raw files exist
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from volforecast import (  # noqa: E402
    build_features,
    calendar,
    clean_intraday,
    ingest_massive,
    make_splits,
    qc_report,
    realized_measures,
)
from volforecast.config import load_config, resolve_path  # noqa: E402

logger = logging.getLogger("volforecast.run_pipeline")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Run the SPY RV data pipeline end to end.")
    parser.add_argument("--test-month", help="YYYY-MM: only download/process this one month (quick smoke test)")
    parser.add_argument("--skip-ingest", action="store_true", help="Reuse already-downloaded raw data")
    parser.add_argument("--force-ingest", action="store_true", help="Re-download months that already exist")
    parser.add_argument("--skip-qc", action="store_true", help="Skip the QC report (figures/tables)")
    args = parser.parse_args()

    cfg = load_config()

    if args.test_month:
        start = f"{args.test_month}-01"
        end = pd.Timestamp(start) + pd.offsets.MonthEnd(0)
        end = end.strftime("%Y-%m-%d")
    else:
        start = cfg["sample_period"]["download_start"]
        end = cfg["sample_period"]["download_end"]

    if not args.skip_ingest:
        logger.info("=== Stage 1: ingest ===")
        ingest_massive.ingest_daily(cfg, start, end, args.force_ingest)
        ingest_massive.ingest_5min(cfg, start, end, args.force_ingest)
    else:
        logger.info("=== Stage 1: ingest (skipped) ===")

    logger.info("=== Stage 2: calendar ===")
    cal_df = calendar.build_calendar(start, end, cfg["calendar"]["exchange"])
    cal_path = resolve_path(cfg["paths"]["calendar_path"])
    cal_path.parent.mkdir(parents=True, exist_ok=True)
    cal_df.to_parquet(cal_path, index=False)
    logger.info("Saved %d sessions to %s", len(cal_df), cal_path)

    logger.info("=== Stage 3: clean intraday ===")
    clean_intraday.main()

    logger.info("=== Stage 4: realized measures ===")
    realized_measures.main()

    logger.info("=== Stage 5: build features ===")
    build_features.main()

    logger.info("=== Stage 6: make splits ===")
    try:
        make_splits.main()
    except ValueError as exc:
        logger.warning("Skipping splits (likely too little data for a --test-month run): %s", exc)

    if not args.skip_qc:
        logger.info("=== Stage 7: QC report ===")
        qc_report.main()
    else:
        logger.info("=== Stage 7: QC report (skipped) ===")

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
