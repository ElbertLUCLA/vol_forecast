"""Stage 1: download raw SPY bars from the Massive (formerly Polygon.io) Aggregates API.

Usage:
    uv run python -m volforecast.ingest_massive --start 2020-03-01 --end 2020-03-31  # test one month
    uv run python -m volforecast.ingest_massive                                       # full history
    uv run python -m volforecast.ingest_massive --force                               # refetch existing months
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import time
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from volforecast.config import PROJECT_ROOT, ensure_parent, load_config, resolve_path

logger = logging.getLogger("volforecast.ingest")

BAR_COLUMNS = {
    "t": "t_ms",
    "o": "open",
    "h": "high",
    "l": "low",
    "c": "close",
    "v": "volume",
    "vw": "vwap",
    "n": "n_trades",
}


class MassiveAPIError(RuntimeError):
    pass


def _api_key() -> str:
    key = os.environ.get("MASSIVE_API_KEY")
    if not key:
        raise MassiveAPIError(
            "MASSIVE_API_KEY not set. Copy .env.example to .env and fill in your key."
        )
    return key


def _log_request(log_path: Path, row: dict) -> None:
    ensure_parent(log_path)
    is_new = not log_path.exists()
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def fetch_aggs(
    ticker: str,
    multiplier: int,
    timespan: str,
    from_date: str,
    to_date: str,
    base_url: str,
    limit: int,
    adjusted: bool,
    max_retries: int,
    backoff_base: float,
    log_path: Path,
) -> pd.DataFrame:
    """Fetch all bars for [from_date, to_date], following next_url pagination."""
    api_key = _api_key()
    headers = {"Authorization": f"Bearer {api_key}"}
    url = f"{base_url}/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
    params = {"adjusted": str(adjusted).lower(), "sort": "asc", "limit": limit}

    all_results = []
    next_url = url
    next_params = params
    page = 0
    while next_url:
        attempt = 0
        while True:
            try:
                resp = requests.get(next_url, params=next_params, headers=headers, timeout=30)
            except requests.RequestException as exc:
                attempt += 1
                if attempt > max_retries:
                    raise MassiveAPIError(f"Network error fetching {next_url}: {exc}") from exc
                time.sleep(backoff_base * (2 ** (attempt - 1)))
                continue

            if resp.status_code == 429 or resp.status_code >= 500:
                attempt += 1
                if attempt > max_retries:
                    raise MassiveAPIError(
                        f"Giving up on {next_url} after {max_retries} retries "
                        f"(status {resp.status_code}): {resp.text[:500]}"
                    )
                time.sleep(backoff_base * (2 ** (attempt - 1)))
                continue
            break

        if resp.status_code != 200:
            raise MassiveAPIError(f"Request failed ({resp.status_code}): {resp.text[:500]}")

        payload = resp.json()
        results = payload.get("results", []) or []
        all_results.extend(results)
        page += 1

        _log_request(
            log_path,
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "ticker": ticker,
                "from_date": from_date,
                "to_date": to_date,
                "page": page,
                "request_id": payload.get("request_id", ""),
                "resultsCount": payload.get("resultsCount", len(results)),
                "status": payload.get("status", ""),
                "status_code": resp.status_code,
            },
        )

        next_url = payload.get("next_url")
        next_params = None  # next_url already has query params baked in, except auth

    if not all_results:
        return pd.DataFrame(columns=list(BAR_COLUMNS.values()))

    df = pd.DataFrame(all_results)
    df = df.rename(columns=BAR_COLUMNS)
    keep = [c for c in BAR_COLUMNS.values() if c in df.columns]
    df = df[keep].copy()
    df["t_utc"] = pd.to_datetime(df["t_ms"], unit="ms", utc=True)
    df = df.sort_values("t_utc").reset_index(drop=True)
    return df


def month_ranges(start: str, end: str) -> list[tuple[str, str, str]]:
    """Return (year_month, month_start, month_end) tuples covering [start, end]."""
    periods = pd.period_range(start=start, end=end, freq="M")
    out = []
    for p in periods:
        month_start = p.start_time.date()
        month_end = min(p.end_time.date(), date.fromisoformat(end))
        out.append((str(p), str(month_start), str(month_end)))
    return out


def ingest_5min(cfg: dict, start: str, end: str, force: bool) -> None:
    ticker = cfg["ticker"]
    api_cfg = cfg["massive_api"]
    raw_dir = resolve_path(cfg["paths"]["raw_5min_dir"])
    log_path = resolve_path(cfg["paths"]["ingest_log_path"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    months = month_ranges(start, end)
    for i, (ym, m_start, m_end) in enumerate(months):
        out_path = raw_dir / f"{ym}.parquet"
        is_last_two = i >= len(months) - 2
        if out_path.exists() and not force and not is_last_two:
            logger.info("Skipping %s (already downloaded)", ym)
            continue

        logger.info("Fetching 5-min bars for %s (%s to %s)", ym, m_start, m_end)
        df = fetch_aggs(
            ticker=ticker,
            multiplier=api_cfg["minute_multiplier"],
            timespan=api_cfg["minute_timespan"],
            from_date=m_start,
            to_date=m_end,
            base_url=api_cfg["base_url"],
            limit=api_cfg["limit"],
            adjusted=api_cfg["adjusted"],
            max_retries=api_cfg["max_retries"],
            backoff_base=api_cfg["backoff_base_seconds"],
            log_path=log_path,
        )
        if df.empty:
            logger.warning("No bars returned for %s", ym)
            continue
        df.to_parquet(out_path, index=False)
        logger.info("Saved %d bars to %s", len(df), out_path)


def ingest_daily(cfg: dict, start: str, end: str, force: bool) -> None:
    ticker = cfg["ticker"]
    api_cfg = cfg["massive_api"]
    out_path = resolve_path(cfg["paths"]["raw_daily_path"])
    log_path = resolve_path(cfg["paths"]["ingest_log_path"])

    if out_path.exists() and not force:
        logger.info("Skipping daily bars (already downloaded, use --force to refetch)")
        return

    logger.info("Fetching daily bars for %s to %s", start, end)
    df = fetch_aggs(
        ticker=ticker,
        multiplier=1,
        timespan=api_cfg["day_timespan"],
        from_date=start,
        to_date=end,
        base_url=api_cfg["base_url"],
        limit=api_cfg["limit"],
        adjusted=api_cfg["adjusted"],
        max_retries=api_cfg["max_retries"],
        backoff_base=api_cfg["backoff_base_seconds"],
        log_path=log_path,
    )
    ensure_parent(out_path)
    df.to_parquet(out_path, index=False)
    logger.info("Saved %d daily bars to %s", len(df), out_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Download raw SPY bars from the Massive API.")
    parser.add_argument("--start", default=cfg["sample_period"]["download_start"])
    parser.add_argument("--end", default=cfg["sample_period"]["download_end"])
    parser.add_argument("--force", action="store_true", help="Refetch months that already exist")
    parser.add_argument("--daily-only", action="store_true")
    parser.add_argument("--five-min-only", action="store_true")
    args = parser.parse_args()

    if not args.five_min_only:
        ingest_daily(cfg, args.start, args.end, args.force)
    if not args.daily_only:
        ingest_5min(cfg, args.start, args.end, args.force)


if __name__ == "__main__":
    main()
