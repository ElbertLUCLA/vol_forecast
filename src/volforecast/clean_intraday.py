"""Stage 3: build a clean regular-trading-hours 5-minute grid with intraday log returns.

For each trading day:
  - keep only bars whose start time is in [open, close - bar_minutes]
  - re-index to the full 5-minute grid, forward-filling missing closes
  - compute r1 = ln(c_open_bar / o_open_bar), r_j = ln(c_j / c_j-1) for j = 2..n_bars
  - flag n_bars_observed, n_missing, is_early_close, max_abs_5min_return
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from volforecast.config import ensure_parent, load_config, resolve_path

logger = logging.getLogger("volforecast.clean_intraday")

NY_TZ = "America/New_York"


def _load_raw_5min(raw_dir: Path) -> pd.DataFrame:
    files = sorted(raw_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No raw 5-min parquet files found in {raw_dir}. Run ingest_massive first.")
    frames = [pd.read_parquet(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset="t_utc").sort_values("t_utc").reset_index(drop=True)
    return df


def _session_grid(open_ts: pd.Timestamp, close_ts: pd.Timestamp, bar_minutes: int) -> pd.DatetimeIndex:
    """5-min bar start times in [open, close - bar_minutes]."""
    last_bar_start = close_ts - pd.Timedelta(minutes=bar_minutes)
    return pd.date_range(start=open_ts, end=last_bar_start, freq=f"{bar_minutes}min")


def clean_intraday(raw_5min: pd.DataFrame, calendar: pd.DataFrame, bar_minutes: int) -> pd.DataFrame:
    raw = raw_5min.copy()
    raw["t_ny"] = raw["t_utc"].dt.tz_convert(NY_TZ)
    raw["session_date"] = raw["t_ny"].dt.date

    cal = calendar.set_index("date")
    day_frames = []

    for day, row in cal.iterrows():
        open_ts = pd.Timestamp(row["market_open"])
        close_ts = pd.Timestamp(row["market_close"])
        if open_ts.tzinfo is None:
            open_ts = open_ts.tz_localize(NY_TZ)
            close_ts = close_ts.tz_localize(NY_TZ)
        else:
            open_ts = open_ts.tz_convert(NY_TZ)
            close_ts = close_ts.tz_convert(NY_TZ)

        grid = _session_grid(open_ts, close_ts, bar_minutes)
        n_grid = len(grid)

        day_bars = raw.loc[raw["session_date"] == day].copy()
        day_bars = day_bars.loc[
            (day_bars["t_ny"] >= open_ts) & (day_bars["t_ny"] <= close_ts - pd.Timedelta(minutes=bar_minutes))
        ]
        day_bars = day_bars.drop_duplicates(subset="t_ny").set_index("t_ny")

        reindexed = day_bars.reindex(grid)
        n_missing = int(reindexed["close"].isna().sum())
        n_observed = n_grid - n_missing

        if n_observed == 0:
            logger.warning("No bars at all for session %s; dropping.", day)
            continue

        # Forward-fill missing closes (zero return in that slot); open likewise.
        reindexed["close"] = reindexed["close"].ffill()
        if pd.isna(reindexed["close"].iloc[0]):
            logger.warning("Session %s has no bar at the open; dropping.", day)
            continue
        reindexed["open"] = reindexed["open"].fillna(reindexed["close"])

        closes = reindexed["close"].to_numpy(dtype=float)
        opens = reindexed["open"].to_numpy(dtype=float)

        r = np.empty(n_grid, dtype=float)
        r[0] = np.log(closes[0] / opens[0])
        if n_grid > 1:
            r[1:] = np.log(closes[1:] / closes[:-1])

        out = pd.DataFrame(
            {
                "t_ny": grid,
                "session_date": day,
                "close": closes,
                "ret_5min": r,
            }
        )
        out["n_bars_observed"] = n_observed
        out["n_missing"] = n_missing
        out["n_grid"] = n_grid
        out["is_early_close"] = bool(row["is_early_close"])
        day_frames.append(out)

    if not day_frames:
        raise RuntimeError("No sessions survived cleaning.")

    result = pd.concat(day_frames, ignore_index=True)
    return result


def day_quality_table(cleaned: pd.DataFrame) -> pd.DataFrame:
    g = cleaned.groupby("session_date", as_index=False).agg(
        n_bars_observed=("n_bars_observed", "first"),
        n_missing=("n_missing", "first"),
        n_grid=("n_grid", "first"),
        is_early_close=("is_early_close", "first"),
        max_abs_5min_return=("ret_5min", lambda s: s.abs().max()),
    )
    g["missing_fraction"] = g["n_missing"] / g["n_grid"]
    return g


def apply_exclusion_rule(cleaned: pd.DataFrame, quality: pd.DataFrame, max_missing_fraction: float) -> pd.DataFrame:
    bad_days = set(quality.loc[quality["missing_fraction"] > max_missing_fraction, "session_date"])
    if bad_days:
        logger.warning("Dropping %d day(s) exceeding %.0f%% missing bars: %s",
                        len(bad_days), max_missing_fraction * 100, sorted(bad_days)[:10])
    return cleaned.loc[~cleaned["session_date"].isin(bad_days)].reset_index(drop=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    raw_dir = resolve_path(cfg["paths"]["raw_5min_dir"])
    calendar_path = resolve_path(cfg["paths"]["calendar_path"])
    out_path = resolve_path(cfg["paths"]["interim_5min_rth_path"])
    bar_minutes = cfg["calendar"]["bar_minutes"]
    max_missing = cfg["calendar"]["max_missing_fraction"]

    if not calendar_path.exists():
        raise FileNotFoundError(f"{calendar_path} not found. Run volforecast.calendar first.")

    raw_5min = _load_raw_5min(raw_dir)
    calendar = pd.read_parquet(calendar_path)

    cleaned = clean_intraday(raw_5min, calendar, bar_minutes)
    quality = day_quality_table(cleaned)
    cleaned = apply_exclusion_rule(cleaned, quality, max_missing)

    ensure_parent(out_path)
    cleaned.to_parquet(out_path, index=False)

    n_days = cleaned["session_date"].nunique()
    logger.info("Saved %d bars across %d sessions to %s", len(cleaned), n_days, out_path)

    normal_days = quality.loc[~quality["is_early_close"]]
    pct_full = (normal_days["n_bars_observed"] == normal_days["n_grid"]).mean() if len(normal_days) else float("nan")
    logger.info("%.2f%% of normal sessions have a fully observed grid", pct_full * 100)


if __name__ == "__main__":
    main()
