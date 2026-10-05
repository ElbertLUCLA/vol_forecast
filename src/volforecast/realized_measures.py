"""Stage 4: compute daily realized measures from the clean 5-minute RTH grid.

All measures are computed from the within-session 5-minute log returns (78 per
normal day, 42 on an early close). Units are decimal variance (not annualized,
not percent) unless noted.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from volforecast.config import ensure_parent, load_config, resolve_path

logger = logging.getLogger("volforecast.realized_measures")

# Andersen, Dobrev & Schaumburg (2012) medRV scaling constant.
_MEDRV_SCALE = np.pi / (6 - 4 * np.sqrt(3) + np.pi)


def _bpv(r: np.ndarray) -> float:
    if len(r) < 2:
        return np.nan
    return (np.pi / 2) * np.sum(np.abs(r[1:]) * np.abs(r[:-1]))


def _medrv(r: np.ndarray) -> float:
    n = len(r)
    if n < 3:
        return np.nan
    triplets = np.stack([r[:-2], r[1:-1], r[2:]], axis=1)
    med = np.median(np.abs(triplets), axis=1)
    return _MEDRV_SCALE * (n / (n - 2)) * np.sum(med ** 2)


def compute_daily_measures(cleaned_5min: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for day, g in cleaned_5min.groupby("session_date", sort=True):
        r = g["ret_5min"].to_numpy(dtype=float)
        rv = float(np.sum(r ** 2))
        rs_pos = float(np.sum(r[r > 0] ** 2))
        rs_neg = float(np.sum(r[r < 0] ** 2))
        rows.append(
            {
                "date": day,
                "rv": rv,
                "rs_pos": rs_pos,
                "rs_neg": rs_neg,
                "delta_j2": rs_pos - rs_neg,
                "bpv": _bpv(r),
                "medrv": _medrv(r),
                # r chains open->close, so the sum telescopes to ln(close/open).
                "oc_return": float(np.sum(r)),
                "rth_last_close": float(g["close"].iloc[-1]),
                "n_bars_observed": int(g["n_bars_observed"].iloc[0]),
                "n_missing": int(g["n_missing"].iloc[0]),
                "is_early_close": bool(g["is_early_close"].iloc[0]),
                "max_abs_5min_return": float(np.abs(r).max()),
            }
        )
    out = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    out["date"] = pd.to_datetime(out["date"])

    # Sanity identity: RS+ + RS- == RV (within floating point tolerance).
    resid = (out["rs_pos"] + out["rs_neg"] - out["rv"]).abs()
    bad = resid > 1e-10
    if bad.any():
        logger.warning("RS+ + RS- != RV on %d day(s); max abs diff = %.3e", bad.sum(), resid.max())

    return out


def attach_daily_bars(measures: pd.DataFrame, daily_raw: pd.DataFrame) -> pd.DataFrame:
    """Add close-to-close returns (for GARCH) and overnight returns from the official daily bars."""
    daily = daily_raw.copy()
    daily["date"] = pd.to_datetime(daily["t_utc"]).dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None)
    daily = daily[["date", "open", "close"]].rename(columns={"open": "daily_open", "close": "daily_close"})
    daily = daily.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)

    out = measures.merge(daily, on="date", how="left")
    out["ret_cc"] = np.log(out["daily_close"] / out["daily_close"].shift(1))
    out["ret_cc_pct"] = out["ret_cc"] * 100.0
    out["ret_overnight"] = np.log(out["daily_open"] / out["daily_close"].shift(1))

    close_diff = (out["rth_last_close"] - out["daily_close"]).abs() / out["daily_close"]
    out["rth_vs_daily_close_abs_pct_diff"] = close_diff * 100.0

    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    interim_path = resolve_path(cfg["paths"]["interim_5min_rth_path"])
    daily_raw_path = resolve_path(cfg["paths"]["raw_daily_path"])
    out_path = resolve_path(cfg["paths"]["daily_measures_path"])

    if not interim_path.exists():
        raise FileNotFoundError(f"{interim_path} not found. Run volforecast.clean_intraday first.")
    if not daily_raw_path.exists():
        raise FileNotFoundError(f"{daily_raw_path} not found. Run volforecast.ingest_massive first.")

    cleaned = pd.read_parquet(interim_path)
    daily_raw = pd.read_parquet(daily_raw_path)

    measures = compute_daily_measures(cleaned)
    measures = attach_daily_bars(measures, daily_raw)

    ensure_parent(out_path)
    measures.to_parquet(out_path, index=False)
    logger.info("Saved %d daily realized-measure rows to %s", len(measures), out_path)

    median_diff = measures["rth_vs_daily_close_abs_pct_diff"].median()
    logger.info("Median |RTH close - daily close| = %.4f%%", median_diff)


if __name__ == "__main__":
    main()
