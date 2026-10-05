"""Out-of-sample forecasts with frozen parameters (section 5) in the shared schema (section 8).

This module only loads and predicts; it never trains. Each day t in
[pre_event.start, post_event.end] gets a forecast of mean RV over [t+1, t+h]
from features known at the close of t. The last `train_tail_days` pre-cutoff
days are added as in-sample fitted values (sample="train_tail") for the
evaluation script's early-warning lift.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from volforecast.models.xgb.data import CellFrames, target_dates
from volforecast.models.xgb.fit import FittedCell, predict_log

FORECAST_SCHEMA: dict[str, str] = {
    "date": "datetime64[ns]",
    "target_start": "datetime64[ns]",
    "target_end": "datetime64[ns]",
    "h": "int64",
    "forecast_log": "float64",
    "forecast_var": "float64",
    "forecast_var_naive": "float64",
    "forecast_var_seed_min": "float64",
    "forecast_var_seed_max": "float64",
    "realized_var": "float64",
    "realized_log": "float64",
    "target_complete": "bool",
    "sample": "str",
    "model": "str",
    "booster": "str",
    "featureset": "str",
    "event": "str",
}
SAMPLES = ("train_tail", "pre_event", "event", "post_event")


def forecast_cell(fitted: FittedCell, frames: CellFrames) -> pd.DataFrame:
    rows = pd.concat([frames.tail.data, frames.oos.data])
    X = rows[fitted.features]
    preds = predict_log(fitted.models, X, fitted.scaler)  # (n_seeds, n_rows)
    S = fitted.smear["S"]
    forecast_log = preds.mean(axis=0)

    if fitted.cell.booster == "gbtree":
        seed_vars = np.exp(preds) * S
        seed_min, seed_max = seed_vars.min(axis=0), seed_vars.max(axis=0)
    else:
        seed_min = seed_max = np.full(len(rows), np.nan)

    h = fitted.cell.h
    dates = pd.DatetimeIndex(rows.index)
    t_start, t_end = target_dates(dates, frames.calendar, h)
    realized = rows[f"rv_h{h}"].to_numpy(dtype=float)

    out = pd.DataFrame({
        "date": dates,
        "target_start": t_start.to_numpy(),
        "target_end": t_end.to_numpy(),
        "h": h,
        "forecast_log": forecast_log,
        "forecast_var": np.exp(forecast_log) * S,
        "forecast_var_naive": np.exp(forecast_log),
        "forecast_var_seed_min": seed_min,
        "forecast_var_seed_max": seed_max,
        "realized_var": realized,
        "realized_log": np.log(realized),
        "target_complete": ~np.isnan(realized),
        "sample": rows["sample"].astype(str).to_numpy(),
        "model": fitted.cell.name,
        "booster": fitted.cell.booster,
        "featureset": fitted.cell.featureset,
        "event": fitted.cell.event,
    })
    return out.astype(FORECAST_SCHEMA)


def validate_forecasts(df: pd.DataFrame, split: dict) -> None:
    """Schema check (sanity check 8). Raises AssertionError on any violation."""
    assert list(df.columns) == list(FORECAST_SCHEMA), f"columns {list(df.columns)}"
    for col, dtype in FORECAST_SCHEMA.items():
        assert str(df[col].dtype) == dtype, f"{col}: {df[col].dtype} != {dtype}"
    assert not df["date"].duplicated().any(), "duplicate forecast dates"
    assert df["date"].is_monotonic_increasing, "dates not sorted"
    assert set(df["sample"]) <= set(SAMPLES), f"unexpected samples {set(df['sample'])}"
    assert np.isfinite(df["forecast_log"]).all() and (df["forecast_var"] > 0).all()
    assert (df["target_complete"] == df["realized_var"].notna()).all()

    cutoff = pd.Timestamp(split["val"]["end"])
    tail = df[df["sample"] == "train_tail"]
    assert (tail["date"] <= cutoff).all(), "train_tail rows after the cutoff"
    for seg in ("pre_event", "event", "post_event"):
        part = df[df["sample"] == seg]
        start, end = pd.Timestamp(split[seg]["start"]), pd.Timestamp(split[seg]["end"])
        assert len(part) > 0, f"no {seg} rows"
        assert part["date"].between(start, end).all(), f"{seg} rows outside {start.date()}..{end.date()}"
    oos = df[df["sample"] != "train_tail"]
    assert (oos["date"] > cutoff).all(), "out-of-sample row on or before the cutoff"


def forecast_path(forecasts_dir: Path, cell) -> Path:
    return Path(forecasts_dir) / cell.event / f"{cell.name}.parquet"


def write_forecasts(df: pd.DataFrame, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path
