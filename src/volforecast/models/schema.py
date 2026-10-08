"""Shared forecast schema for every model (PROJECT_CONTEXT.md section 5.3).

One parquet per (event, model cell): data/processed/forecasts/{event}/{model}.parquet,
one row per forecast origin. The evaluation code reads only these files, so all
models are scored identically.

Kept free of model-library imports so any environment (CPU or ROCm) can use it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

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
# Out-of-sample segments in time order; "calm" is optional (a later addition to the split files).
OOS_SEGMENTS = ("pre_event", "event", "post_event", "calm")
SAMPLES = ("train_tail",) + OOS_SEGMENTS


def oos_segments(split: dict) -> list[str]:
    return [s for s in OOS_SEGMENTS if s in split]


def oos_range(split: dict) -> tuple[pd.Timestamp, pd.Timestamp]:
    segs = oos_segments(split)
    return pd.Timestamp(split[segs[0]]["start"]), pd.Timestamp(split[segs[-1]]["end"])


def sample_labels(dates, split: dict) -> np.ndarray:
    """Label each out-of-sample date with its segment (pre_event / event / post_event / calm)."""
    dates = pd.DatetimeIndex(dates)
    labels = np.full(len(dates), None, dtype=object)
    for seg in oos_segments(split):
        start, end = pd.Timestamp(split[seg]["start"]), pd.Timestamp(split[seg]["end"])
        labels[(dates >= start) & (dates <= end)] = seg
    return labels


def validate_forecasts(df: pd.DataFrame, split: dict) -> None:
    """Schema check (XGBoost sanity check 8, applied to every model). Raises AssertionError."""
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
    for seg in oos_segments(split):
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
