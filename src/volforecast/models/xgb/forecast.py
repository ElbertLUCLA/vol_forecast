"""Out-of-sample forecasts with frozen parameters (section 5) in the shared schema (section 8).

This module only loads and predicts; it never trains. Each day t in
[pre_event.start, calm.end] (post_event.end if the split has no calm window)
gets a forecast of mean RV over [t+1, t+h] from features known at the close of t. The last `train_tail_days` pre-cutoff
days are added as in-sample fitted values (sample="train_tail") for the
evaluation script's early-warning lift.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from volforecast.models.schema import (  # noqa: F401  (re-exported for existing callers)
    FORECAST_SCHEMA,
    SAMPLES,
    forecast_path,
    validate_forecasts,
    write_forecasts,
)
from volforecast.models.xgb.data import CellFrames, target_dates
from volforecast.models.xgb.fit import FittedCell, predict_log


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
