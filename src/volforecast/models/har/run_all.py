"""Log-HAR baseline (PROJECT_CONTEXT.md section 6.2): fit -> smear -> refit/freeze -> forecast.

Per cell (event x feature set x horizon):
  1. OLS of y_h{h} on the feature set over the train block.
  2. Validation residuals of that train-only fit -> Duan smearing factor S
     (the same rule as XGBoost, so the log models are back-transformed identically).
  3. Refit on all purged pre-cutoff rows (train + val), with Newey-West standard
     errors for the coefficient table, and freeze.
  4. Forecast train_tail + every out-of-sample day: forecast_var = exp(X b) * S.

Usage:
    uv run python -m volforecast.models.har.run_all
    uv run python -m volforecast.models.har.run_all --event covid --fs base --h 1
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from volforecast import adapters
from volforecast.models.baseline_config import load_baseline_config
from volforecast.models.common import append_run_log, qlike, read_json, smearing, write_json
from volforecast.models.schema import FORECAST_SCHEMA, forecast_path, validate_forecasts, write_forecasts
from volforecast.models.xgb.data import CellFrames, build_frames, target_dates

logger = logging.getLogger("volforecast.models.har.run_all")
NAIVE = {1: "log_rv_d", 5: "log_rv_w", 22: "log_rv_m"}


def model_name(fs: str, h: int) -> str:
    return f"har_{fs}_h{h}"


def _design(X: pd.DataFrame) -> pd.DataFrame:
    return sm.add_constant(X, has_constant="add")


def fit_cell(frames: CellFrames, h: int, hac_min_lags: int) -> dict:
    train_fit = sm.OLS(frames.train.y, _design(frames.train.X)).fit()
    val_pred = train_fit.predict(_design(frames.val.X)).to_numpy()
    smear = smearing(frames.val.y, val_pred)

    lags = max(hac_min_lags, h)
    refit = sm.OLS(frames.refit.y, _design(frames.refit.X)).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    rv_val = frames.val.rv.to_numpy()
    naive = np.exp(frames.val.data[NAIVE[h]].to_numpy())
    return {
        "coef": refit.params.to_dict(),
        "se_hac": refit.bse.to_dict(),
        "tstat_hac": refit.tvalues.to_dict(),
        "hac_lags": lags,
        "r2_refit": float(refit.rsquared),
        "coef_train_only": train_fit.params.to_dict(),
        "smear": smear,
        "val": {
            "mse_log": float(np.mean((frames.val.y.to_numpy() - val_pred) ** 2)),
            "mse_log_naive": float(np.mean((frames.val.y.to_numpy() - np.log(naive)) ** 2)),
            "qlike": qlike(rv_val, np.exp(val_pred) * smear["S"]),
            "qlike_naive": qlike(rv_val, naive),
        },
        "n_rows": {"train": len(frames.train), "val": len(frames.val), "refit": len(frames.refit)},
        "cutoff": frames.cutoff,
    }


def forecast_cell(params: dict, frames: CellFrames, event: str, fs: str, h: int) -> pd.DataFrame:
    rows = pd.concat([frames.tail.data, frames.oos.data])
    coef = pd.Series(params["coef"])
    X = _design(rows[frames.train.features])[coef.index]
    log_fc = X.to_numpy() @ coef.to_numpy()
    S = params["smear"]["S"]
    dates = pd.DatetimeIndex(rows.index)
    t_start, t_end = target_dates(dates, frames.calendar, h)
    realized = rows[f"rv_h{h}"].to_numpy(dtype=float)
    out = pd.DataFrame({
        "date": dates, "target_start": t_start.to_numpy(), "target_end": t_end.to_numpy(), "h": h,
        "forecast_log": log_fc, "forecast_var": np.exp(log_fc) * S, "forecast_var_naive": np.exp(log_fc),
        "forecast_var_seed_min": np.nan, "forecast_var_seed_max": np.nan,
        "realized_var": realized, "realized_log": np.log(realized), "target_complete": ~np.isnan(realized),
        "sample": rows["sample"].astype(str).to_numpy(),
        "model": model_name(fs, h), "booster": "ols", "featureset": fs, "event": event,
    })
    return out.astype(FORECAST_SCHEMA)


def run_cell(event: str, fs: str, h: int, features_df: pd.DataFrame, split: dict, cfg: dict,
             forecast_only: bool = False) -> dict:
    t0 = time.perf_counter()
    frames = build_frames(features_df, split, adapters.FEATURE_SETS[fs], h, cfg["train_tail_days"])
    model_dir = Path(cfg["paths"]["models_dir"]) / event / model_name(fs, h)
    if forecast_only:
        params = read_json(model_dir / "params.json")
    else:
        params = fit_cell(frames, h, cfg["hac_min_lags"])
        write_json(model_dir / "params.json", params)
    forecasts = forecast_cell(params, frames, event, fs, h)
    validate_forecasts(forecasts, split)
    write_forecasts(forecasts, forecast_path(cfg["paths"]["forecasts_dir"], _Named(event, model_name(fs, h))))
    v = params["val"]
    logger.info("%s/%s: val QLIKE %.4f (RW %.4f), S=%.3f, coef %s", event, model_name(fs, h), v["qlike"],
                v["qlike_naive"], params["smear"]["S"],
                {k: round(c, 3) for k, c in params["coef"].items()})
    return {"event": event, "model": model_name(fs, h), "featureset": fs, "h": h,
            "val_mse_log": v["mse_log"], "val_mse_log_rw": v["mse_log_naive"],
            "val_qlike": v["qlike"], "val_qlike_rw": v["qlike_naive"], "S": params["smear"]["S"],
            "r2_refit": params["r2_refit"], "sec_total": time.perf_counter() - t0}


class _Named:
    def __init__(self, event: str, name: str):
        self.event, self.name = event, name


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--event", nargs="+")
    p.add_argument("--fs", nargs="+")
    p.add_argument("--h", nargs="+", type=int)
    p.add_argument("--forecast-only", action="store_true")
    args = p.parse_args(argv)

    cfg = load_baseline_config("har")
    features_df = adapters.load_features(cfg["paths"]["features_path"])
    rows = []
    for event in args.event or cfg["events"]:
        split = adapters.load_split(event, cfg["paths"]["splits_dir"])
        for fs in args.fs or cfg["featuresets"]:
            for h in args.h or cfg["horizons"]:
                rows.append(run_cell(event, fs, h, features_df, split, cfg, args.forecast_only))
    if not args.forecast_only:
        logger.info("Run log: %s", append_run_log(rows, Path(cfg["paths"]["models_dir"]) / "run_log.csv"))


if __name__ == "__main__":
    main()
