"""GARCH(1,1)-t baseline on close-to-close returns (PROJECT_CONTEXT.md section 6.1).

Per event:
  1. Fit GARCH(1,1), constant mean, Student-t errors on ret_cc_pct over [train.start, cutoff].
     Freeze the parameters.
  2. With the parameters fixed, filter the conditional variance through every later day
     and forecast 1..22 steps ahead from each origin t (analytic). The forecast for horizon h
     is the mean of steps 1..h, converted from %^2 to decimal variance.
  3. Scale mismatch: GARCH models close-to-close variance (overnight gap included); RV covers
     09:30-16:00 only. c = mean(RV) / mean(GARCH conditional variance) on the training sample.
     forecast_var = c * GARCH variance (comparable to RV); forecast_var_naive = unscaled GARCH variance.
     Early-warning lift is unaffected by c.
  4. Validation metrics come from a separate train-only fit, forecasting the val block with fixed parameters.

Usage:
    uv run python -m volforecast.models.garch.run_all
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from arch import arch_model

from volforecast import adapters
from volforecast.models.baseline_config import load_baseline_config
from volforecast.models.common import append_run_log, qlike, write_json
from volforecast.models.schema import FORECAST_SCHEMA, forecast_path, validate_forecasts, write_forecasts
from volforecast.models.xgb.data import build_frames, target_dates

logger = logging.getLogger("volforecast.models.garch.run_all")
MAX_H = 22


def _model(returns: pd.Series, spec: dict):
    return arch_model(returns, mean=spec["mean"], vol=spec["vol"], p=spec["p"], o=spec["o"],
                      q=spec["q"], dist=spec["dist"], rescale=False)


def fit(returns: pd.Series, spec: dict):
    res = _model(returns, spec).fit(disp="off")
    if res.convergence_flag != 0:
        raise RuntimeError(f"GARCH did not converge (flag {res.convergence_flag})")
    return res


def mean_variance_forecasts(returns: pd.Series, params: pd.Series, spec: dict, start, horizons) -> pd.DataFrame:
    """Decimal-variance forecasts of mean variance over t+1..t+h, for every origin t >= start."""
    fixed = _model(returns, spec).fix(params)
    var = fixed.forecast(horizon=MAX_H, start=pd.Timestamp(start), reindex=False, method="analytic").variance
    cum = var.to_numpy().cumsum(axis=1)
    return pd.DataFrame({h: cum[:, h - 1] / h / 1e4 for h in horizons}, index=pd.DatetimeIndex(var.index))


def run_event(event: str, features_df: pd.DataFrame, split: dict, cfg: dict) -> list[dict]:
    t0 = time.perf_counter()
    spec, horizons = cfg["spec"], cfg["horizons"]
    df = features_df.set_index("date")
    returns = df["ret_cc_pct"].dropna()
    train_start, cutoff = pd.Timestamp(split["train"]["start"]), pd.Timestamp(split["val"]["end"])
    val_start = pd.Timestamp(split["val"]["start"])

    res = fit(returns.loc[train_start:cutoff], spec)
    in_sample_var = (res.conditional_volatility ** 2) / 1e4
    c = float(df.loc[in_sample_var.index, "rv"].mean() / in_sample_var.mean())

    # Validation metrics from a train-only fit (same idea as the other models' tuning block).
    res_train = fit(returns.loc[train_start:val_start - pd.Timedelta(days=1)], spec)
    c_train = float(df.loc[res_train.conditional_volatility.index, "rv"].mean()
                    / ((res_train.conditional_volatility ** 2) / 1e4).mean())
    val_fc = mean_variance_forecasts(returns.loc[:cutoff], res_train.params, spec, val_start, horizons)

    first_origin = None
    rows = []
    for h in horizons:
        frames = build_frames(features_df, split, ["ret_cc_pct"], h, cfg["train_tail_days"])
        first_origin = frames.tail.dates.min() if first_origin is None else first_origin
        fc = mean_variance_forecasts(returns, res.params, spec, frames.tail.dates.min(), [h])[h]
        origins = pd.concat([frames.tail.data, frames.oos.data])
        raw = fc.reindex(origins.index).to_numpy()
        dates = pd.DatetimeIndex(origins.index)
        t_start, t_end = target_dates(dates, frames.calendar, h)
        realized = origins[f"rv_h{h}"].to_numpy(dtype=float)
        name = f"garch_h{h}"
        out = pd.DataFrame({
            "date": dates, "target_start": t_start.to_numpy(), "target_end": t_end.to_numpy(), "h": h,
            "forecast_log": np.log(c * raw), "forecast_var": c * raw, "forecast_var_naive": raw,
            "forecast_var_seed_min": np.nan, "forecast_var_seed_max": np.nan,
            "realized_var": realized, "realized_log": np.log(realized), "target_complete": ~np.isnan(realized),
            "sample": origins["sample"].astype(str).to_numpy(),
            "model": name, "booster": "garch11t", "featureset": "returns", "event": event,
        }).astype(FORECAST_SCHEMA)
        validate_forecasts(out, split)
        write_forecasts(out, Path(cfg["paths"]["forecasts_dir"]) / event / f"{name}.parquet")

        vrows = frames.val.data
        v = c_train * val_fc[h].reindex(vrows.index).to_numpy()
        val_q = qlike(vrows[f"rv_h{h}"].to_numpy(), v)
        rows.append({"event": event, "model": name, "h": h, "val_qlike": val_q,
                     "val_mse_log": float(np.mean((vrows[f"y_h{h}"].to_numpy() - np.log(v)) ** 2))})

    params = {
        "spec": spec, "params": res.params.to_dict(), "std_err": res.std_err.to_dict(),
        "persistence": float(res.params["alpha[1]"] + res.params["beta[1]"]),
        "loglik": float(res.loglikelihood), "aic": float(res.aic), "bic": float(res.bic),
        "n_obs": int(res.nobs), "fit_range": [str(train_start.date()), str(cutoff.date())],
        "rv_scale_c": c, "train_only": {"params": res_train.params.to_dict(), "rv_scale_c": c_train},
        "val": rows,
    }
    write_json(Path(cfg["paths"]["models_dir"]) / event / "params.json", params)
    sec = time.perf_counter() - t0
    logger.info("%s: alpha+beta=%.4f nu=%.2f c=%.3f | val QLIKE %s", event, params["persistence"],
                res.params.get("nu", np.nan), c, {r["model"]: round(r["val_qlike"], 4) for r in rows})
    return [{**r, "persistence": params["persistence"], "rv_scale_c": c, "loglik": params["loglik"],
             "sec_total": sec} for r in rows]


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--event", nargs="+")
    args = p.parse_args(argv)
    cfg = load_baseline_config("garch")
    features_df = adapters.load_features(cfg["paths"]["features_path"])
    rows = []
    for event in args.event or cfg["events"]:
        rows += run_event(event, features_df, adapters.load_split(event, cfg["paths"]["splits_dir"]), cfg)
    logger.info("Run log: %s", append_run_log(rows, Path(cfg["paths"]["models_dir"]) / "run_log.csv"))


if __name__ == "__main__":
    main()
