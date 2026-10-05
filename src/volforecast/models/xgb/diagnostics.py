"""Per-cell diagnostics -> diagnostics.json (sections 6 and 10).

- Extrapolation ceiling for trees: exact upper bound of each booster's output
  (base_score + sum of each tree's largest leaf) and an empirical max over
  training rows plus a grid of extreme feature values; days where realized
  variance exceeded the ceiling in the forecast period.
- Sanity check 5: Linear-XGB vs HAR-OLS coefficients (base, h=1, gblinear).
- Sanity check 6: tuned model beats the random walk on validation.
- Sanity check 7: smearing factor range and validation calibration.
- Seed spread for gbtree.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from volforecast.models.xgb.data import CellFrames
from volforecast.models.xgb.fit import FittedCell, Scaler, predict_log, train_booster

SMEAR_RANGE = (1.05, 1.5)
CALIBRATION_TOL = 0.10


def base_score(model: xgb.Booster) -> float:
    raw = json.loads(model.save_config())["learner"]["learner_model_param"]["base_score"]
    return float(str(raw).strip("[]").split(",")[0])


def tree_upper_bound(model: xgb.Booster) -> float:
    """Largest log forecast the booster can produce: base_score + sum_t max leaf_t."""
    trees = model.trees_to_dataframe()
    leaves = trees[trees["Feature"] == "Leaf"]
    return base_score(model) + float(leaves.groupby("Tree")["Gain"].max().sum())


def _extreme_grid(X: pd.DataFrame, max_rows: int = 5000) -> pd.DataFrame:
    """Each feature at {min, max, max + 3 sd} of the training data, all combinations."""
    levels = [[X[c].min(), X[c].max(), X[c].max() + 3 * X[c].std()] for c in X.columns]
    combos = list(itertools.islice(itertools.product(*levels), max_rows))
    return pd.DataFrame(combos, columns=X.columns)


def ceiling(fitted: FittedCell, frames: CellFrames) -> dict:
    train_max = float(frames.refit.y.max())
    out = {"max_train_target_log": train_max, "max_train_target_var": float(np.exp(train_max))}
    if fitted.cell.booster != "gbtree":
        out.update(bound_log=None, note="gblinear extrapolates linearly; no ceiling")
        return out

    per_seed = [tree_upper_bound(m) for m in fitted.models]
    # The seed average can never exceed the average of the per-seed bounds.
    bound = float(np.mean(per_seed))
    probe = pd.concat([frames.refit.X, _extreme_grid(frames.refit.X)], ignore_index=True)
    empirical = float(predict_log(fitted.models, probe, fitted.scaler).mean(axis=0).max())
    S = fitted.smear["S"]
    out.update(
        bound_log=bound,
        bound_log_per_seed=per_seed,
        empirical_max_log=empirical,
        ceiling_var=float(np.exp(bound) * S),
        bound_below_train_max=bool(bound < train_max),
    )
    return out


def ceiling_exceedances(forecasts: pd.DataFrame, ceiling_var: float | None) -> dict:
    if ceiling_var is None:
        return {}
    res = {}
    for seg in ("pre_event", "event", "post_event"):
        part = forecasts[(forecasts["sample"] == seg) & forecasts["target_complete"]]
        res[seg] = int((part["realized_var"] > ceiling_var).sum())
    return {"days_realized_above_ceiling": res}


def beats_naive(fitted_dir: Path) -> dict:
    """Sanity check 6 on the train-only model's validation predictions."""
    val = pd.read_parquet(Path(fitted_dir) / "val_predictions.parquet")
    mse_model = float(np.mean((val["y"] - val["pred_log"]) ** 2))
    mse_naive = float(np.mean((val["y"] - val["naive_log"]) ** 2))
    return {"val_mse_log": mse_model, "val_mse_log_naive": mse_naive, "pass": mse_model < mse_naive}


def smearing_sanity(fitted: FittedCell, fitted_dir: Path) -> dict:
    """Sanity check 7: S in range and smeared validation forecasts calibrated on average."""
    val = pd.read_parquet(Path(fitted_dir) / "val_predictions.parquet")
    S = fitted.smear["S"]
    mean_fc = float((np.exp(val["pred_log"]) * S).mean())
    mean_rv = float(val["rv"].mean())
    rel = mean_fc / mean_rv - 1.0
    return {
        "S": S,
        "S_in_range": SMEAR_RANGE[0] <= S <= SMEAR_RANGE[1],
        "val_mean_forecast_var": mean_fc,
        "val_mean_realized_var": mean_rv,
        "val_rel_bias": rel,
        "pass": SMEAR_RANGE[0] <= S <= SMEAR_RANGE[1] and abs(rel) <= CALIBRATION_TOL,
    }


def gblinear_coefficients(model: xgb.Booster, features: list[str]) -> pd.Series:
    """Weights on standardized features; the last entry of the weight vector is the bias."""
    weights = json.loads(model.save_raw("json").decode())["learner"]["gradient_booster"]["model"]["weights"]
    return pd.Series(weights[: len(features)], index=features, dtype=float)


def ols_coefficients(X: pd.DataFrame, y: pd.Series, scaler: Scaler) -> pd.Series:
    Xs = scaler.transform(X)
    design = np.column_stack([np.ones(len(Xs)), Xs.to_numpy(dtype=float)])
    beta, *_ = np.linalg.lstsq(design, y.to_numpy(dtype=float), rcond=None)
    return pd.Series(beta[1:], index=X.columns, dtype=float)


def linear_vs_har(frames: CellFrames, cfg: dict) -> dict:
    """Sanity check 5: weakly penalized gblinear should land near HAR-OLS (standardized units)."""
    scaler = Scaler.fit(frames.train.X)
    hp = {"learning_rate": 0.3, "reg_alpha": 0.0, "reg_lambda": 0.0, "n_estimators": 60}
    model = train_booster("gblinear", hp, frames.train, cfg, seed=0, scaler=scaler)
    xgb_coef = gblinear_coefficients(model, frames.train.features)
    ols_coef = ols_coefficients(frames.train.X, frames.train.y, scaler)
    max_abs_diff = float((xgb_coef - ols_coef).abs().max())
    return {
        "gblinear_coef": xgb_coef.to_dict(),
        "ols_coef": ols_coef.to_dict(),
        "max_abs_diff": max_abs_diff,
        "pass": max_abs_diff < 0.05,
    }


def seed_spread(forecasts: pd.DataFrame) -> dict:
    if forecasts["forecast_var_seed_min"].isna().all():
        return {}
    oos = forecasts[forecasts["sample"] != "train_tail"]
    rel = (oos["forecast_var_seed_max"] - oos["forecast_var_seed_min"]) / oos["forecast_var"]
    return {"oos_rel_seed_range_mean": float(rel.mean()), "oos_rel_seed_range_max": float(rel.max())}


def run_diagnostics(
    fitted: FittedCell,
    frames: CellFrames,
    forecasts: pd.DataFrame,
    cfg: dict,
    model_dir: Path,
    timings: dict | None = None,
) -> dict:
    ceil = ceiling(fitted, frames)
    diag = {
        "model": fitted.cell.name,
        "event": fitted.cell.event,
        "ceiling": ceil,
        **ceiling_exceedances(forecasts, ceil.get("ceiling_var")),
        "check6_beats_naive": beats_naive(model_dir),
        "check7_smearing": smearing_sanity(fitted, model_dir),
        "seed_spread": seed_spread(forecasts),
        "timings_sec": timings or {},
    }
    if fitted.cell.booster == "gblinear" and fitted.cell.featureset == "base" and fitted.cell.h == 1:
        diag["check5_linear_vs_har"] = linear_vs_har(frames, cfg)
    with open(Path(model_dir) / "diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(diag, f, indent=2, default=float)
    return diag

