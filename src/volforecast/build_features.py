"""Stage 5: build the feature store and forecasting targets from daily realized measures.

Every row is indexed by trading day t. Features use information up to and
including the close of day t; targets start at t+1 (see plan.md section 5).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from volforecast.config import ensure_parent, load_config, resolve_path

logger = logging.getLogger("volforecast.build_features")


def _trailing_mean(s: pd.Series, window: int) -> pd.Series:
    """Mean of the current value and the (window-1) preceding ones."""
    return s.rolling(window=window, min_periods=window).mean()


def _forward_mean(s: pd.Series, h: int) -> pd.Series:
    """Mean of the h values strictly after the current one: mean(s[t+1..t+h])."""
    reversed_incl = s.iloc[::-1].rolling(window=h, min_periods=h).mean().iloc[::-1]
    return reversed_incl.shift(-1)


def _floor_and_log(s: pd.Series, floor_percentile: float, name: str) -> tuple[pd.Series, float]:
    positive = s[s > 0]
    eps = float(positive.quantile(floor_percentile)) if len(positive) else 1e-12
    eps = max(eps, 1e-12)
    n_floored = int((s <= 0).sum())
    if n_floored:
        logger.warning("Flooring %d non-positive %s value(s) at eps=%.3e", n_floored, name, eps)
    floored = s.clip(lower=eps)
    return np.log(floored), eps


def build_features(measures: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = measures.sort_values("date").reset_index(drop=True).copy()

    windows = cfg["features"]["base_windows"]
    w_week, w_month = windows["weekly"], windows["monthly"]
    horizons = cfg["features"]["horizons"]
    floor_pct = cfg["realized_measures"]["zero_rv_floor_percentile"]

    # --- Base features ---
    df["log_rv_d"], eps_rv = _floor_and_log(df["rv"], floor_pct, "rv")
    rv_w = _trailing_mean(df["rv"], w_week)
    rv_m = _trailing_mean(df["rv"], w_month)
    df["log_rv_w"], _ = _floor_and_log(rv_w, floor_pct, "rv_w")
    df["log_rv_m"], _ = _floor_and_log(rv_m, floor_pct, "rv_m")

    # --- Extended features ---
    df["log_rs_neg"], eps_rs_neg = _floor_and_log(df["rs_neg"], floor_pct, "rs_neg")
    df["log_rs_pos"], _ = _floor_and_log(df["rs_pos"], floor_pct, "rs_pos")
    rs_neg_w = _trailing_mean(df["rs_neg"], w_week)
    rs_neg_m = _trailing_mean(df["rs_neg"], w_month)
    df["log_rs_neg_w"], _ = _floor_and_log(rs_neg_w, floor_pct, "rs_neg_w")
    df["log_rs_neg_m"], _ = _floor_and_log(rs_neg_m, floor_pct, "rs_neg_m")

    # Patton & Sheppard (2015) log-safe signed-jump transform; undefined only if RV<=0.
    rv_safe = df["rv"].clip(lower=eps_rv)
    df["jump_pct"] = np.log(1.0 + df["delta_j2"] / rv_safe)
    df["sj_raw"] = df["delta_j2"]

    # --- Optional / robustness features ---
    df["ret_cc"] = df["ret_cc"]
    df["ret_overnight"] = df["ret_overnight"]
    df["mom"] = rv_w - rv_m
    df["log_medrv"], _ = _floor_and_log(df["medrv"], floor_pct, "medrv")

    # --- Original Multi-Transformer spec inputs (kept; that model was replaced by Mamba) ---
    df["ret_cc_roll_std_5"] = df["ret_cc"].rolling(window=w_week, min_periods=w_week).std()

    # --- Targets (direct multi-step forecasting) ---
    for h in horizons:
        rv_fwd_mean = _forward_mean(df["rv"], h)
        df[f"rv_h{h}"] = rv_fwd_mean
        log_fwd, _ = _floor_and_log(rv_fwd_mean.fillna(eps_rv), floor_pct, f"rv_h{h}")
        df[f"y_h{h}"] = log_fwd.where(rv_fwd_mean.notna())

    return df


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    measures_path = resolve_path(cfg["paths"]["daily_measures_path"])
    out_path = resolve_path(cfg["paths"]["features_path"])

    if not measures_path.exists():
        raise FileNotFoundError(f"{measures_path} not found. Run volforecast.realized_measures first.")

    measures = pd.read_parquet(measures_path)
    features = build_features(measures, cfg)

    modelling_start = pd.Timestamp(cfg["sample_period"]["modelling_start"])
    features = features.loc[features["date"] >= modelling_start].reset_index(drop=True)

    ensure_parent(out_path)
    features.to_parquet(out_path, index=False)
    logger.info("Saved %d rows x %d columns to %s", len(features), len(features.columns), out_path)


if __name__ == "__main__":
    main()
