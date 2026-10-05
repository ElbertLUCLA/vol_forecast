"""Stage 8: model-specific views onto the single feature store.

Every adapter reads data/processed/spy_features.parquet and a split dict
(as written by make_splits.py) and returns exactly what one model family
expects. Keeping one feature store with thin adapters is what makes "same
information set across models" true by construction (plan.md section 5/8).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from volforecast.config import resolve_path

BASE_FEATURES = ["log_rv_d", "log_rv_w", "log_rv_m"]
EXTENDED_FEATURES = BASE_FEATURES + [
    "log_rs_neg",
    "log_rs_pos",
    "log_rs_neg_w",
    "log_rs_neg_m",
    "jump_pct",
]
# HARNet eq. 29 extended baseline in logs, without RS+ (RS+ = RV - RS- makes it redundant).
# Shared by HAR and XGBoost (configs/xgb.yaml) so both models see identical inputs.
EXT_FEATURES = BASE_FEATURES + [
    "log_rs_neg",
    "log_rs_neg_w",
    "log_rs_neg_m",
    "jump_pct",
]
MT_ORIGINAL_FEATURES = ["ret_cc", "ret_cc_roll_std_5"]

FEATURE_SETS = {
    "base": BASE_FEATURES,
    "extended": EXTENDED_FEATURES,
    "ext": EXT_FEATURES,
    "mt_original": MT_ORIGINAL_FEATURES,
}


def load_features(features_path: str | Path | None = None) -> pd.DataFrame:
    if features_path is None:
        from volforecast.config import load_config

        features_path = resolve_path(load_config()["paths"]["features_path"])
    df = pd.read_parquet(features_path)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def load_split(event: str, splits_dir: str | Path | None = None) -> dict:
    if splits_dir is None:
        from volforecast.config import load_config

        splits_dir = resolve_path(load_config()["paths"]["splits_dir"])
    with open(Path(splits_dir) / f"{event}.json", "r", encoding="utf-8") as f:
        return json.load(f)


def _segment_bounds(split: dict, segment: str, h: int | None = None) -> tuple[pd.Timestamp, pd.Timestamp]:
    block = split[segment]
    start = pd.Timestamp(block["start"])
    if segment in ("train", "val") and h is not None:
        end = pd.Timestamp(block["end_by_horizon"][str(h)])
    else:
        end = pd.Timestamp(block["end"])
    return start, end


def _slice_dates(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    mask = (df["date"] >= start) & (df["date"] <= end)
    return df.loc[mask].reset_index(drop=True)


def garch_view(features_df: pd.DataFrame, split: dict) -> dict:
    """Training returns (for fitting) plus the out-of-sample dates to forecast.

    GARCH is fit once on returns up to the end of the validation block (i.e.
    the frozen cutoff in plan.md section 3), then forecasts variance forward
    with frozen parameters over the pre-event/event/post-event windows.
    """
    cutoff = pd.Timestamp(split["val"]["end"])
    train_start = pd.Timestamp(split["train"]["start"])
    train_df = _slice_dates(features_df, train_start, cutoff)

    oos_start = pd.Timestamp(split["pre_event"]["start"])
    oos_end = pd.Timestamp(split["post_event"]["end"])
    oos_df = _slice_dates(features_df, oos_start, oos_end)

    return {
        "train_returns": train_df.set_index("date")["ret_cc_pct"],
        "oos_dates": oos_df["date"].reset_index(drop=True),
        "cutoff": cutoff,
    }


def tabular_view(
    features_df: pd.DataFrame, split: dict, segment: str, featureset: str, h: int
) -> tuple[pd.DataFrame, pd.Series]:
    """X, y for HAR and XGBoost: tabular HAR-style features -> y_h{h} target.

    For segment in {"train", "val"}, the horizon-purged cutoff is used so no
    row's target window crosses the freeze date. For the OOS segments, the
    segment's own (unpurged) bounds are used since nothing there is training data.
    """
    cols = FEATURE_SETS[featureset]
    start, end = _segment_bounds(split, segment, h)
    sliced = _slice_dates(features_df, start, end)
    target_col = f"y_h{h}"
    valid = sliced.dropna(subset=cols + [target_col])
    X = valid[cols].reset_index(drop=True)
    y = valid[target_col].reset_index(drop=True)
    X.index = valid["date"].reset_index(drop=True)
    y.index = X.index
    return X, y


def harnet_view(features_df: pd.DataFrame, split: dict, segment: str, featureset: str) -> pd.DataFrame:
    """Raw daily series for HARNet. HARNet builds its own 5/20/80-day dilated-conv

    aggregations internally, so it needs the undifferenced daily series, not
    precomputed rolling averages (plan.md section 4/8).

    featureset="base"     -> single-channel RV series, column ["rv"]
    featureset="extended" -> 3-channel [rv, rs_neg, delta_j2] matching HARNet's
                              HARSVJ input convention (src/harnet/util.py get_MAN_data)
    """
    start = pd.Timestamp(split["train"]["start"])
    end = pd.Timestamp(split[segment]["end"] if segment != "train" else split["train"]["end"])
    sliced = _slice_dates(features_df, start, end).set_index("date")
    if featureset == "extended":
        return sliced[["rv", "rs_neg", "delta_j2"]]
    return sliced[["rv"]]


def sequence_view(
    features_df: pd.DataFrame, split: dict, segment: str, featureset: str, h: int, lags: int = 10
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """3-D array (samples, lags, n_features) -> y_h{h}, for the Multi-Transformer.

    Each sample at day t uses features from days [t-lags+1, t] to predict y_h{h}
    at t. Samples whose lag window would reach before the segment/global start,
    or whose target is NaN, are dropped.
    """
    cols = FEATURE_SETS[featureset]
    target_col = f"y_h{h}"

    seg_start, seg_end = _segment_bounds(split, segment, h)
    full = features_df.dropna(subset=cols).reset_index(drop=True)
    full_dates = full["date"]

    in_segment = full.index[(full_dates >= seg_start) & (full_dates <= seg_end)]
    in_segment = in_segment[in_segment >= lags - 1]

    X_list, y_list, date_list = [], [], []
    feat_matrix = full[cols].to_numpy(dtype=float)
    targets = full[target_col].to_numpy(dtype=float)

    for idx in in_segment:
        target = targets[idx]
        if np.isnan(target):
            continue
        window = feat_matrix[idx - lags + 1 : idx + 1, :]
        if window.shape[0] < lags or np.isnan(window).any():
            continue
        X_list.append(window)
        y_list.append(target)
        date_list.append(full_dates.iloc[idx])

    if not X_list:
        return (
            np.empty((0, lags, len(cols))),
            np.empty((0,)),
            pd.DatetimeIndex([]),
        )

    return np.stack(X_list), np.array(y_list), pd.DatetimeIndex(date_list)
