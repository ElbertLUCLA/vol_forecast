"""Scaled daily series and label positions for one HARNet cell.

HARNet consumes the raw daily series (not precomputed HAR averages) and builds
its own 5/20-day aggregations, so the inputs are the whole feature-store series
and each segment is a set of *forecast-origin positions* in it. The label at
origin t is rv_h{h}(t) = mean(RV_{t+1..t+h}), the same direct target as the
other models (rather than the reference code's one-step-ahead labels).

Segment membership and horizon purging come from xgb.data.build_frames, so
HARNet uses exactly the same rows, and the same check against the split file's
end_by_horizon, as XGBoost.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from volforecast.models.xgb.data import build_frames, trading_calendar


@dataclass
class SeriesData:
    """Everything indexed by position in the feature store."""

    x: np.ndarray            # (n, C) scaled input channels
    y: np.ndarray            # (n,) scaled label rv_h{h}/scale, NaN where incomplete
    rv: np.ndarray           # (n,) unscaled label (variance units), NaN where incomplete
    dates: pd.DatetimeIndex  # (n,)
    calendar: pd.DatetimeIndex
    channels: list[str]
    scaler: "MinMaxScaler"
    positions: dict[str, np.ndarray]  # segment -> sorted origin positions
    samples: dict[int, str]           # oos/tail position -> sample label
    cutoff: pd.Timestamp


class MinMaxScaler:
    """Fixed map from the paper: x -> (x - min) / (max - min). Not fit to data."""

    def __init__(self, vmin: float, vmax: float):
        self.min, self.max = float(vmin), float(vmax)

    @property
    def scale(self) -> float:
        return self.max - self.min

    def transform(self, a):
        return (np.asarray(a, dtype=float) - self.min) / self.scale

    def inverse_transform(self, a):
        return np.asarray(a, dtype=float) * self.scale + self.min


def build_series(
    features_df: pd.DataFrame,
    split: dict,
    channels: list[str],
    h: int,
    max_lag: int,
    scaler: MinMaxScaler,
    train_tail_days: int = 63,
) -> SeriesData:
    df = features_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    if df[channels].isna().any().any():
        raise ValueError(f"NaN in HARNet input channels {channels}; the conv inputs must be contiguous.")

    frames = build_frames(df, split, channels, h, train_tail_days)
    pos_of = pd.Series(np.arange(len(df)), index=pd.DatetimeIndex(df["date"]))

    def positions(frame) -> np.ndarray:
        p = pos_of.loc[frame.dates].to_numpy()
        return p[p >= max_lag - 1]  # need a full receptive field of past inputs

    seg_pos = {name: positions(getattr(frames, name)) for name in ("train", "val", "refit", "oos", "tail")}
    samples = {int(p): s for p, s in zip(pos_of.loc[frames.oos.dates], frames.oos.data["sample"])}
    samples.update({int(p): "train_tail" for p in pos_of.loc[frames.tail.dates]})

    rv = df[f"rv_h{h}"].to_numpy(dtype=float)
    return SeriesData(
        x=scaler.transform(df[channels].to_numpy(dtype=float)),
        y=scaler.transform(rv),
        rv=rv,
        dates=pd.DatetimeIndex(df["date"]),
        calendar=trading_calendar(df),
        channels=list(channels),
        scaler=scaler,
        positions=seg_pos,
        samples=samples,
        cutoff=frames.cutoff,
    )


def window_starts(positions: np.ndarray, label_length: int) -> np.ndarray:
    """Origins p such that p, p+1, ..., p+L-1 are all allowed label positions."""
    allowed = set(positions.tolist())
    return np.array([p for p in positions if all(p + k in allowed for k in range(label_length))], dtype=np.int64)
