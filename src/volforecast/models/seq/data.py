"""Config loading and lookback windows of raw daily inputs for the sequence models.

The sample at forecast origin t is the window of daily inputs [t-L+1, t] (known at the
close of t) and the label is y_h{h}(t) = ln mean(RV_{t+1..t+h}), the same direct log
target as HAR/XGBoost. Segment membership and horizon purging come from
xgb.data.build_frames, so every model is trained and scored on the same rows.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from volforecast.config import PROJECT_ROOT, load_config, resolve_path
from volforecast.models.xgb.data import build_frames, trading_calendar

SEQ_CONFIG_PATH = PROJECT_ROOT / "configs" / "seq.yaml"


@dataclass(frozen=True)
class Cell:
    event: str
    arch: str
    featureset: str
    h: int

    @property
    def name(self) -> str:
        return f"{self.arch}_{self.featureset}_h{self.h}"


def load_seq_config(path: str | Path | None = None) -> dict:
    with open(path or SEQ_CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    pipe = load_config()["paths"]
    paths = cfg.setdefault("paths", {})
    for key in ("features_path", "splits_dir", "forecasts_dir"):
        paths.setdefault(key, pipe[key])
    for key, value in list(paths.items()):
        paths[key] = resolve_path(value)
    return cfg


@dataclass
class SeqData:
    x: np.ndarray                     # (n, C) standardized daily inputs (train-block mean/std)
    y: np.ndarray                     # (n,) log target y_h{h}, NaN where incomplete
    rv: np.ndarray                    # (n,) raw target rv_h{h}
    dates: pd.DatetimeIndex
    calendar: pd.DatetimeIndex
    channels: list[str]
    mean: np.ndarray
    std: np.ndarray
    positions: dict[str, np.ndarray]  # segment -> origin positions
    samples: dict[int, str]           # oos/tail position -> sample label
    cutoff: pd.Timestamp

    def windows(self, positions: np.ndarray, lookback: int) -> np.ndarray:
        """(len(positions), lookback, C) input windows ending at each origin."""
        idx = np.asarray(positions)[:, None] + np.arange(-lookback + 1, 1)
        return self.x[idx]


def build_seq_data(features_df: pd.DataFrame, split: dict, channels: list[str], h: int,
                   max_lookback: int, train_tail_days: int = 63) -> SeqData:
    df = features_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    if df[channels].isna().any().any():
        raise ValueError(f"NaN in sequence inputs {channels}")

    frames = build_frames(df, split, channels, h, train_tail_days)
    pos_of = pd.Series(np.arange(len(df)), index=pd.DatetimeIndex(df["date"]))

    def positions(frame) -> np.ndarray:
        p = pos_of.loc[frame.dates].to_numpy()
        return p[p >= max_lookback - 1]  # every lookback in the tuning grid sees the same rows

    seg = {name: positions(getattr(frames, name)) for name in ("train", "val", "refit", "oos", "tail")}
    samples = {int(p): s for p, s in zip(pos_of.loc[frames.oos.dates], frames.oos.data["sample"])}
    samples.update({int(p): "train_tail" for p in pos_of.loc[frames.tail.dates]})

    raw = df[channels].to_numpy(dtype=float)
    train_rows = raw[: seg["train"].max() + 1]  # inputs available up to the last train origin
    mean, std = train_rows.mean(axis=0), train_rows.std(axis=0)
    std[std == 0] = 1.0
    return SeqData(
        x=((raw - mean) / std).astype(np.float32),
        y=df[f"y_h{h}"].to_numpy(dtype=float),
        rv=df[f"rv_h{h}"].to_numpy(dtype=float),
        dates=pd.DatetimeIndex(df["date"]),
        calendar=trading_calendar(df),
        channels=list(channels),
        mean=mean,
        std=std,
        positions=seg,
        samples=samples,
        cutoff=frames.cutoff,
    )
