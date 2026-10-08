"""Train / val / refit / forecast frames for one XGBoost cell.

Reads the frozen feature store and an event split file (make_splits.py).
"Trading day t+h" always means h rows ahead in the feature store, which is
exactly how build_features.py builds the y_h{h} targets, so the store's own
dates are the trading calendar here.

Segments (XGBoost_implementation.md section 4.1):
  train  : [train.start, val.start), purged so target_end < val.start
  val    : [val.start, val.end],     purged so target_end <= val.end
  refit  : [train.start, cutoff],    purged so target_end <= cutoff
  oos    : [pre_event.start, calm.end (or post_event.end)], not purged (target may be NaN)
  tail   : last `train_tail_days` trading days <= cutoff (in-sample fitted)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from volforecast.models.schema import oos_range, sample_labels


@dataclass
class Frame:
    """Rows of the feature store for one segment, indexed by forecast-origin date."""

    data: pd.DataFrame
    features: list[str]
    h: int

    @property
    def X(self) -> pd.DataFrame:
        return self.data[self.features]

    @property
    def y(self) -> pd.Series:
        return self.data[f"y_h{self.h}"]

    @property
    def rv(self) -> pd.Series:
        return self.data[f"rv_h{self.h}"]

    @property
    def dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.data.index)

    def __len__(self) -> int:
        return len(self.data)


@dataclass
class CellFrames:
    train: Frame
    val: Frame
    refit: Frame
    oos: Frame
    tail: Frame
    cutoff: pd.Timestamp
    calendar: pd.DatetimeIndex


def trading_calendar(features_df: pd.DataFrame) -> pd.DatetimeIndex:
    cal = pd.DatetimeIndex(pd.to_datetime(features_df["date"])).sort_values()
    if cal.has_duplicates:
        raise ValueError("Feature store has duplicate dates.")
    return cal


def target_dates(dates, calendar: pd.DatetimeIndex, h: int) -> tuple[pd.Series, pd.Series]:
    """(target_start, target_end) = (t+1, t+h) trading days; NaT past the end of the store."""
    dates = pd.DatetimeIndex(dates)
    pos = calendar.get_indexer(dates)
    if (pos < 0).any():
        raise ValueError("Some dates are not in the trading calendar.")

    def shifted(k: int) -> pd.Series:
        out = np.full(len(dates), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
        tgt = pos + k
        ok = tgt < len(calendar)
        out[ok] = calendar.values[tgt[ok]]
        return pd.Series(out, index=dates)

    return shifted(1), shifted(h)


def purge(df: pd.DataFrame, boundary_date, h: int, calendar: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    """Drop rows whose target window [t+1, t+h] ends after `boundary_date`.

    `df` has a `date` column. If `calendar` is omitted, df's own dates are used,
    which is correct as long as df is either the full store or a contiguous slice
    (rows whose t+h falls past the slice are dropped too).
    """
    cal = calendar if calendar is not None else trading_calendar(df)
    _, end = target_dates(df["date"], cal, h)
    keep = end.notna().to_numpy() & (end.to_numpy() <= np.datetime64(pd.Timestamp(boundary_date)))
    return df.loc[keep]


def _rows(df: pd.DataFrame, start, end=None, *, end_exclusive=None) -> pd.DataFrame:
    mask = df["date"] >= pd.Timestamp(start)
    if end is not None:
        mask &= df["date"] <= pd.Timestamp(end)
    if end_exclusive is not None:
        mask &= df["date"] < pd.Timestamp(end_exclusive)
    return df.loc[mask]


def _frame(df: pd.DataFrame, cols: list[str], h: int, require_target: bool) -> Frame:
    subset = cols + ([f"y_h{h}"] if require_target else [])
    clean = df.dropna(subset=subset).set_index("date")
    return Frame(clean, list(cols), h)


def build_frames(
    features_df: pd.DataFrame,
    split: dict,
    features: list[str],
    h: int,
    train_tail_days: int = 63,
    val_window: tuple[str, str] | None = None,
) -> CellFrames:
    """All frames for one (event, feature set, h).

    `val_window` overrides the split's validation block (robustness check:
    COVID-free validation for 2022). The refit block still runs to the cutoff.
    """
    df = features_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    missing = [c for c in features + [f"y_h{h}", f"rv_h{h}"] if c not in df.columns]
    if missing:
        raise KeyError(f"Feature store is missing columns {missing}.")

    cal = trading_calendar(df)
    cutoff = pd.Timestamp(split["val"]["end"])
    train_start = pd.Timestamp(split["train"]["start"])
    if val_window is None:
        val_start, val_end = pd.Timestamp(split["val"]["start"]), cutoff
    else:
        val_start, val_end = (pd.Timestamp(d) for d in val_window)
        if not (train_start < val_start < val_end <= cutoff):
            raise ValueError(f"val_window {val_window} must lie inside the pre-cutoff sample.")

    last_train_day = cal[cal < val_start][-1]
    train = purge(_rows(df, train_start, end_exclusive=val_start), last_train_day, h, cal)
    val = purge(_rows(df, val_start, val_end), val_end, h, cal)
    refit = purge(_rows(df, train_start, cutoff), cutoff, h, cal)

    if val_window is None:
        _check_against_split(train, val, split, h)

    pre_cutoff = _rows(df, train_start, cutoff).dropna(subset=features)
    tail = pre_cutoff.tail(train_tail_days).copy()
    tail["sample"] = "train_tail"

    oos = _rows(df, *oos_range(split)).dropna(subset=features).copy()
    oos["sample"] = sample_labels(oos["date"], split)

    return CellFrames(
        train=_frame(train, features, h, require_target=True),
        val=_frame(val, features, h, require_target=True),
        refit=_frame(refit, features, h, require_target=True),
        oos=_frame(oos, features, h, require_target=False),
        tail=_frame(tail, features, h, require_target=False),
        cutoff=cutoff,
        calendar=cal,
    )


def _check_against_split(train: pd.DataFrame, val: pd.DataFrame, split: dict, h: int) -> None:
    """The purge here must reproduce make_splits.py's end_by_horizon exactly."""
    for name, part in (("train", train), ("val", val)):
        expected = pd.Timestamp(split[name]["end_by_horizon"][str(h)])
        got = part["date"].max()
        if got != expected:
            raise ValueError(
                f"{name} purge for h={h} ends {got.date()}, split file says {expected.date()}. "
                "The split file may be stale; re-run volforecast.make_splits."
            )
