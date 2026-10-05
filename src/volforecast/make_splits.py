"""Stage 6: write per-event split files (train/val/pre_event/event/post_event date ranges).

Each split file also records, per forecasting horizon h, a purged training
cutoff: the last date whose h-step-ahead target window does not cross the
freeze date. Training rows must use the purged cutoff for their horizon, not
the raw `train_end`, to avoid leaking event-window data into training
(see plan.md section 3 and section 6, "Train/test cutoffs per event").
"""
from __future__ import annotations

import json
import logging

import pandas as pd

from volforecast.config import ensure_parent, load_config, resolve_path

logger = logging.getLogger("volforecast.make_splits")

VAL_FRACTION = 0.10


def _purged_cutoff(trading_days: pd.Series, cutoff: pd.Timestamp, h: int) -> pd.Timestamp:
    days = trading_days[trading_days <= cutoff].sort_values().reset_index(drop=True)
    if len(days) <= h:
        raise ValueError(f"Not enough trading days before {cutoff.date()} to purge horizon {h}.")
    return days.iloc[-(h + 1)]


def build_split(event_name: str, event_cfg: dict, all_dates: pd.Series, modelling_start: pd.Timestamp,
                 horizons: list[int]) -> dict:
    train_end = pd.Timestamp(event_cfg["train_end"])
    pre_start, pre_end = (pd.Timestamp(d) for d in event_cfg["pre_event_window"])
    ev_start, ev_end = (pd.Timestamp(d) for d in event_cfg["event_window"])
    post_start, post_end = (pd.Timestamp(d) for d in event_cfg["post_event_window"])

    pre_cutoff_days = all_dates[(all_dates >= modelling_start) & (all_dates <= train_end)].sort_values()
    if pre_cutoff_days.empty:
        raise ValueError(f"No trading days found for event '{event_name}' before {train_end.date()}.")

    n = len(pre_cutoff_days)
    n_val = max(1, round(n * VAL_FRACTION))
    val_start = pre_cutoff_days.iloc[-n_val]
    train_start = modelling_start
    train_end_boundary = pre_cutoff_days.iloc[-n_val - 1] if n_val < n else pre_cutoff_days.iloc[0]

    train_end_by_horizon = {
        str(h): _purged_cutoff(all_dates, train_end_boundary, h).strftime("%Y-%m-%d") for h in horizons
    }
    val_end_by_horizon = {
        str(h): _purged_cutoff(all_dates, train_end, h).strftime("%Y-%m-%d") for h in horizons
    }

    return {
        "event": event_name,
        "peak": event_cfg["peak"],
        "trough": event_cfg["trough"],
        "modelling_start": modelling_start.strftime("%Y-%m-%d"),
        "train": {
            "start": train_start.strftime("%Y-%m-%d"),
            "end": train_end_boundary.strftime("%Y-%m-%d"),
            "end_by_horizon": train_end_by_horizon,
        },
        "val": {
            "start": val_start.strftime("%Y-%m-%d"),
            "end": train_end.strftime("%Y-%m-%d"),
            "end_by_horizon": val_end_by_horizon,
        },
        "pre_event": {"start": pre_start.strftime("%Y-%m-%d"), "end": pre_end.strftime("%Y-%m-%d")},
        "event": {"start": ev_start.strftime("%Y-%m-%d"), "end": ev_end.strftime("%Y-%m-%d")},
        "post_event": {"start": post_start.strftime("%Y-%m-%d"), "end": post_end.strftime("%Y-%m-%d")},
        "horizons": horizons,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    features_path = resolve_path(cfg["paths"]["features_path"])
    splits_dir = resolve_path(cfg["paths"]["splits_dir"])

    if not features_path.exists():
        raise FileNotFoundError(f"{features_path} not found. Run volforecast.build_features first.")

    features = pd.read_parquet(features_path)
    all_dates = pd.to_datetime(features["date"])
    modelling_start = pd.Timestamp(cfg["sample_period"]["modelling_start"])
    horizons = cfg["features"]["horizons"]

    splits_dir.mkdir(parents=True, exist_ok=True)
    for event_name, event_cfg in cfg["events"].items():
        split = build_split(event_name, event_cfg, all_dates, modelling_start, horizons)
        out_path = splits_dir / f"{event_name}.json"
        ensure_parent(out_path)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(split, f, indent=2)
        logger.info(
            "Wrote %s: train %s..%s (n~%d), val ends %s, event %s..%s",
            out_path.name,
            split["train"]["start"],
            split["train"]["end"],
            round((pd.Timestamp(split["train"]["end"]) - pd.Timestamp(split["train"]["start"])).days * 5 / 7),
            split["val"]["end"],
            split["event"]["start"],
            split["event"]["end"],
        )


if __name__ == "__main__":
    main()
