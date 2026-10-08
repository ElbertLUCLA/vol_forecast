"""Freeze/load HARNet artifacts and produce forecasts in the shared schema.

HARNet is trained on the RV level with QLIKE, so its output already is a variance
forecast: forecast_var = forecast_var_naive = seed-average prediction (no smearing),
forecast_log = ln(forecast_var). The seed min/max band is kept as for XGBoost.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from volforecast.models.harnet.config import Cell
from volforecast.models.harnet.data import SeriesData
from volforecast.models.harnet.model import HARNetEnsemble
from volforecast.models.harnet.train import predict
from volforecast.models.xgb.data import target_dates
from volforecast.models.schema import FORECAST_SCHEMA

WEIGHTS_FILE = "weights.pt"


def save_model(model: HARNetEnsemble, model_dir: Path, params: dict) -> dict:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / WEIGHTS_FILE
    torch.save(model.state_dict(), path)
    params = {
        **params,
        "architecture": {"lags": model.lags, "n_avg": model.n_avg, "n_jump": model.n_jump,
                         "n_seeds": model.n_seeds, "clip_min": float(model.clip_min),
                         "clip_max": float(model.clip_max)},
        "weights_sha256": _sha256(path),
        "torch_version": torch.__version__,
    }
    with open(model_dir / "params.json", "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, default=_json_default)
    return params


def load_model(model_dir: Path) -> tuple[HARNetEnsemble, dict]:
    """Rebuild the frozen ensemble from disk, verifying the weights hash saved at refit."""
    model_dir = Path(model_dir)
    with open(model_dir / "params.json", "r", encoding="utf-8") as f:
        params = json.load(f)
    path = model_dir / WEIGHTS_FILE
    if _sha256(path) != params["weights_sha256"]:
        raise RuntimeError(f"{path} changed after refit; the frozen model was modified.")
    arch = params["architecture"]
    n_out = 1 + arch["n_avg"] * len(arch["lags"]) + arch["n_jump"]
    model = HARNetEnsemble(arch["lags"], arch["n_avg"], arch["n_jump"], np.zeros(n_out),
                           arch["n_seeds"], arch["clip_min"], arch["clip_max"])
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    return model, params


def forecast_cell(model: HARNetEnsemble, data: SeriesData, cell: Cell, variant: str) -> pd.DataFrame:
    positions = np.array(sorted(data.samples), dtype=np.int64)
    seed_var = data.scaler.inverse_transform(predict(model, data.x, positions))  # (S, n)
    fc = seed_var.mean(axis=0)
    dates = data.dates[positions]
    t_start, t_end = target_dates(dates, data.calendar, cell.h)
    realized = data.rv[positions]
    out = pd.DataFrame({
        "date": dates,
        "target_start": t_start.to_numpy(),
        "target_end": t_end.to_numpy(),
        "h": cell.h,
        "forecast_log": np.log(fc),
        "forecast_var": fc,
        "forecast_var_naive": fc,
        "forecast_var_seed_min": seed_var.min(axis=0),
        "forecast_var_seed_max": seed_var.max(axis=0),
        "realized_var": realized,
        "realized_log": np.log(realized),
        "target_complete": ~np.isnan(realized),
        "sample": [data.samples[int(p)] for p in positions],
        "model": cell.name,
        "booster": f"{variant}{cell.max_lag}",
        "featureset": cell.featureset,
        "event": cell.event,
    })
    return out.astype(FORECAST_SCHEMA)


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, pd.Timestamp):
        return str(o.date())
    raise TypeError(f"Not JSON serializable: {type(o)}")
