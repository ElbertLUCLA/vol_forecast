"""HAR-level: HARNet's OLS level-HAR initialization (iteration 0), frozen at the cutoff and forecast out of sample.

For each of the 12 HARNet cells this fits ols_fit on the same refit rows (all purged pre-cutoff labels) that
initialize the frozen HARNet, builds the network at that initialization (which reproduces the level HAR
exactly, model.py) and forecasts with HARNet's forecast_cell, so the files match the shared schema. No
training: the existing HARNet models are only read, to check that the refit OLS coefficients agree.

    uv run python -m volforecast.models.harnet.har_level

Writes data/processed/forecasts/{event}/harnet20_init_{fs}_h{h}.parquet (family "HAR-level" in the
evaluation) and models/harnet_init/{event}/{model}/params.json.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from volforecast import adapters
from volforecast.config import PROJECT_ROOT
from volforecast.models.harnet.config import Cell, load_harnet_config
from volforecast.models.harnet.data import MinMaxScaler, build_series
from volforecast.models.harnet.forecast import _json_default, forecast_cell
from volforecast.models.harnet.train import make_model
from volforecast.models.schema import validate_forecasts, write_forecasts

logger = logging.getLogger("volforecast.models.harnet.har_level")

MODELS_DIR = PROJECT_ROOT / "models" / "harnet_init"


def model_name(cell: Cell) -> str:
    return f"harnet{cell.max_lag}_init_{cell.featureset}_h{cell.h}"


def run_cell(cell: Cell, features_df: pd.DataFrame, split: dict, cfg: dict,
             models_dir: Path = MODELS_DIR) -> pd.DataFrame:
    fs = cfg["featuresets"][cell.featureset]
    scaler = MinMaxScaler(cfg["scaler"]["min"], cfg["scaler"]["max"])
    features_df = features_df.sort_values("date").reset_index(drop=True)
    data = build_series(features_df, split, fs["channels"], cell.h, cell.max_lag, scaler, cfg["train_tail_days"])

    # Same OLS fit and clip as the HARNet refit initialization; one copy, since there is no seed to vary.
    model, coef = make_model(data, data.positions["refit"], fs, cfg["lags"], 1, cfg["train"]["clip_max"])
    model.eval()

    harnet_params = Path(cfg["paths"]["models_dir"]) / cell.event / cell.name / "params.json"
    if harnet_params.exists():
        frozen = json.loads(harnet_params.read_text(encoding="utf-8"))
        if not np.allclose(coef, frozen["ols_coef_refit"], rtol=1e-8, atol=1e-12):
            raise RuntimeError(f"{cell.event}/{cell.name}: OLS coefficients differ from the frozen HARNet refit init.")

    name = model_name(cell)
    out = forecast_cell(model, data, cell, fs["variant"])
    out["model"] = name
    out["booster"] = f"HAR-level{cell.max_lag}"
    validate_forecasts(out, split)
    write_forecasts(out, Path(cfg["paths"]["forecasts_dir"]) / cell.event / f"{name}.parquet")

    model_dir = Path(models_dir) / cell.event / name
    model_dir.mkdir(parents=True, exist_ok=True)
    with open(model_dir / "params.json", "w", encoding="utf-8") as f:
        json.dump({"cell": {"event": cell.event, "featureset": cell.featureset, "h": cell.h, "model": name,
                            "variant": fs["variant"], "channels": data.channels},
                   "ols_coef_refit": coef, "clip_min_scaled": float(model.clip_min),
                   "clip_max_scaled": float(model.clip_max), "scaler": cfg["scaler"], "lags": cfg["lags"],
                   "cutoff": data.cutoff, "n_refit_labels": int(len(data.positions["refit"]))},
                  f, indent=2, default=_json_default)
    return out


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_harnet_config()
    features_df = adapters.load_features(cfg["paths"]["features_path"])
    max_lag = max(cfg["lags"])
    for event in cfg["events"]:
        split = adapters.load_split(event, cfg["paths"]["splits_dir"])
        for fs in cfg["default_featuresets"]:
            for h in cfg["horizons"]:
                cell = Cell(event, fs, h, max_lag)
                out = run_cell(cell, features_df, split, cfg)
                logger.info("%s/%s: %d rows", event, model_name(cell), len(out))


if __name__ == "__main__":
    main()
