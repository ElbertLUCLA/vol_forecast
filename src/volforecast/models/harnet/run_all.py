"""Run the HARNet grid: select iterations on validation -> refit/freeze -> forecast -> diagnostics.

Usage:
    uv run python -m volforecast.models.harnet.run_all                     # 2 events x {base, ext} x h {1,5,22}
    uv run python -m volforecast.models.harnet.run_all --event covid --fs base --h 1 --max-iterations 2000
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from volforecast import adapters
from volforecast.models.harnet.config import Cell, load_harnet_config
from volforecast.models.harnet.data import MinMaxScaler, SeriesData, build_series
from volforecast.models.harnet.forecast import forecast_cell, load_model, save_model
from volforecast.models.harnet.model import har_regressors
from volforecast.models.harnet.train import TrainResult, predict, qlike, train
from volforecast.models.xgb.explain import plot_forecast
from volforecast.models.schema import forecast_path, validate_forecasts, write_forecasts

logger = logging.getLogger("volforecast.models.harnet.run_all")

# Random-walk benchmark (level), same choice as XGBoost's sanity check 6.
NAIVE = {1: "log_rv_d", 5: "log_rv_w", 22: "log_rv_m"}


def val_diagnostics(sel: TrainResult, data: SeriesData, features_df: pd.DataFrame, cell: Cell, variant: str) -> dict:
    """Validation QLIKE of the OLS-initialized HAR (iteration 0), trained HARNet, and the random walk."""
    vp = data.positions["val"]
    y = data.rv[vp]
    harnet = data.scaler.inverse_transform(predict(sel.model, data.x, vp).mean(axis=0))
    n_avg = 1 if variant == "HARNet" else 2
    X = har_regressors(data.x, sel.model.lags, n_avg, len(data.channels) - n_avg)[vp]
    har = data.scaler.inverse_transform(np.maximum(X @ sel.coef, float(sel.model.clip_min)))
    naive = np.exp(features_df[NAIVE[cell.h]].to_numpy(dtype=float)[vp]) if cell.h in NAIVE else None

    def stats(pred):
        return {"qlike": qlike(y, pred), "mse": float(np.mean((y - pred) ** 2))}

    out = {"harnet": stats(harnet), "har_ols_init": stats(har), "best_iter": sel.best_iter}
    if naive is not None:
        out["random_walk"] = stats(naive)
        out["beats_naive"] = out["harnet"]["qlike"] < out["random_walk"]["qlike"]
    out["improves_on_init"] = out["harnet"]["qlike"] < out["har_ols_init"]["qlike"]
    return out


def run_cell(cell: Cell, features_df: pd.DataFrame, split: dict, cfg: dict,
             max_iterations: int | None = None, plots: bool = True) -> dict:
    fs = cfg["featuresets"][cell.featureset]
    variant = fs["variant"]
    seeds = list(cfg["seeds"])
    scaler = MinMaxScaler(cfg["scaler"]["min"], cfg["scaler"]["max"])
    features_df = features_df.sort_values("date").reset_index(drop=True)
    data = build_series(features_df, split, fs["channels"], cell.h, cell.max_lag, scaler, cfg["train_tail_days"])
    model_dir = Path(cfg["paths"]["models_dir"]) / cell.event / cell.name
    n_max = max_iterations or cfg["train"]["max_iterations"]

    # 1-3: train-only fit, pick the iteration count on validation QLIKE.
    sel = train(data, data.positions["train"], fs, cfg, seeds, n_max, val_positions=data.positions["val"])
    diag = {"validation": val_diagnostics(sel, data, features_df, cell, variant)}

    # 4: refit on all purged pre-cutoff rows for best_iter iterations, freeze.
    fit = train(data, data.positions["refit"], fs, cfg, seeds, sel.best_iter)
    params = save_model(fit.model, model_dir, {
        "cell": {"event": cell.event, "featureset": cell.featureset, "h": cell.h, "model": cell.name,
                 "variant": variant, "channels": data.channels},
        "train_cfg": cfg["train"], "seeds": seeds, "scaler": cfg["scaler"],
        "best_iter": sel.best_iter, "max_iterations": n_max, "stopped_at": sel.curve[-1]["iteration"],
        "ols_coef_train": sel.coef, "ols_coef_refit": fit.coef,
        "cutoff": data.cutoff,
        "n_labels": {k: int(len(v)) for k, v in data.positions.items()},
    })
    pd.DataFrame(sel.curve).to_csv(model_dir / "val_curve.csv", index=False)

    # Forecasting uses only what was frozen to disk; the hash is re-checked after.
    frozen, _ = load_model(model_dir)
    forecasts = forecast_cell(frozen, data, cell, variant)
    validate_forecasts(forecasts, split)
    out_path = write_forecasts(forecasts, forecast_path(cfg["paths"]["forecasts_dir"], cell))
    load_model(model_dir)

    diag["timings_sec"] = {"select": sel.seconds, "refit": fit.seconds}
    oos = forecasts[forecasts["sample"] != "train_tail"]
    diag["oos_rel_seed_range_mean"] = float(
        ((oos["forecast_var_seed_max"] - oos["forecast_var_seed_min"]) / oos["forecast_var"]).mean())
    with open(model_dir / "diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(diag, f, indent=2, default=float)

    if plots:
        plot_forecast(forecasts, split, None, f"{cell.event} · {cell.name}",
                      Path(cfg["paths"]["reports_dir"]) / cell.event / f"{cell.name}_forecast.png",
                      forecast_label="forecast")

    v = diag["validation"]
    logger.info("%s/%s: best_iter %d, val QLIKE %.4f (HAR init %.4f, RW %s) -> %s",
                cell.event, cell.name, sel.best_iter, v["harnet"]["qlike"], v["har_ols_init"]["qlike"],
                f"{v['random_walk']['qlike']:.4f}" if "random_walk" in v else "n/a", out_path)
    return {"event": cell.event, "model": cell.name, "variant": variant, "featureset": cell.featureset,
            "h": cell.h, "best_iter": sel.best_iter, "stopped_at": params["stopped_at"],
            "val_qlike": v["harnet"]["qlike"], "val_qlike_har_init": v["har_ols_init"]["qlike"],
            "val_qlike_rw": v.get("random_walk", {}).get("qlike"),
            "sec_select": sel.seconds, "sec_refit": fit.seconds}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--event", nargs="+")
    p.add_argument("--fs", nargs="+", help="base ext (default)")
    p.add_argument("--h", nargs="+", type=int)
    p.add_argument("--max-iterations", type=int, help="override train.max_iterations (smoke runs)")
    p.add_argument("--config", help="path to an alternative harnet.yaml")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--overwrite", action="store_true", help="refit cells that already have results")
    return p.parse_args(argv)


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    cfg = load_harnet_config(args.config)
    features_df = adapters.load_features(cfg["paths"]["features_path"])
    events = args.event or cfg["events"]
    max_lag = max(cfg["lags"])
    cells = [Cell(e, fs, h, max_lag) for e in events
             for fs in (args.fs or cfg["default_featuresets"]) for h in (args.h or cfg["horizons"])]
    splits = {e: adapters.load_split(e, cfg["paths"]["splits_dir"]) for e in events}

    models_dir = Path(cfg["paths"]["models_dir"])
    for i, cell in enumerate(cells, 1):
        if (models_dir / cell.event / cell.name / "diagnostics.json").exists() and not args.overwrite:
            logger.info("[%d/%d] %s / %s already done, skipping", i, len(cells), cell.event, cell.name)
            continue
        t0 = time.perf_counter()
        logger.info("[%d/%d] %s / %s", i, len(cells), cell.event, cell.name)
        run_cell(cell, features_df, splits[cell.event], cfg, args.max_iterations, not args.no_plots)
        logger.info("    %.0fs", time.perf_counter() - t0)
        rebuild_run_log(models_dir)  # after every cell, so a crash loses at most the cell in progress
    logger.info("Done. Run log: %s", rebuild_run_log(models_dir))


def rebuild_run_log(models_dir: Path) -> Path:
    """run_log.csv from every finished cell's params.json + diagnostics.json (the source of truth)."""
    rows = []
    for diag_path in sorted(Path(models_dir).glob("*/*/diagnostics.json")):
        params = json.loads((diag_path.parent / "params.json").read_text(encoding="utf-8"))
        diag = json.loads(diag_path.read_text(encoding="utf-8"))
        v, c = diag["validation"], params["cell"]
        rows.append({"event": c["event"], "model": c["model"], "variant": c["variant"],
                     "featureset": c["featureset"], "h": c["h"], "best_iter": params["best_iter"],
                     "stopped_at": params["stopped_at"], "val_qlike": v["harnet"]["qlike"],
                     "val_qlike_har_init": v["har_ols_init"]["qlike"],
                     "val_qlike_rw": v.get("random_walk", {}).get("qlike"),
                     "sec_select": diag["timings_sec"]["select"], "sec_refit": diag["timings_sec"]["refit"]})
    log_path = Path(models_dir) / "run_log.csv"
    pd.DataFrame(rows).to_csv(log_path, index=False)
    return log_path


if __name__ == "__main__":
    main()
