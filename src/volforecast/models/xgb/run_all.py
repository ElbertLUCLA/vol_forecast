"""Run the XGBoost grid: tune -> refit/freeze -> forecast -> diagnostics -> explain, per cell.

Usage:
    uv run python -m volforecast.models.xgb.run_all                      # full grid (24 cells)
    uv run python -m volforecast.models.xgb.run_all --event covid --fs base --h 1 --booster gbtree --n-trials 10
    uv run python -m volforecast.models.xgb.run_all --forecast-only                    # re-forecast frozen models
    uv run python -m volforecast.models.xgb.run_all --fs ext_x --tag ext_x             # robustness: CJX-style set
    uv run python -m volforecast.models.xgb.run_all --space wide --booster gbtree --tag wide
    uv run python -m volforecast.models.xgb.run_all --event bear2022 \\
        --val-window 2018-01-02 2020-01-31 --tag covidfreeval                     # COVID-free validation
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import optuna
import pandas as pd

from volforecast import adapters
from volforecast.models.xgb import diagnostics, explain
from volforecast.models.xgb.config import Cell, load_xgb_config
from volforecast.models.xgb.data import build_frames
from volforecast.models.xgb.fit import load_fitted, refit_cell
from volforecast.models.xgb.forecast import forecast_cell, forecast_path, validate_forecasts, write_forecasts
from volforecast.models.xgb.tune import best_hyperparameters, tune_cell

logger = logging.getLogger("volforecast.models.xgb.run_all")


def run_cell(
    cell: Cell,
    features_df: pd.DataFrame,
    split: dict,
    cfg: dict,
    n_trials: int | None = None,
    space_name: str = "default",
    val_window: tuple[str, str] | None = None,
    plots: bool = True,
    forecast_only: bool = False,
) -> dict:
    paths = cfg["paths"]
    model_dir = Path(paths["models_dir"]) / cell.event / cell.name
    frames = build_frames(features_df, split, cfg["featuresets"][cell.featureset], cell.h,
                          cfg["train_tail_days"], val_window)

    t0 = time.perf_counter()
    if forecast_only:
        # Re-forecast from the frozen model (e.g. after the forecast range was extended); no fitting.
        tuning_info = load_fitted(cell, model_dir).params.get("tuning", {})
        t1 = t2 = time.perf_counter()
    else:
        study = tune_cell(cell, frames, cfg, n_trials, space_name)
        hp, tuning_info = best_hyperparameters(study)
        t1 = time.perf_counter()
        refit_cell(cell, frames, hp, cfg, model_dir, tuning_info)
        t2 = time.perf_counter()

    # Forecasting uses only what refit_cell froze to disk.
    frozen = load_fitted(cell, model_dir)
    forecasts = forecast_cell(frozen, frames)
    validate_forecasts(forecasts, split)
    out_path = write_forecasts(forecasts, forecast_path(paths["forecasts_dir"], cell))
    load_fitted(cell, model_dir)  # re-verifies the model hashes after forecasting (check 3)
    t3 = time.perf_counter()

    timings = {"tune": t1 - t0, "refit": t2 - t1, "forecast": t3 - t2}
    diag = diagnostics.run_diagnostics(frozen, frames, forecasts, cfg, model_dir, timings)
    explain.importance(frozen, frames, model_dir)
    contribs = explain.contributions(frozen, frames, model_dir)
    if plots:
        fig_dir = Path(paths["reports_dir"]) / cell.event
        title = f"{cell.event} · {cell.name}"
        explain.plot_forecast(forecasts, split, diag["ceiling"].get("ceiling_var"), title,
                              fig_dir / f"{cell.name}_forecast.png")
        if cell.h in cfg["explain"]["contrib_plot_horizons"]:
            explain.plot_contributions(contribs, split, title, fig_dir / f"{cell.name}_contribs.png")

    logger.info(
        "%s/%s: val MSE %.4f (naive %.4f), S=%.3f, %d forecasts -> %s",
        cell.event, cell.name, diag["check6_beats_naive"]["val_mse_log"],
        diag["check6_beats_naive"]["val_mse_log_naive"], frozen.smear["S"], len(forecasts), out_path,
    )
    return {
        "event": cell.event, "model": cell.name, "booster": cell.booster, "featureset": cell.featureset,
        "h": cell.h, "space": space_name, **tuning_info,
        "S": frozen.smear["S"],
        "beats_naive": diag["check6_beats_naive"]["pass"],
        "smearing_ok": diag["check7_smearing"]["pass"],
        "ceiling_var": diag["ceiling"].get("ceiling_var"),
        **{f"sec_{k}": v for k, v in timings.items()},
    }


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--event", nargs="+", help="covid, bear2022 (default: all in configs/xgb.yaml)")
    p.add_argument("--fs", nargs="+", help="feature sets (default: base ext)")
    p.add_argument("--h", nargs="+", type=int, help="horizons (default: 1 5 22)")
    p.add_argument("--booster", nargs="+", choices=["gbtree", "gblinear"])
    p.add_argument("--n-trials", type=int, help="override optuna.n_trials (e.g. 10 for a smoke run)")
    p.add_argument("--space", default="default", help="search space name in configs/xgb.yaml")
    p.add_argument("--val-window", nargs=2, metavar=("START", "END"), help="override the validation block")
    p.add_argument("--tag", default="", help="suffix for model names (robustness variants)")
    p.add_argument("--config", help="path to an alternative xgb.yaml")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--forecast-only", action="store_true",
                   help="re-forecast from the frozen models without tuning or refitting")
    return p.parse_args(argv)


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    args = parse_args(argv)
    if (args.space != "default" or args.val_window) and not args.tag:
        raise SystemExit("Robustness variants need --tag so they don't overwrite the main results.")

    cfg = load_xgb_config(args.config)
    features_df = adapters.load_features(cfg["paths"]["features_path"])
    events = args.event or cfg["events"]
    cells = [
        Cell(event, fs, h, booster, args.tag)
        for event in events
        for fs in (args.fs or cfg["default_featuresets"])
        for h in (args.h or cfg["horizons"])
        for booster in (args.booster or cfg["boosters"])
    ]
    logger.info("Running %d cell(s)", len(cells))

    splits = {event: adapters.load_split(event, cfg["paths"]["splits_dir"]) for event in events}
    rows = []
    for i, cell in enumerate(cells, 1):
        logger.info("[%d/%d] %s / %s", i, len(cells), cell.event, cell.name)
        rows.append(run_cell(cell, features_df, splits[cell.event], cfg, args.n_trials, args.space,
                             tuple(args.val_window) if args.val_window else None, not args.no_plots,
                             args.forecast_only))

    models_dir = Path(cfg["paths"]["models_dir"])
    if args.forecast_only:
        logger.info("Done (forecast only; run log unchanged).")
        return
    log_path = models_dir / "run_log.csv"
    new = pd.DataFrame(rows).assign(finished=pd.Timestamp.now().isoformat(timespec="seconds"))
    if log_path.exists():
        new = pd.concat([pd.read_csv(log_path), new], ignore_index=True)
    new.to_csv(log_path, index=False)
    explain.importance_summary(models_dir)
    logger.info("Done. Run log: %s", log_path)


if __name__ == "__main__":
    main()
