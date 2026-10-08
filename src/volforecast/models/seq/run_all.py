"""Run the Mamba / LSTM grid: tune lookback -> train seeds -> smear -> refit/freeze -> forecast.

GPU (AMD ROCm environment, see README):
    .venv-rocm/Scripts/python -m volforecast.models.seq.run_all --arch mamba
    .venv-rocm/Scripts/python -m volforecast.models.seq.run_all --arch lstm --event covid
CPU works too (slower): uv run python -m volforecast.models.seq.run_all --device cpu ...

Finished cells are skipped (resume after a crash) unless --overwrite is given.
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from volforecast import adapters
from volforecast.models.common import append_run_log, qlike, read_json, sha256, smearing, write_json
from volforecast.models.schema import FORECAST_SCHEMA, forecast_path, validate_forecasts, write_forecasts
from volforecast.models.seq.data import Cell, SeqData, build_seq_data, load_seq_config
from volforecast.models.seq.train import fit, make_model, predict
from volforecast.models.xgb.data import target_dates

logger = logging.getLogger("volforecast.models.seq.run_all")
NAIVE = {1: "log_rv_d", 5: "log_rv_w", 22: "log_rv_m"}


def tune(cell: Cell, data: SeqData, cfg: dict, device: str) -> tuple[dict, list[dict]]:
    rows = []
    seed = cfg["seeds"][0]
    for lookback in cfg["tuning"]["lookback"]:
        for lr in cfg["tuning"]["learning_rate"]:
            r = fit(data, cfg, cell.arch, seed, lookback, lr, data.positions["train"], device,
                    val_positions=data.positions["val"])
            rows.append({"lookback": lookback, "learning_rate": lr, "val_mse": r.best_val_mse,
                         "best_epoch": r.best_epoch, "seconds": r.seconds})
            logger.info("    tune L=%d lr=%g: val MSE %.4f at epoch %d (%.0fs)", lookback, lr,
                        r.best_val_mse, r.best_epoch, r.seconds)
    best = min(rows, key=lambda r: r["val_mse"])
    return {"lookback": best["lookback"], "learning_rate": best["learning_rate"]}, rows


def run_cell(cell: Cell, features_df: pd.DataFrame, split: dict, cfg: dict, device: str) -> dict:
    t0 = time.perf_counter()
    channels = cfg["featuresets"][cell.featureset]
    data = build_seq_data(features_df, split, channels, cell.h, max(cfg["tuning"]["lookback"]),
                          cfg["train_tail_days"])
    model_dir = Path(cfg["paths"]["models_dir"]) / cell.event / cell.name
    model_dir.mkdir(parents=True, exist_ok=True)

    hp, tuning_rows = tune(cell, data, cfg, device)
    L, lr = hp["lookback"], hp["learning_rate"]
    t_tune = time.perf_counter()

    # Selection fits (train block, every seed) -> best epochs + out-of-sample smearing.
    vp = data.positions["val"]
    sel = [fit(data, cfg, cell.arch, s, L, lr, data.positions["train"], device, val_positions=vp)
           for s in cfg["seeds"]]
    val_pred = np.mean([predict(r.model, data, vp, L, device) for r in sel], axis=0)
    smear = smearing(data.y[vp], val_pred)

    # Refit each seed on train + val for its own best-epoch count, then freeze.
    refit = [fit(data, cfg, cell.arch, s, L, lr, data.positions["refit"], device, n_epochs=r.best_epoch)
             for s, r in zip(cfg["seeds"], sel)]
    t_fit = time.perf_counter()
    weights = model_dir / "weights.pt"
    torch.save([r.model.state_dict() for r in refit], weights)

    rv_val = data.rv[vp]
    naive = np.exp(features_df.sort_values("date")[NAIVE[cell.h]].to_numpy(dtype=float)[vp])
    val = {
        "mse_log": float(np.mean((data.y[vp] - val_pred) ** 2)),
        "mse_log_naive": float(np.mean((data.y[vp] - np.log(naive)) ** 2)),
        "qlike": qlike(rv_val, np.exp(val_pred) * smear["S"]),
        "qlike_naive": qlike(rv_val, naive),
    }
    ceilings = [r.model.output_ceiling() for r in refit]
    params = {
        "cell": {"event": cell.event, "arch": cell.arch, "featureset": cell.featureset, "h": cell.h,
                 "model": cell.name, "channels": channels},
        "model_cfg": cfg["model"], "train_cfg": cfg["train"], "chosen": hp, "tuning": tuning_rows,
        "seeds": cfg["seeds"], "best_epochs": [r.best_epoch for r in sel],
        "sel_val_mse_per_seed": [r.best_val_mse for r in sel],
        "input_mean": data.mean, "input_std": data.std, "smear": smear, "val": val,
        "ceiling_log_per_seed": ceilings, "cutoff": data.cutoff, "device": device,
        "torch_version": torch.__version__, "weights_sha256": sha256(weights),
        "n_rows": {k: int(len(v)) for k, v in data.positions.items()},
    }
    write_json(model_dir / "params.json", params)
    pd.DataFrame([{"seed": s, **c} for s, r in zip(cfg["seeds"], sel) for c in r.curve]).to_csv(
        model_dir / "train_curves.csv", index=False)

    forecasts = forecast_cell(cell, data, model_dir, cfg, device)
    validate_forecasts(forecasts, split)
    write_forecasts(forecasts, forecast_path(cfg["paths"]["forecasts_dir"], cell))

    ceiling_var = float(np.exp(np.mean(ceilings)) * smear["S"])
    ev = forecasts[(forecasts["sample"] == "event") & forecasts["target_complete"]]
    diag = {"val": val, "ceiling_log": float(np.mean(ceilings)), "ceiling_var": ceiling_var,
            "max_train_target_log": float(np.nanmax(data.y[data.positions["refit"]])),
            "event_days_realized_above_ceiling": int((ev["realized_var"] > ceiling_var).sum()),
            "timings_sec": {"tune": t_tune - t0, "select_refit": t_fit - t_tune}}
    write_json(model_dir / "diagnostics.json", diag)
    logger.info("%s/%s: L=%d, epochs %s, val QLIKE %.4f (RW %.4f), S=%.3f, ceiling %.0f%% ann. vol (%.0fs)",
                cell.event, cell.name, L, params["best_epochs"], val["qlike"], val["qlike_naive"], smear["S"],
                100 * np.sqrt(252 * ceiling_var), time.perf_counter() - t0)
    return {"event": cell.event, "model": cell.name, "arch": cell.arch, "featureset": cell.featureset,
            "h": cell.h, "lookback": L, "learning_rate": lr, "best_epochs": str(params["best_epochs"]),
            "val_mse_log": val["mse_log"], "val_mse_log_rw": val["mse_log_naive"], "val_qlike": val["qlike"],
            "val_qlike_rw": val["qlike_naive"], "S": smear["S"], "ceiling_var": ceiling_var,
            "sec_tune": t_tune - t0, "sec_select_refit": t_fit - t_tune, "device": device}


def load_frozen(cell: Cell, model_dir: Path, cfg: dict, device: str) -> tuple[list, dict]:
    params = read_json(model_dir / "params.json")
    weights = model_dir / "weights.pt"
    if sha256(weights) != params["weights_sha256"]:
        raise RuntimeError(f"{weights} changed after refit; the frozen model was modified.")
    models = []
    for seed, state in zip(params["seeds"], torch.load(weights, map_location=device, weights_only=True)):
        m = make_model(cfg, cell.arch, len(params["cell"]["channels"]), seed).to(device)
        m.load_state_dict(state)
        models.append(m.eval())
    return models, params


def forecast_cell(cell: Cell, data: SeqData, model_dir: Path, cfg: dict, device: str) -> pd.DataFrame:
    """Forecasts from the frozen models on disk only (hash-checked)."""
    models, params = load_frozen(cell, model_dir, cfg, device)
    L, S = params["chosen"]["lookback"], params["smear"]["S"]
    positions = np.array(sorted(data.samples), dtype=np.int64)
    preds = np.vstack([predict(m, data, positions, L, device) for m in models])  # (seeds, n)
    log_fc = preds.mean(axis=0)
    seed_var = np.exp(preds) * S
    dates = data.dates[positions]
    t_start, t_end = target_dates(dates, data.calendar, cell.h)
    realized = data.rv[positions]
    return pd.DataFrame({
        "date": dates, "target_start": t_start.to_numpy(), "target_end": t_end.to_numpy(), "h": cell.h,
        "forecast_log": log_fc, "forecast_var": np.exp(log_fc) * S, "forecast_var_naive": np.exp(log_fc),
        "forecast_var_seed_min": seed_var.min(axis=0), "forecast_var_seed_max": seed_var.max(axis=0),
        "realized_var": realized, "realized_log": np.log(realized), "target_complete": ~np.isnan(realized),
        "sample": [data.samples[int(p)] for p in positions],
        "model": cell.name, "booster": cell.arch, "featureset": cell.featureset, "event": cell.event,
    }).astype(FORECAST_SCHEMA)


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arch", nargs="+", choices=["mamba", "lstm"])
    p.add_argument("--event", nargs="+")
    p.add_argument("--fs", nargs="+")
    p.add_argument("--h", nargs="+", type=int)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-epochs", type=int, help="override train.max_epochs (smoke runs)")
    p.add_argument("--lookback", nargs="+", type=int, help="override the tuning lookback grid")
    p.add_argument("--overwrite", action="store_true", help="refit cells that already have results")
    p.add_argument("--threads", type=int, help="cap CPU threads (keep the machine usable)")
    args = p.parse_args(argv)

    if args.threads:
        torch.set_num_threads(args.threads)
    cfg = load_seq_config()
    if args.max_epochs:
        cfg["train"]["max_epochs"] = args.max_epochs
    if args.lookback:
        cfg["tuning"]["lookback"] = args.lookback
    features_df = adapters.load_features(cfg["paths"]["features_path"])
    cells = [Cell(e, a, fs, h) for a in (args.arch or cfg["archs"]) for e in (args.event or cfg["events"])
             for fs in (args.fs or cfg["default_featuresets"]) for h in (args.h or cfg["horizons"])]
    logger.info("Running %d cell(s) on %s (%s)", len(cells), args.device,
                torch.cuda.get_device_name(0) if args.device == "cuda" else "cpu")

    rows = []
    for i, cell in enumerate(cells, 1):
        done = Path(cfg["paths"]["models_dir"]) / cell.event / cell.name / "diagnostics.json"
        if done.exists() and not args.overwrite:
            logger.info("[%d/%d] %s/%s already done, skipping", i, len(cells), cell.event, cell.name)
            continue
        logger.info("[%d/%d] %s/%s", i, len(cells), cell.event, cell.name)
        split = adapters.load_split(cell.event, cfg["paths"]["splits_dir"])
        row = run_cell(cell, features_df, split, cfg, args.device)
        append_run_log([row], Path(cfg["paths"]["models_dir"]) / "run_log.csv")  # per cell: survives crashes
        rows.append(row)
    logger.info("Done: %d cell(s) fitted.", len(rows))


if __name__ == "__main__":
    main()
