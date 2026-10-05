"""Bayesian hyperparameter search for one cell (TPE, section 4.2).

Objective: validation MSE on the log target (as in Teller et al.). Validation
QLIKE on the smeared variance scale is stored per trial as a user attribute, so a
QLIKE-tuned variant can be picked later without re-running. Studies live in one
sqlite file and are resumable: re-running tops a study up to n_trials.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import optuna

from volforecast.models.xgb.config import Cell
from volforecast.models.xgb.data import CellFrames
from volforecast.models.xgb.fit import Scaler, predict_log, qlike, resolve_rounds, smearing, train_booster
from volforecast.models.xgb.space import early_stopping_rounds, get_space, suggest

logger = logging.getLogger("volforecast.models.xgb.tune")


def storage_url(path: Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{path.resolve().as_posix()}"


def make_objective(cell: Cell, frames: CellFrames, cfg: dict, space_name: str):
    space = get_space(cfg, space_name, cell.booster)
    es_rounds = early_stopping_rounds(cfg, space_name) if cell.booster == "gbtree" else None
    seed = cfg["optuna"]["seed"]
    scaler = Scaler.fit(frames.train.X) if cell.booster == "gblinear" else None
    y_val = frames.val.y
    rv_val = frames.val.rv.to_numpy(dtype=float)

    def objective(trial: optuna.Trial) -> float:
        hp = suggest(trial, space)
        model = train_booster(cell.booster, hp, frames.train, cfg, seed, scaler,
                              eval_frame=frames.val, early_stopping=es_rounds)
        pred = predict_log([model], frames.val.X, scaler)[0]
        mse = float(np.mean((y_val.to_numpy() - pred) ** 2))

        smear = smearing(y_val, pred)
        trial.set_user_attr("val_qlike", qlike(rv_val, np.exp(pred) * smear["S"]))
        trial.set_user_attr("val_mae_log", float(np.mean(np.abs(y_val.to_numpy() - pred))))
        trial.set_user_attr("smear_S", smear["S"])
        best_it = model.attributes().get("best_iteration")
        if best_it is not None:
            trial.set_user_attr("best_iteration", int(best_it))
        return mse

    return objective


def tune_cell(
    cell: Cell,
    frames: CellFrames,
    cfg: dict,
    n_trials: int | None = None,
    space_name: str = "default",
    storage: Path | None = None,
) -> optuna.Study:
    n_trials = n_trials if n_trials is not None else cfg["optuna"]["n_trials"]
    sampler = optuna.samplers.TPESampler(seed=cfg["optuna"]["seed"], multivariate=cfg["optuna"]["multivariate"])
    study = optuna.create_study(
        study_name=cell.study_name,
        storage=storage_url(storage or cfg["optuna"]["storage"]),
        sampler=sampler,
        direction="minimize",
        load_if_exists=True,
    )
    study.set_user_attr("space", space_name)
    done = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    remaining = max(0, n_trials - done)
    if remaining:
        logger.info("%s: running %d trial(s) (%d already done)", cell.study_name, remaining, done)
        study.optimize(make_objective(cell, frames, cfg, space_name), n_trials=remaining)
    else:
        logger.info("%s: %d trials already complete, skipping search", cell.study_name, done)
    return study


def best_hyperparameters(study: optuna.Study) -> tuple[dict, dict]:
    """(hp for refit, tuning summary for params.json)."""
    best = study.best_trial
    hp = resolve_rounds(best.params, best.user_attrs.get("best_iteration"))
    info = {
        "study": study.study_name,
        "space": study.user_attrs.get("space"),
        "n_trials": len(study.trials),
        "best_trial": best.number,
        "best_val_mse_log": best.value,
        "best_val_qlike": best.user_attrs.get("val_qlike"),
        "best_val_mae_log": best.user_attrs.get("val_mae_log"),
    }
    return hp, info
