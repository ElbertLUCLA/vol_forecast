"""Optuna search spaces (Teller et al. 2022, Table 4), read from configs/xgb.yaml."""
from __future__ import annotations

import optuna


def get_space(cfg: dict, space_name: str, booster: str) -> dict:
    spaces = cfg["search_spaces"]
    if space_name not in spaces:
        raise KeyError(f"Unknown search space '{space_name}'. Options: {sorted(spaces)}")
    return spaces[space_name][booster]


def early_stopping_rounds(cfg: dict, space_name: str) -> int | None:
    return cfg["search_spaces"][space_name].get("early_stopping_rounds")


def suggest(trial: optuna.Trial, space: dict) -> dict:
    """Sample one hyperparameter set. Keys are xgboost names; n_estimators = boosting rounds."""
    params = {}
    for name, spec in space.items():
        if spec["type"] == "float":
            params[name] = trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))
        elif spec["type"] == "int":
            params[name] = trial.suggest_int(name, spec["low"], spec["high"])
        else:
            raise ValueError(f"Unsupported parameter type {spec['type']!r} for {name}")
    return params
