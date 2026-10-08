"""Training primitives, plus refit-and-freeze for one cell (section 4.3).

Refit: the best trial's hyperparameters are refit on all purged pre-cutoff rows
(train + val). gbtree is refit with every seed in refit.seeds and the forecast
is the seed average; gblinear (cyclic coordinate descent) is deterministic, so
one fit is enough.

Smearing: S = mean(exp(residual)) on validation-block residuals of the
*train-only* models with the same hyperparameters and seeds, so S is out-of-sample.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from volforecast.models.common import qlike, sha256, smearing  # noqa: F401  (re-exported)
from volforecast.models.xgb.config import Cell, n_threads
from volforecast.models.xgb.data import CellFrames, Frame

# booster=gblinear is deprecated from xgboost 3.x but still supported (pinned <4 in pyproject).
warnings.filterwarnings("ignore", message=r".*gblinear.* is deprecated")


class Scaler:
    """Standardizes features with mean/std from the training block (gblinear only)."""

    def __init__(self, mean: pd.Series, std: pd.Series):
        self.mean = mean
        self.std = std.replace(0.0, 1.0)

    @classmethod
    def fit(cls, X: pd.DataFrame) -> "Scaler":
        return cls(X.mean(), X.std(ddof=0))

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return (X[self.mean.index] - self.mean) / self.std

    def to_dict(self) -> dict:
        return {"mean": self.mean.to_dict(), "std": self.std.to_dict()}

    @classmethod
    def from_dict(cls, d: dict) -> "Scaler":
        return cls(pd.Series(d["mean"], dtype=float), pd.Series(d["std"], dtype=float))


def native_params(booster: str, hp: dict, cfg: dict, seed: int) -> tuple[dict, int]:
    """Map sampled hyperparameters to xgb.train params. Returns (params, num_boost_round)."""
    hp = dict(hp)
    n_rounds = int(hp.pop("n_estimators"))
    fixed = cfg["fixed_params"]
    params = {
        "booster": booster,
        "objective": fixed["objective"],
        "nthread": n_threads(cfg),
        "verbosity": 1,
        **hp,
    }
    if booster == "gbtree":
        params.update(tree_method=fixed["tree_method"], seed=seed)
    elif booster == "gblinear":
        params.update(updater=fixed["updater"], feature_selector=fixed["feature_selector"])
    else:
        raise ValueError(f"Unknown booster {booster!r}")
    return params, n_rounds


def dmatrix(X: pd.DataFrame, y: pd.Series | None = None, scaler: Scaler | None = None) -> xgb.DMatrix:
    Xs = scaler.transform(X) if scaler is not None else X
    label = None if y is None else y.to_numpy(dtype=float)
    return xgb.DMatrix(Xs.to_numpy(dtype=float), label=label, feature_names=list(X.columns))


def train_booster(
    booster: str,
    hp: dict,
    frame: Frame,
    cfg: dict,
    seed: int,
    scaler: Scaler | None = None,
    eval_frame: Frame | None = None,
    early_stopping: int | None = None,
) -> xgb.Booster:
    params, n_rounds = native_params(booster, hp, cfg, seed)
    dtrain = dmatrix(frame.X, frame.y, scaler)
    kwargs = {}
    if early_stopping and eval_frame is not None:
        kwargs = dict(
            evals=[(dmatrix(eval_frame.X, eval_frame.y, scaler), "val")],
            early_stopping_rounds=early_stopping,
            verbose_eval=False,
        )
    return xgb.train(params, dtrain, num_boost_round=n_rounds, **kwargs)


def predict_log(models: list[xgb.Booster], X: pd.DataFrame, scaler: Scaler | None = None) -> np.ndarray:
    """Log forecasts, shape (n_models, n_rows). Uses the early-stopped iteration range if set."""
    d = dmatrix(X, scaler=scaler)
    return np.vstack([m.predict(d, **_iteration_kwargs(m)) for m in models])


def _iteration_kwargs(model: xgb.Booster) -> dict:
    best = model.attributes().get("best_iteration")
    return {"iteration_range": (0, int(best) + 1)} if best is not None else {}


@dataclass
class FittedCell:
    cell: Cell
    models: list[xgb.Booster]
    seeds: list[int]
    features: list[str]
    smear: dict
    scaler: Scaler | None = None
    params: dict = field(default_factory=dict)


def seeds_for(booster: str, cfg: dict) -> list[int]:
    seeds = list(cfg["refit"]["seeds"])
    return seeds if booster == "gbtree" else seeds[:1]


def resolve_rounds(hp: dict, best_iteration: int | None) -> dict:
    """With early stopping, the refit uses the number of rounds actually kept."""
    hp = dict(hp)
    if best_iteration is not None:
        hp["n_estimators"] = int(best_iteration) + 1
    return hp


def refit_cell(
    cell: Cell,
    frames: CellFrames,
    hp: dict,
    cfg: dict,
    model_dir: Path,
    tuning_info: dict | None = None,
) -> FittedCell:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    features = list(frames.train.features)
    seeds = seeds_for(cell.booster, cfg)
    scaler = Scaler.fit(frames.train.X) if cell.booster == "gblinear" else None

    # Train-only models -> out-of-sample validation residuals -> smearing factor.
    train_models = [train_booster(cell.booster, hp, frames.train, cfg, s, scaler) for s in seeds]
    val_pred = predict_log(train_models, frames.val.X, scaler).mean(axis=0)
    smear = smearing(frames.val.y, val_pred)

    # Refit on train + val (all purged pre-cutoff rows) and freeze.
    models = [train_booster(cell.booster, hp, frames.refit, cfg, s, scaler) for s in seeds]
    hashes = {}
    for seed, model in zip(seeds, models):
        path = model_dir / f"model_seed{seed}.json"
        model.save_model(path)
        hashes[path.name] = sha256(path)

    native, n_rounds = native_params(cell.booster, hp, cfg, seeds[0])
    params = {
        "cell": {"event": cell.event, "featureset": cell.featureset, "h": cell.h,
                 "booster": cell.booster, "tag": cell.tag, "model": cell.name},
        "hyperparameters": hp,
        "native_params": {k: v for k, v in native.items() if k != "seed"},
        "num_boost_round": n_rounds,
        "seeds": seeds,
        "cutoff": str(frames.cutoff.date()),
        "train_range": [str(frames.train.dates.min().date()), str(frames.train.dates.max().date())],
        "val_range": [str(frames.val.dates.min().date()), str(frames.val.dates.max().date())],
        "refit_range": [str(frames.refit.dates.min().date()), str(frames.refit.dates.max().date())],
        "n_rows": {"train": len(frames.train), "val": len(frames.val), "refit": len(frames.refit)},
        "model_sha256": hashes,
        "xgboost_version": xgb.__version__,
        "tuning": tuning_info or {},
    }
    _write_json(model_dir / "params.json", params)
    _write_json(model_dir / "features.json", features)
    _write_json(model_dir / "smear.json", smear)
    if scaler is not None:
        _write_json(model_dir / "scaler.json", scaler.to_dict())

    pd.DataFrame(
        {"y": frames.val.y, "pred_log": val_pred, "rv": frames.val.rv,
         "naive_log": frames.val.data[cfg["naive_feature"][cell.h]]},
        index=frames.val.dates,
    ).rename_axis("date").to_parquet(model_dir / "val_predictions.parquet")

    return FittedCell(cell, models, seeds, features, smear, scaler, params)


def load_fitted(cell: Cell, model_dir: Path) -> FittedCell:
    """Load a frozen cell from disk, verifying model files against the hashes saved at refit."""
    model_dir = Path(model_dir)
    params = _read_json(model_dir / "params.json")
    models = []
    for seed in params["seeds"]:
        path = model_dir / f"model_seed{seed}.json"
        if sha256(path) != params["model_sha256"][path.name]:
            raise RuntimeError(f"{path} changed after refit; the frozen model was modified.")
        booster = xgb.Booster()
        booster.load_model(path)
        models.append(booster)
    scaler_path = model_dir / "scaler.json"
    scaler = Scaler.from_dict(_read_json(scaler_path)) if scaler_path.exists() else None
    return FittedCell(
        cell=cell,
        models=models,
        seeds=params["seeds"],
        features=_read_json(model_dir / "features.json"),
        smear=_read_json(model_dir / "smear.json"),
        scaler=scaler,
        params=params,
    )


def _write_json(path: Path, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def _read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (pd.Timestamp,)):
        return str(o.date())
    raise TypeError(f"Not JSON serializable: {type(o)}")
