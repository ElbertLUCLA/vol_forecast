"""Helpers shared by every model: smearing, QLIKE, file hashes, JSON output.

No model-library imports, so the CPU and ROCm environments can both use it.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def smearing(y_log: pd.Series, pred_log: np.ndarray) -> dict:
    """Duan (1983) smearing factor S = mean(exp(residual)) from out-of-sample log residuals."""
    resid = np.asarray(y_log, dtype=float) - np.asarray(pred_log, dtype=float)
    return {
        "S": float(np.mean(np.exp(resid))),
        "resid_mean": float(resid.mean()),
        "resid_std": float(resid.std(ddof=1)),
        "n": int(resid.size),
        "source": "val_residuals_train_only_model",
    }


def qlike(rv: np.ndarray, forecast_var: np.ndarray) -> float:
    """Patton (2011) QLIKE: mean(rv/f - ln(rv/f) - 1)."""
    ratio = np.asarray(rv, dtype=float) / np.asarray(forecast_var, dtype=float)
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path: Path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=json_default)


def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, pd.Timestamp):
        return str(o.date())
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Not JSON serializable: {type(o)}")


def append_run_log(rows: list[dict], log_path: Path) -> Path:
    """Append this run's per-cell training metrics to the model's run_log.csv."""
    log_path = Path(log_path)
    new = pd.DataFrame(rows).assign(finished=pd.Timestamp.now().isoformat(timespec="seconds"))
    if log_path.exists():
        new = pd.concat([pd.read_csv(log_path), new], ignore_index=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(log_path, index=False)
    return log_path
