"""Load configs/xgb.yaml and merge in the pipeline paths from config.yaml.

The pipeline config stays the single source for dates, cutoffs and data paths;
configs/xgb.yaml only adds model settings (grid, search space, seeds).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from volforecast.config import PROJECT_ROOT, load_config, resolve_path

XGB_CONFIG_PATH = PROJECT_ROOT / "configs" / "xgb.yaml"


@dataclass(frozen=True)
class Cell:
    """One fit in the 24-cell grid: event x feature set x horizon x booster."""

    event: str
    featureset: str
    h: int
    booster: str
    tag: str = ""  # suffix for robustness variants, e.g. "wide" or "covidfreeval"

    @property
    def name(self) -> str:
        base = f"xgb-{self.booster}_{self.featureset}_h{self.h}"
        return f"{base}_{self.tag}" if self.tag else base

    @property
    def study_name(self) -> str:
        return f"{self.event}__{self.name}"


def load_xgb_config(path: str | Path | None = None, pipeline_cfg: dict | None = None) -> dict[str, Any]:
    with open(path or XGB_CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    pipeline_cfg = pipeline_cfg if pipeline_cfg is not None else load_config()
    pipe_paths = pipeline_cfg["paths"]
    paths = cfg.setdefault("paths", {})
    paths.setdefault("features_path", pipe_paths["features_path"])
    paths.setdefault("splits_dir", pipe_paths["splits_dir"])
    paths.setdefault("forecasts_dir", pipe_paths["forecasts_dir"])
    paths.setdefault("daily_measures_path", pipe_paths["daily_measures_path"])
    for key, value in list(paths.items()):
        paths[key] = resolve_path(value)
    cfg["optuna"]["storage"] = resolve_path(cfg["optuna"]["storage"])

    missing = set(cfg["horizons"]) - set(pipeline_cfg["features"]["horizons"])
    if missing:
        raise ValueError(f"Horizons {sorted(missing)} are not built by the feature store (config.yaml).")
    cfg["naive_feature"] = {int(k): v for k, v in cfg["naive_feature"].items()}
    return cfg


def n_threads(cfg: dict) -> int:
    n = int(cfg["fixed_params"].get("n_jobs", -1))
    return (os.cpu_count() or 1) if n <= 0 else n
