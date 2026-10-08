"""Load configs/harnet.yaml and merge in the pipeline paths from config.yaml."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from volforecast.config import PROJECT_ROOT, load_config, resolve_path

HARNET_CONFIG_PATH = PROJECT_ROOT / "configs" / "harnet.yaml"


@dataclass(frozen=True)
class Cell:
    """One HARNet fit: event x feature set x horizon."""

    event: str
    featureset: str
    h: int
    max_lag: int = 20
    tag: str = ""

    @property
    def name(self) -> str:
        base = f"harnet{self.max_lag}_{self.featureset}_h{self.h}"
        return f"{base}_{self.tag}" if self.tag else base


def load_harnet_config(path: str | Path | None = None, pipeline_cfg: dict | None = None) -> dict[str, Any]:
    with open(path or HARNET_CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    pipeline_cfg = pipeline_cfg if pipeline_cfg is not None else load_config()
    pipe_paths = pipeline_cfg["paths"]
    paths = cfg.setdefault("paths", {})
    for key in ("features_path", "splits_dir", "forecasts_dir"):
        paths.setdefault(key, pipe_paths[key])
    for key, value in list(paths.items()):
        paths[key] = resolve_path(value)

    missing = set(cfg["horizons"]) - set(pipeline_cfg["features"]["horizons"])
    if missing:
        raise ValueError(f"Horizons {sorted(missing)} are not built by the feature store (config.yaml).")
    lags = cfg["lags"]
    if lags[0] != 1 or any(b % a for a, b in zip(lags, lags[1:])):
        raise ValueError(f"lags {lags} must start at 1 and each must be a multiple of the previous one.")
    return cfg
