"""Config loader shared by the HAR and GARCH baselines (configs/{har,garch}.yaml + config.yaml paths)."""
from __future__ import annotations

from pathlib import Path

import yaml

from volforecast.config import PROJECT_ROOT, load_config, resolve_path


def load_baseline_config(name: str, path: str | Path | None = None) -> dict:
    with open(path or PROJECT_ROOT / "configs" / f"{name}.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    pipe = load_config()["paths"]
    paths = cfg.setdefault("paths", {})
    for key in ("features_path", "splits_dir", "forecasts_dir"):
        paths.setdefault(key, pipe[key])
    for key, value in list(paths.items()):
        paths[key] = resolve_path(value)
    return cfg
