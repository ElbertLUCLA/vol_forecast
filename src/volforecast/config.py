"""Load config.yaml and resolve paths relative to the project root."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    load_dotenv(PROJECT_ROOT / ".env")
    path = Path(config_path) if config_path else PROJECT_ROOT / "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def resolve_path(relative_path: str | Path) -> Path:
    """Resolve a path from config.yaml (given relative to the project root)."""
    p = Path(relative_path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
