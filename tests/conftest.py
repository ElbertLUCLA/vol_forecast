"""Shared fixtures: a synthetic feature store pushed through the real pipeline functions."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from volforecast.build_features import build_features  # noqa: E402
from volforecast.config import load_config  # noqa: E402
from volforecast.make_splits import build_split  # noqa: E402

PIPE_CFG = load_config()


def synthetic_measures(seed: int = 0) -> pd.DataFrame:
    """HAR-like log-RV process on business days covering both events."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2004-01-02", "2023-06-30")
    n = len(dates)
    log_rv = np.empty(n)
    log_rv[:22] = -10.0
    for t in range(22, n):
        log_rv[t] = (-10.0 * 0.05 + 0.4 * log_rv[t - 1] + 0.35 * log_rv[t - 5:t].mean()
                     + 0.2 * log_rv[t - 22:t].mean() + 0.45 * rng.standard_normal())
    rv = np.exp(log_rv)
    neg_share = rng.uniform(0.3, 0.7, n)
    rs_neg, rs_pos = rv * neg_share, rv * (1 - neg_share)
    return pd.DataFrame({
        "date": dates, "rv": rv, "rs_neg": rs_neg, "rs_pos": rs_pos, "delta_j2": rs_pos - rs_neg,
        "medrv": rv * 0.95, "ret_cc": rng.normal(0, 0.01, n), "ret_overnight": rng.normal(0, 0.003, n),
    })


@pytest.fixture(scope="session")
def synth_features() -> pd.DataFrame:
    return build_features(synthetic_measures(), PIPE_CFG)


@pytest.fixture(scope="session")
def synth_splits(synth_features) -> dict:
    dates = pd.to_datetime(synth_features["date"])
    start = pd.Timestamp(PIPE_CFG["sample_period"]["modelling_start"])
    return {
        name: build_split(name, ev, dates, start, PIPE_CFG["features"]["horizons"])
        for name, ev in PIPE_CFG["events"].items()
    }
