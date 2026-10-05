"""Tests for the XGBoost model (XGBoost_implementation.md section 10, checks 1-4 and 8).

Synthetic tests build a fake daily-measures table and push it through the real
pipeline functions (build_features, make_splits), so they always run. Tests
marked `real` use data/processed/... and are skipped if the pipeline hasn't run.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from volforecast import adapters  # noqa: E402
from volforecast.build_features import build_features  # noqa: E402
from volforecast.config import load_config, resolve_path  # noqa: E402
from volforecast.make_splits import build_split  # noqa: E402
from volforecast.models.xgb import fit as xfit  # noqa: E402
from volforecast.models.xgb.config import Cell, load_xgb_config  # noqa: E402
from volforecast.models.xgb.data import build_frames, purge, target_dates, trading_calendar  # noqa: E402
from volforecast.models.xgb.forecast import FORECAST_SCHEMA, validate_forecasts  # noqa: E402
from volforecast.models.xgb.run_all import run_cell  # noqa: E402

PIPE_CFG = load_config()


# ---------------------------------------------------------------- synthetic data

def _synthetic_measures(seed: int = 0) -> pd.DataFrame:
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


@pytest.fixture(scope="module")
def synth_features() -> pd.DataFrame:
    return build_features(_synthetic_measures(), PIPE_CFG)


@pytest.fixture(scope="module")
def synth_splits(synth_features) -> dict:
    dates = pd.to_datetime(synth_features["date"])
    start = pd.Timestamp(PIPE_CFG["sample_period"]["modelling_start"])
    return {
        name: build_split(name, ev, dates, start, PIPE_CFG["features"]["horizons"])
        for name, ev in PIPE_CFG["events"].items()
    }


@pytest.fixture()
def tmp_cfg(tmp_path) -> dict:
    cfg = load_xgb_config()
    cfg["paths"].update(models_dir=tmp_path / "models", forecasts_dir=tmp_path / "forecasts",
                        reports_dir=tmp_path / "reports")
    cfg["optuna"]["storage"] = tmp_path / "optuna.db"
    cfg["refit"]["seeds"] = [42, 43]
    cfg["fixed_params"]["n_jobs"] = 2
    return cfg


# ---------------------------------------------------------------- config congruence

def test_featuresets_match_adapters():
    cfg = load_xgb_config()
    assert cfg["featuresets"]["base"] == adapters.FEATURE_SETS["base"]
    assert cfg["featuresets"]["ext"] == adapters.FEATURE_SETS["ext"]


# ---------------------------------------------------------------- check 2: purge

def test_purge_drops_rows_whose_target_crosses_boundary():
    df = pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=30)})
    boundary = df["date"].iloc[19]
    for h in (1, 5, 22):
        kept = purge(df, boundary, h)
        _, end = target_dates(kept["date"], trading_calendar(df), h)
        assert (end <= boundary).all()
        assert len(kept) == max(0, 20 - h)


@pytest.mark.parametrize("event", ["covid", "bear2022"])
@pytest.mark.parametrize("h", [1, 5, 22])
def test_frames_are_purged_and_match_split(synth_features, synth_splits, event, h):
    split = synth_splits[event]
    fr = build_frames(synth_features, split, adapters.FEATURE_SETS["ext"], h)  # also asserts vs end_by_horizon
    cal = fr.calendar
    _, train_end = target_dates(fr.train.dates, cal, h)
    _, refit_end = target_dates(fr.refit.dates, cal, h)
    assert train_end.max() < fr.val.dates.min()
    assert refit_end.max() <= fr.cutoff
    assert fr.oos.dates.min() > fr.cutoff
    assert fr.tail.dates.max() <= fr.cutoff and len(fr.tail) == 63


def test_frames_agree_with_tabular_view(synth_features, synth_splits):
    split = synth_splits["covid"]
    fr = build_frames(synth_features, split, adapters.BASE_FEATURES, 5)
    for seg, frame in (("train", fr.train), ("val", fr.val)):
        X, y = adapters.tabular_view(synth_features, split, seg, "base", 5)
        pd.testing.assert_frame_equal(frame.X, X, check_names=False)
        np.testing.assert_allclose(frame.y.to_numpy(), y.to_numpy())


# ---------------------------------------------------------------- checks 3, 4, 8: end to end

def _run(cell, features, split, cfg):
    return run_cell(cell, features, split, cfg, n_trials=4, plots=False)


@pytest.mark.parametrize("booster", ["gbtree", "gblinear"])
def test_end_to_end_schema_frozen_and_reproducible(synth_features, synth_splits, tmp_cfg, monkeypatch, booster):
    split = synth_splits["covid"]
    cell = Cell("covid", "ext", 5, booster)
    _run(cell, synth_features, split, tmp_cfg)

    model_dir = Path(tmp_cfg["paths"]["models_dir"]) / "covid" / cell.name
    fc_path = Path(tmp_cfg["paths"]["forecasts_dir"]) / "covid" / f"{cell.name}.parquet"
    first = pd.read_parquet(fc_path)

    # Check 8: schema.
    assert list(first.columns) == list(FORECAST_SCHEMA)
    validate_forecasts(first, split)
    assert set(first["sample"]) == {"train_tail", "pre_event", "event", "post_event"}
    for name in ("params.json", "features.json", "smear.json", "diagnostics.json", "importance.csv"):
        assert (model_dir / name).exists(), name
    assert (model_dir / "scaler.json").exists() == (booster == "gblinear")

    # Check 3: forecasting never trains, and loading verifies the frozen hashes.
    from volforecast.models.xgb import forecast as xforecast
    frames = build_frames(synth_features, split, tmp_cfg["featuresets"]["ext"], 5)

    def no_training(*a, **k):
        raise AssertionError("forecast path called xgb.train")

    monkeypatch.setattr(xfit.xgb, "train", no_training)
    frozen = xfit.load_fitted(cell, model_dir)
    pd.testing.assert_frame_equal(xforecast.forecast_cell(frozen, frames), first)
    monkeypatch.undo()

    seed_file = model_dir / f"model_seed{frozen.seeds[0]}.json"
    seed_file.write_text(seed_file.read_text() + " ")
    with pytest.raises(RuntimeError, match="changed after refit"):
        xfit.load_fitted(cell, model_dir)

    # Check 4: a fresh run with the same seed gives identical forecasts.
    for p in model_dir.glob("*"):
        p.unlink()
    tmp_cfg["optuna"]["storage"] = Path(tmp_cfg["optuna"]["storage"]).with_name("optuna_rerun.db")
    _run(cell, synth_features, split, tmp_cfg)
    pd.testing.assert_frame_equal(pd.read_parquet(fc_path), first)


# ---------------------------------------------------------------- check 1: no look-ahead (real data)

FEATURES_PATH = resolve_path(PIPE_CFG["paths"]["features_path"])
MEASURES_PATH = resolve_path(PIPE_CFG["paths"]["daily_measures_path"])
SPLITS_DIR = resolve_path(PIPE_CFG["paths"]["splits_dir"])
real = pytest.mark.skipif(
    not (FEATURES_PATH.exists() and MEASURES_PATH.exists() and (SPLITS_DIR / "covid.json").exists()),
    reason="Pipeline has not been run yet.",
)


@real
@pytest.mark.parametrize("h", [1, 5, 22])
def test_targets_use_only_future_rv(h):
    features = adapters.load_features(FEATURES_PATH)
    measures = pd.read_parquet(MEASURES_PATH).sort_values("date").reset_index(drop=True)
    measures = measures[measures["date"] >= features["date"].min()].reset_index(drop=True)
    rng = np.random.default_rng(h)
    for i in rng.choice(len(features) - h - 1, size=50, replace=False):
        expected = measures["rv"].iloc[i + 1 : i + 1 + h].mean()
        assert features[f"rv_h{h}"].iloc[i] == pytest.approx(expected, rel=1e-12)


@real
def test_features_at_t_use_only_data_up_to_t():
    features = adapters.load_features(FEATURES_PATH)
    measures = pd.read_parquet(MEASURES_PATH).sort_values("date").reset_index(drop=True)
    cols = adapters.FEATURE_SETS["ext"]
    for date in pd.to_datetime(["2008-10-10", "2020-02-21", "2020-03-16", "2022-06-13"]):
        truncated = build_features(measures[measures["date"] <= date], PIPE_CFG)
        recomputed = truncated.set_index("date").loc[date, cols].astype(float)
        stored = features.set_index("date").loc[date, cols].astype(float)
        # Log-floor eps is a quantile of the history, so only exact for non-floored days.
        np.testing.assert_allclose(recomputed.to_numpy(), stored.to_numpy(), rtol=1e-10)


@real
@pytest.mark.parametrize("event", ["covid", "bear2022"])
def test_real_splits_purge_consistent(event):
    features = adapters.load_features(FEATURES_PATH)
    split = adapters.load_split(event, SPLITS_DIR)
    for h in (1, 5, 22):
        build_frames(features, split, adapters.FEATURE_SETS["ext"], h)  # raises on mismatch
