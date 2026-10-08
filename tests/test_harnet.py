"""Tests for the PyTorch HARNet port.

Port fidelity: the network at initialization equals the OLS HAR; convolutions are
causal; the seed-vectorized ensemble trains each copy exactly as it would alone.
Design: label positions are purged like XGBoost; forecasts are in the shared
schema, frozen (hash-checked) and reproducible.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from volforecast import adapters
from volforecast.models.harnet.config import Cell, load_harnet_config
from volforecast.models.harnet.data import MinMaxScaler, build_series, window_starts
from volforecast.models.harnet.forecast import load_model
from volforecast.models.harnet.model import DTYPE, conv_specs, har_regressors
from volforecast.models.harnet.run_all import run_cell
from volforecast.models.harnet.train import make_model, predict, train
from volforecast.models.xgb.data import target_dates
from volforecast.models.schema import FORECAST_SCHEMA, validate_forecasts

CFG = load_harnet_config()
SCALER = MinMaxScaler(CFG["scaler"]["min"], CFG["scaler"]["max"])


@pytest.fixture()
def tmp_cfg(tmp_path) -> dict:
    cfg = load_harnet_config()
    cfg["paths"].update(models_dir=tmp_path / "models", forecasts_dir=tmp_path / "forecasts",
                        reports_dir=tmp_path / "reports")
    cfg["seeds"] = [42, 43]
    cfg["train"].update(max_iterations=300, eval_every=50)
    return cfg


def _series(features, split, fs, h, lags=(1, 5, 20)):
    return build_series(features, split, CFG["featuresets"][fs]["channels"], h, max(lags), SCALER)


def test_conv_specs_match_paper():
    assert conv_specs([1, 5, 20]) == [(5, 1), (4, 5)]
    assert conv_specs([1, 5, 20, 40, 80]) == [(5, 1), (4, 5), (2, 20), (2, 40)]


@pytest.mark.parametrize("fs", ["base", "ext"])
@pytest.mark.parametrize("lags", [[1, 5, 20], [1, 5, 20, 40, 80]])
def test_init_reproduces_ols_har(synth_features, synth_splits, fs, lags):
    data = _series(synth_features, synth_splits["covid"], fs, 5, lags)
    variant = CFG["featuresets"][fs]
    model, coef = make_model(data, data.positions["train"], variant, lags, n_seeds=3, clip_max=1000.0)
    n_avg = 1 if variant["variant"] == "HARNet" else 2
    X = har_regressors(data.x, lags, n_avg, len(data.channels) - n_avg)
    pos = data.positions["val"]
    expected = np.maximum(X[pos] @ coef, float(model.clip_min))
    got = predict(model, data.x, pos)
    assert got.shape == (3, len(pos))
    np.testing.assert_allclose(got, np.broadcast_to(expected, got.shape), atol=1e-12)


def test_convolutions_are_causal(synth_features, synth_splits):
    data = _series(synth_features, synth_splits["covid"], "ext", 1)
    model, _ = make_model(data, data.positions["train"], CFG["featuresets"]["ext"], [1, 5, 20], 1, 1000.0)
    with torch.no_grad():
        for w in model.convs:
            w.add_(torch.rand_like(w))  # break the plain-average init so all taps matter
    x = torch.as_tensor(data.x[:200], dtype=DTYPE)[None, None]
    base = model(x)
    x2 = x.clone()
    x2[..., 150:, :] += 5.0
    changed = model(x2)
    assert torch.equal(base[..., :150], changed[..., :150])
    assert not torch.equal(base[..., 150:], changed[..., 150:])


def test_seed_vectorization_matches_single_runs(synth_features, synth_splits):
    data = _series(synth_features, synth_splits["covid"], "base", 5)
    pos = data.positions["val"]
    variant = CFG["featuresets"]["base"]
    joint = train(data, data.positions["train"], variant, CFG, [42, 43], 200)
    alone = train(data, data.positions["train"], variant, CFG, [43], 200)
    np.testing.assert_allclose(predict(joint.model, data.x, pos)[1], predict(alone.model, data.x, pos)[0],
                               rtol=1e-9)


@pytest.mark.parametrize("h", [1, 5, 22])
def test_label_positions_are_purged(synth_features, synth_splits, h):
    split = synth_splits["bear2022"]
    data = _series(synth_features, split, "ext", h)
    _, train_end = target_dates(data.dates[data.positions["train"]], data.calendar, h)
    _, refit_end = target_dates(data.dates[data.positions["refit"]], data.calendar, h)
    assert train_end.max() < pd.Timestamp(split["val"]["start"])
    assert refit_end.max() <= data.cutoff
    assert data.dates[data.positions["train"]].max() == pd.Timestamp(split["train"]["end_by_horizon"][str(h)])
    assert data.positions["train"].min() >= 19  # full 20-day receptive field
    assert np.isfinite(data.y[data.positions["refit"]]).all()
    starts = window_starts(data.positions["train"], 5)
    assert np.isin(starts + 4, data.positions["train"]).all()


@pytest.mark.parametrize("fs", ["base", "ext"])
def test_end_to_end_schema_frozen_and_reproducible(synth_features, synth_splits, tmp_cfg, fs):
    split = synth_splits["covid"]
    cell = Cell("covid", fs, 5, 20)
    run_cell(cell, synth_features, split, tmp_cfg, plots=False)

    model_dir = Path(tmp_cfg["paths"]["models_dir"]) / "covid" / cell.name
    fc_path = Path(tmp_cfg["paths"]["forecasts_dir"]) / "covid" / f"{cell.name}.parquet"
    first = pd.read_parquet(fc_path)
    assert list(first.columns) == list(FORECAST_SCHEMA)
    validate_forecasts(first, split)
    assert (first["forecast_var"] == first["forecast_var_naive"]).all()  # level model: no smearing
    for name in ("params.json", "weights.pt", "val_curve.csv", "diagnostics.json"):
        assert (model_dir / name).exists(), name

    weights = model_dir / "weights.pt"
    original = weights.read_bytes()
    weights.write_bytes(original + b"\0")
    with pytest.raises(RuntimeError, match="changed after refit"):
        load_model(model_dir)
    weights.write_bytes(original)

    tmp_cfg["paths"]["models_dir"] = Path(tmp_cfg["paths"]["models_dir"]).with_name("models_rerun")
    run_cell(cell, synth_features, split, tmp_cfg, plots=False)
    pd.testing.assert_frame_equal(pd.read_parquet(fc_path), first)


def test_harnet_view_accepts_ext_alias(synth_features, synth_splits):
    a = adapters.harnet_view(synth_features, synth_splits["covid"], "train", "ext")
    assert list(a.columns) == ["rv", "rs_neg", "delta_j2"]
