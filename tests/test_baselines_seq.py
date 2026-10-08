"""Tests for HAR, GARCH and the sequence models (Mamba / LSTM), on synthetic data.

Mamba is only checked for forward pass and output bound here (no training), so the test
suite stays fast on CPU.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from volforecast.models.baseline_config import load_baseline_config
from volforecast.models.schema import FORECAST_SCHEMA, validate_forecasts


@pytest.fixture()
def paths(tmp_path):
    return {"models_dir": tmp_path / "models", "forecasts_dir": tmp_path / "forecasts"}


def test_har_end_to_end(synth_features, synth_splits, paths):
    from volforecast.models.har.run_all import run_cell

    cfg = load_baseline_config("har")
    cfg["paths"].update(paths)
    row = run_cell("covid", "base", 5, synth_features, synth_splits["covid"], cfg)
    fc = pd.read_parquet(paths["forecasts_dir"] / "covid" / "har_base_h5.parquet")
    validate_forecasts(fc, synth_splits["covid"])
    assert "calm" in set(fc["sample"])
    assert row["val_qlike"] < row["val_qlike_rw"]  # the synthetic process is HAR, so HAR must beat RW
    # forecast-only reproduces the file from the saved coefficients
    run_cell("covid", "base", 5, synth_features, synth_splits["covid"], cfg, forecast_only=True)
    pd.testing.assert_frame_equal(pd.read_parquet(paths["forecasts_dir"] / "covid" / "har_base_h5.parquet"), fc)


def test_garch_end_to_end(synth_features, synth_splits, paths):
    from volforecast.models.garch.run_all import run_event

    cfg = load_baseline_config("garch")
    cfg["paths"].update(paths)
    feats = synth_features.assign(ret_cc_pct=synth_features["ret_cc"] * 100)
    rows = run_event("covid", feats, synth_splits["covid"], cfg)
    assert [r["model"] for r in rows] == ["garch_h1", "garch_h5", "garch_h22"]
    assert all(0 < r["persistence"] < 1 for r in rows)
    for h in (1, 5, 22):
        fc = pd.read_parquet(paths["forecasts_dir"] / "covid" / f"garch_h{h}.parquet")
        validate_forecasts(fc, synth_splits["covid"])
        assert (fc["forecast_var_naive"] > 0).all()


@pytest.mark.parametrize("arch", ["mamba", "lstm"])
def test_output_ceiling_is_an_upper_bound(arch):
    from volforecast.models.seq.model import SeqRegressor

    torch.manual_seed(0)
    m = SeqRegressor(arch, 3, 16, 2, {"state_size": 16, "expand": 2, "conv_kernel": 4}).eval()
    with torch.no_grad():
        out = m(torch.randn(256, 22, 3) * 50)  # extreme inputs
    assert float(out.max()) <= m.output_ceiling() + 1e-5


def test_lstm_end_to_end(synth_features, synth_splits, tmp_path):
    from volforecast.models.seq.data import Cell, build_seq_data, load_seq_config
    from volforecast.models.seq.run_all import forecast_cell, run_cell

    cfg = load_seq_config()
    cfg["paths"].update(models_dir=tmp_path / "models", forecasts_dir=tmp_path / "forecasts")
    cfg["tuning"]["lookback"] = [5, 10]
    cfg["train"].update(max_epochs=2, patience=1)
    cfg["seeds"] = [42, 43]
    split = synth_splits["covid"]
    cell = Cell("covid", "lstm", "ext", 5)
    run_cell(cell, synth_features, split, cfg, "cpu")

    fc = pd.read_parquet(tmp_path / "forecasts" / "covid" / "lstm_ext_h5.parquet")
    assert list(fc.columns) == list(FORECAST_SCHEMA)
    validate_forecasts(fc, split)
    assert (fc["forecast_var_seed_min"] <= fc["forecast_var"] * (1 + 1e-9)).all()

    # Frozen: reloading the saved weights reproduces the forecasts; tampering is detected.
    data = build_seq_data(synth_features, split, cfg["featuresets"]["ext"], 5, 10)
    model_dir = tmp_path / "models" / "covid" / cell.name
    pd.testing.assert_frame_equal(forecast_cell(cell, data, model_dir, cfg, "cpu"), fc)
    weights = model_dir / "weights.pt"
    weights.write_bytes(weights.read_bytes() + b"\0")
    with pytest.raises(RuntimeError, match="changed after refit"):
        forecast_cell(cell, data, model_dir, cfg, "cpu")


def test_seq_windows_are_causal(synth_features, synth_splits):
    from volforecast.models.seq.data import build_seq_data

    data = build_seq_data(synth_features, synth_splits["covid"], ["log_rv_d"], 5, 22)
    pos = data.positions["val"][:3]
    w = data.windows(pos, 22)
    assert w.shape == (3, 22, 1)
    np.testing.assert_array_equal(w[:, -1, 0], data.x[pos, 0])  # last step is day t itself
    assert data.positions["train"].min() >= 21
