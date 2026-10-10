"""Descriptive statistics (qc_report) and the HAR-level forecasts (harnet.har_level) on synthetic data."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import PIPE_CFG, synthetic_measures
from volforecast.qc_report import descriptive_stats


def test_descriptive_stats_matches_feature_store(synth_features):
    m = synthetic_measures()
    s = descriptive_stats(m, PIPE_CFG["realized_measures"]["zero_rv_floor_percentile"]).set_index("series")
    assert list(s.index) == ["rv", "ln_rv", "rs_neg", "jump_pct", "delta_j2"]
    f = synth_features.sort_values("date").reset_index(drop=True)
    assert s.loc["ln_rv", "mean"] == pytest.approx(f["log_rv_d"].mean())
    assert s.loc["jump_pct", "sd"] == pytest.approx(f["jump_pct"].std())
    assert s.loc["rv", "acf_5"] == pytest.approx(m["rv"].autocorr(5))
    assert s.loc["rv", "excess_kurtosis"] == pytest.approx(m["rv"].kurt())
    assert s.loc["rv", "ann_mean_vol_pct"] == pytest.approx(np.sqrt(252 * m["rv"].mean()) * 100)
    assert np.isnan(s.loc["ln_rv", "ann_mean_vol_pct"])


def test_har_level_equals_ols_at_cutoff(synth_features, synth_splits, tmp_path):
    from volforecast.models.harnet import har_level
    from volforecast.models.harnet.config import Cell, load_harnet_config
    from volforecast.models.harnet.data import MinMaxScaler, build_series
    from volforecast.models.harnet.model import har_regressors, ols_fit

    cfg = load_harnet_config()
    cfg["paths"].update(forecasts_dir=tmp_path / "forecasts", models_dir=tmp_path / "harnet")
    cell = Cell("covid", "ext", 5, max(cfg["lags"]))
    out = har_level.run_cell(cell, synth_features, synth_splits["covid"], cfg, models_dir=tmp_path / "init")

    assert (tmp_path / "forecasts/covid/harnet20_init_ext_h5.parquet").exists()
    assert set(out["model"]) == {"harnet20_init_ext_h5"}
    fs = cfg["featuresets"]["ext"]
    data = build_series(synth_features.sort_values("date").reset_index(drop=True), synth_splits["covid"],
                        fs["channels"], 5, cell.max_lag, MinMaxScaler(cfg["scaler"]["min"], cfg["scaler"]["max"]),
                        cfg["train_tail_days"])
    coef = ols_fit(data.x, data.y, data.positions["refit"], cfg["lags"], 2, 1)
    pos = np.array(sorted(data.samples))
    clip_min = 0.5 * data.x[data.positions["refit"], 0].min()
    expected = data.scaler.inverse_transform(np.maximum(har_regressors(data.x, cfg["lags"], 2, 1)[pos] @ coef, clip_min))
    np.testing.assert_allclose(out["forecast_var"].to_numpy(), expected, rtol=1e-10)
    assert (out["forecast_var"] == out["forecast_var_seed_min"]).all()
    assert (out.loc[out["sample"] != "train_tail", "date"] > pd.Timestamp(synth_splits["covid"]["val"]["end"])).all()
