"""Loss columns and DM tests in volforecast.evaluation, on a small hand-made forecast frame."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from volforecast import evaluation as E


def forecast_frame(seed: int = 0, n: int = 60) -> pd.DataFrame:
    """Two models (HAR-base benchmark + one challenger) at h = 5, split over the event and calm windows."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-02-03", periods=n)
    rv = np.exp(rng.normal(-9.5, 0.6, n))
    parts = []
    for model, family, fs, noise in (("har_base_h5", "HAR", "base", 0.3), ("xgb-gbtree_ext_h5", "XGB-tree", "ext", 0.5)):
        fc = rv * np.exp(rng.normal(0, noise, n))
        parts.append(pd.DataFrame({
            "event": "covid", "model": model, "family": family, "fs": fs, "h": 5, "date": dates,
            "sample": np.where(np.arange(n) < n // 2, "event", "calm"), "target_complete": True,
            "realized_var": rv, "forecast_var": fc,
            "realized_log": np.log(rv), "forecast_log": np.log(fc), "log_ratio": np.log(fc / rv),
            "coverage": fc / rv, "lift": fc / rv.mean(),
        }))
    return pd.concat(parts, ignore_index=True)


def test_accuracy_table_variance_scale_losses():
    df = forecast_frame()
    acc = E.accuracy_table(df)
    for _, row in acc.iterrows():
        g = df[(df["model"] == row["model"]) & (df["sample"] == row["sample"])]
        r, f = g["realized_var"].to_numpy(), g["forecast_var"].to_numpy()
        assert row["mse_var"] == pytest.approx(np.mean(((r - f) * 1e4) ** 2))
        assert row["mae_var"] == pytest.approx(np.mean(np.abs(r - f) * 1e4))
    bench = acc[acc["model"] == "har_base_h5"]
    assert np.allclose(bench["mse_var_rel"], 1.0)


def test_tracking_table_has_mse_var():
    df = forecast_frame()
    t = E.tracking_table(df, "covid")
    g = df[(df["model"] == "har_base_h5") & (df["sample"] == "event")]
    expected = np.mean(((g["realized_var"] - g["forecast_var"]) * 1e4) ** 2)
    assert t.set_index("model").loc["har_base_h5", "mse_var"] == pytest.approx(expected)


def test_dm_mse_var_zero_when_forecasts_equal_benchmark():
    df = forecast_frame()
    bench = df[df["model"] == "har_base_h5"]
    clone = bench.assign(model="clone_h5", family="XGB-tree", fs="ext")
    dm = E.dm_vs_benchmark(pd.concat([df, clone], ignore_index=True), loss="mse_var")
    row = dm.set_index("model").loc["clone_h5"]
    assert row["loss"] == "mse_var"
    assert row["mse_var"] == pytest.approx(row["mse_var_har_base"])
    assert row["dm_stat"] == 0 and row["p_value"] == 1


def test_dm_default_is_qlike():
    df = forecast_frame()
    dm = E.dm_vs_benchmark(df)
    d = df[df["sample"] == "event"]
    q = lambda m: E._qlike(d.loc[d["model"] == m, "realized_var"], d.loc[d["model"] == m, "forecast_var"])
    a, b = q("xgb-gbtree_ext_h5").to_numpy(), q("har_base_h5").to_numpy()
    row = dm.iloc[0]
    assert row["loss"] == "qlike"
    assert row["qlike"] == pytest.approx(a.mean())
    assert row["qlike_har_base"] == pytest.approx(b.mean())
    assert (row["dm_stat"], row["p_value"]) == pytest.approx(E.dm_test(a, b, 5))
    with pytest.raises(ValueError):
        E.dm_vs_benchmark(df, loss="mse_log")

