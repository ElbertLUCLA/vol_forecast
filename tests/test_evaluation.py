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


def test_family_har_level_not_harnet():
    assert E.family("harnet20_init_ext_h5") == "HAR-level"
    assert E.family("harnet20_ext_h5") == "HARNet"


def test_lead_days_positive_before_peak():
    """lead_days = peak position - first crossing: a crossing 3 trading days before the peak gives +3."""
    dates = pd.bdate_range("2020-02-03", "2020-03-23")
    peak = list(dates).index(pd.Timestamp(E.PEAKS["covid"]))
    lift = np.ones(len(dates))
    lift[peak - 3:] = 2.5
    df = pd.DataFrame({"event": "covid", "model": "har_base_h1", "family": "HAR", "fs": "base", "h": 1,
                       "date": dates, "sample": "event", "lift": lift})
    lt = E.lead_times(df, thresholds=(2.0, 3.0))
    assert lt["lead_days_2.0x"].iloc[0] == 3
    assert lt["lead_days_3.0x"].iloc[0] is None or np.isnan(lt["lead_days_3.0x"].iloc[0])


def test_ext_vs_base():
    acc = pd.DataFrame({"event": "covid", "family": "HAR", "fs": ["base", "ext", "base", "-"],
                        "h": 5, "sample": ["event", "event", "calm", "event"], "qlike": [0.5, 0.4, 9.0, 7.0]})
    acc.loc[3, "family"] = "GARCH"
    lt = pd.DataFrame({"event": "covid", "family": ["HAR", "HAR", "GARCH"], "fs": ["base", "ext", "-"], "h": 5,
                       "lead_days_2.0x": [None, -4, 1]})
    al = pd.DataFrame({"event": "covid", "family": ["HAR", "HAR", "HAR", "GARCH"], "fs": ["base", "ext", "ext", "-"],
                       "h": 5, "tau": [2.0, 2.0, 3.0, 2.0], "peirce": [0.2, 0.3, 0.9, 0.5]})
    t = E.ext_vs_base(acc, lt, al, tau=2.0)
    assert list(t["family"]) == ["HAR"]
    row = t.iloc[0]
    assert row["qlike_event_base"] == 0.5 and row["qlike_event_ext"] == 0.4
    assert row["qlike_event_diff"] == pytest.approx(-0.1)
    assert np.isnan(row["lead_days_2.0x_base"]) and row["lead_days_2.0x_ext"] == -4
    assert row["peirce_tau2_base"] == 0.2 and row["peirce_tau2_ext"] == 0.3


def _write_json(path, obj):
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def test_ceilings_reads_xgb_and_seq_keys(tmp_path):
    _write_json(tmp_path / "xgb/covid/xgb-gbtree_ext_h22/diagnostics.json",
                {"ceiling": {"ceiling_var": 4e-4, "max_train_target_var": 1e-3},
                 "days_realized_above_ceiling": {"pre_event": 1, "event": 7, "post_event": 0}})
    _write_json(tmp_path / "xgb/covid/xgb-gblinear_ext_h22/diagnostics.json",
                {"ceiling": {"bound_log": None, "max_train_target_var": 1e-3}})
    _write_json(tmp_path / "seq/bear2022/mamba_base_h5/diagnostics.json",
                {"ceiling_var": 2e-3, "max_train_target_log": np.log(3e-3), "event_days_realized_above_ceiling": 0})
    c = E.ceilings(tmp_path).set_index("model")
    assert set(c.index) == {"xgb-gbtree_ext_h22", "mamba_base_h5"}  # gblinear has no ceiling
    x, m = c.loc["xgb-gbtree_ext_h22"], c.loc["mamba_base_h5"]
    assert (x["family"], x["fs"], x["h"], x["event_days_above_ceiling"]) == ("XGB-tree", "ext", 22, 7)
    assert x["ceiling_vol_pct"] == pytest.approx(np.sqrt(252 * 4e-4) * 100)
    assert m["max_train_target_var"] == pytest.approx(3e-3)
    assert m["max_train_target_vol_pct"] == pytest.approx(np.sqrt(252 * 3e-3) * 100)


def test_selection_summary_cap_flag(tmp_path):
    (tmp_path / "harnet").mkdir()
    pd.DataFrame({"event": "covid", "model": ["harnet20_base_h1", "harnet20_ext_h1"], "featureset": ["base", "ext"],
                  "h": 1, "best_iter": [500, 9750], "stopped_at": [8500, 10000]}).to_csv(
        tmp_path / "harnet/run_log.csv", index=False)
    for m in ("harnet20_base_h1", "harnet20_ext_h1"):
        _write_json(tmp_path / f"harnet/covid/{m}/params.json", {"max_iterations": 10000})
    (tmp_path / "seq").mkdir()
    pd.DataFrame({"event": ["covid"], "model": ["lstm_base_h1"], "featureset": ["base"], "h": [1], "lookback": [66],
                  "best_epochs": ["[3, 5, 4]"]}).to_csv(tmp_path / "seq/run_log.csv", index=False)
    (tmp_path / "xgb").mkdir()
    pd.DataFrame({"event": ["covid"], "model": ["xgb-gbtree_base_h1"], "featureset": ["base"], "h": [1],
                  "best_trial": [11], "n_trials": [100]}).to_csv(tmp_path / "xgb/run_log.csv", index=False)
    s = E.selection_summary(tmp_path).set_index("model")
    assert not s.loc["harnet20_base_h1", "cap_bound"] and s.loc["harnet20_ext_h1", "cap_bound"]
    assert s.loc["lstm_base_h1", "best_epochs_median"] == 4
    assert s.loc["xgb-gbtree_base_h1", "best_trial"] == 11
