"""Smoke tests: can GARCH(1,1) and HAR actually be fit on the pipeline's transformed output?

These are integration tests against the real pipeline artifacts (data/processed/...).
They're skipped if the pipeline hasn't been run yet, rather than failing, since
this repo ships no fixture data (SPY history must be downloaded from the Massive API).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from volforecast import adapters  # noqa: E402
from volforecast.config import load_config, resolve_path  # noqa: E402

CFG = load_config()
FEATURES_PATH = resolve_path(CFG["paths"]["features_path"])
SPLITS_DIR = resolve_path(CFG["paths"]["splits_dir"])

pytestmark = pytest.mark.skipif(
    not FEATURES_PATH.exists() or not (SPLITS_DIR / "covid.json").exists(),
    reason="Pipeline has not been run yet (data/processed/spy_features.parquet missing). "
    "Run `uv run python scripts/run_pipeline.py` first.",
)


@pytest.fixture(scope="module")
def features_df():
    return adapters.load_features(FEATURES_PATH)


@pytest.fixture(scope="module")
def covid_split():
    return adapters.load_split("covid", SPLITS_DIR)


def test_garch_view_fits(features_df, covid_split):
    """GARCH(1,1) on ret_cc_pct should converge and produce finite forecasts."""
    arch_model = pytest.importorskip("arch").arch_model

    view = adapters.garch_view(features_df, covid_split)
    returns = view["train_returns"].dropna()
    assert len(returns) > 500, "expected a multi-year training window"

    am = arch_model(returns, vol="Garch", p=1, q=1, dist="normal", rescale=False)
    res = am.fit(disp="off")

    assert res.convergence_flag == 0
    omega, alpha, beta = res.params["omega"], res.params["alpha[1]"], res.params["beta[1]"]
    assert omega > 0
    assert alpha >= 0 and beta >= 0
    assert alpha + beta < 1.0  # covariance-stationarity

    n_oos = len(view["oos_dates"])
    fcast = res.forecast(horizon=n_oos, reindex=False)
    variance_pct2 = fcast.variance.to_numpy().flatten()
    assert np.all(np.isfinite(variance_pct2))
    assert np.all(variance_pct2 > 0)

    print(f"\nGARCH(1,1) on {len(returns)} obs: omega={omega:.4g} alpha={alpha:.4g} beta={beta:.4g}")


def test_har_view_fits_and_has_sane_coefficients(features_df, covid_split):
    """HAR-OLS on the base feature set should give the textbook HAR coefficient pattern."""
    X, y = adapters.tabular_view(features_df, covid_split, segment="train", featureset="base", h=1)
    assert len(X) > 500, "expected a multi-year training window"
    assert list(X.columns) == adapters.BASE_FEATURES
    assert not X.isna().any().any()
    assert not y.isna().any()

    X_design = sm.add_constant(X)
    model = sm.OLS(y, X_design).fit()

    beta_d = model.params["log_rv_d"]
    beta_w = model.params["log_rv_w"]
    beta_m = model.params["log_rv_m"]
    print(f"\nHAR-OLS (h=1, n={len(X)}): const={model.params['const']:.4g} "
          f"beta_d={beta_d:.4g} beta_w={beta_w:.4g} beta_m={beta_m:.4g} R2={model.rsquared:.3f}")

    # All three HAR slopes should be positive and roughly sum towards persistence < 1,
    # consistent with the usual Corsi (2009) HAR-RV estimates (plan.md checklist).
    assert beta_d > 0
    assert beta_w > 0
    assert beta_m > 0
    assert model.rsquared > 0.3


def test_tabular_view_purges_train_val_boundary(features_df, covid_split):
    """Train rows for h=22 must not have target windows crossing the purged cutoff."""
    X1, _ = adapters.tabular_view(features_df, covid_split, segment="train", featureset="base", h=1)
    X22, _ = adapters.tabular_view(features_df, covid_split, segment="train", featureset="base", h=22)
    assert X22.index.max() <= X1.index.max()
    assert (X1.index.max() - X22.index.max()) >= pd.Timedelta(days=21)


def test_harnet_view_matches_harnet_input_convention(features_df, covid_split):
    """harnet_view's extended column order must match HARNet's HARSVJ input convention."""
    base = adapters.harnet_view(features_df, covid_split, segment="train", featureset="base")
    assert list(base.columns) == ["rv"]
    assert (base["rv"] > 0).all()

    ext = adapters.harnet_view(features_df, covid_split, segment="train", featureset="extended")
    assert list(ext.columns) == ["rv", "rs_neg", "delta_j2"]


def test_sequence_view_shape(features_df, covid_split):
    lags = CFG["sequence_view"]["lags"]
    Xseq, yseq, dates = adapters.sequence_view(
        features_df, covid_split, segment="train", featureset="base", h=1, lags=lags
    )
    assert Xseq.shape[0] == len(yseq) == len(dates)
    assert Xseq.shape[1:] == (lags, len(adapters.BASE_FEATURES))
    assert np.isfinite(Xseq).all()
    assert np.isfinite(yseq).all()
