"""Stage 7: quality-control checks and figures for the data chapter (plan.md section 5)."""
from __future__ import annotations

import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from volforecast.config import ensure_parent, load_config, resolve_path

logger = logging.getLogger("volforecast.qc_report")


def run_checks(measures: pd.DataFrame, calendar: pd.DataFrame) -> dict:
    checks = {}

    normal = measures.loc[~measures["is_early_close"]]
    checks["pct_normal_days_full_grid"] = float((normal["n_bars_observed"] == 78).mean() * 100) if len(normal) else float("nan")
    early = measures.loc[measures["is_early_close"]]
    checks["pct_early_close_days_full_grid"] = float((early["n_bars_observed"] == 42).mean() * 100) if len(early) else float("nan")

    cal_dates = set(pd.to_datetime(calendar["date"]))
    data_dates = set(pd.to_datetime(measures["date"]))
    checks["n_calendar_days_missing_from_data"] = len(cal_dates - data_dates)
    checks["n_data_days_not_in_calendar"] = len(data_dates - cal_dates)

    checks["median_abs_pct_diff_rth_vs_daily_close"] = float(measures["rth_vs_daily_close_abs_pct_diff"].median())

    resid = (measures["rs_pos"] + measures["rs_neg"] - measures["rv"]).abs()
    checks["max_abs_semivariance_identity_error"] = float(resid.max())

    ann_vol = float(np.sqrt(252 * measures["rv"].mean()) * 100)
    checks["annualized_mean_vol_pct"] = ann_vol
    checks["plausible_annualized_vol"] = bool(13 <= ann_vol <= 17)

    return checks


def top_outliers(measures: pd.DataFrame, n: int = 20) -> dict[str, pd.DataFrame]:
    cols = ["date", "rv", "max_abs_5min_return", "is_early_close"]
    return {
        "top_rv": measures.nlargest(n, "rv")[cols],
        "top_abs_return": measures.nlargest(n, "max_abs_5min_return")[cols],
    }


def descriptive_stats(measures: pd.DataFrame, floor_percentile: float = 0.001) -> pd.DataFrame:
    """Table 3.1: full-sample summary of the daily series the models use.

    RV, ln RV, RS- (daily decimal variances), and the jump measure in both forms used by the models:
    jump_pct = ln(1 + dJ2 / RV) (HAR, XGBoost, LSTM, Mamba; Patton & Sheppard 2015) and the raw signed
    jump dJ2 = RS+ - RS- (HARNet). Skewness and excess kurtosis are pandas' bias-corrected estimators;
    ACF is the sample autocorrelation at lags 1, 5 and 22. ann_mean_vol_pct = sqrt(252 * mean) * 100,
    only for the variance series. ln RV and jump_pct use the feature store's floor (RV clipped at its
    `floor_percentile` quantile, config realized_measures.zero_rv_floor_percentile), so they equal
    log_rv_d and jump_pct in spy_features.parquet; RV, RS- and dJ2 are unfloored.
    """
    from volforecast.build_features import _floor_and_log

    df = measures.sort_values("date")
    rv = df["rv"].astype(float)
    ln_rv, floor = _floor_and_log(rv, floor_percentile, "rv")
    series = {
        "rv": rv,
        "ln_rv": ln_rv,
        "rs_neg": df["rs_neg"].astype(float),
        "jump_pct": np.log(1.0 + df["delta_j2"] / rv.clip(lower=floor)),
        "delta_j2": df["delta_j2"].astype(float),
    }
    rows = []
    for name, s in series.items():
        s = s.reset_index(drop=True)
        rows.append({
            "series": name, "n": int(s.notna().sum()), "mean": s.mean(), "sd": s.std(), "min": s.min(),
            "max": s.max(), "skewness": s.skew(), "excess_kurtosis": s.kurt(),
            "acf_1": s.autocorr(1), "acf_5": s.autocorr(5), "acf_22": s.autocorr(22),
            "ann_mean_vol_pct": float(np.sqrt(252 * s.mean()) * 100) if name in ("rv", "rs_neg") else np.nan,
            "start": df["date"].min().date(), "end": df["date"].max().date(),
        })
    return pd.DataFrame(rows)


def make_figures(measures: pd.DataFrame, cfg: dict, out_dir):
    """log RV time series, histogram and ACF in the paper format (evaluation_plots): 6.5 in wide,
    text >= 9 pt, events shaded peak -> trough as in every other figure; PNG and vector PDF."""
    from volforecast import evaluation_plots as P

    P.use_paper_style()
    df = measures.sort_values("date")
    log_rv = np.log(df["rv"].clip(lower=1e-12))

    fig, ax = plt.subplots(figsize=(P.FIG_WIDTH, 2.8))
    P.style(ax, "ln RV")
    for event in cfg["events"]:
        P.shade_event(ax, event)
    ax.plot(df["date"], log_rv, lw=0.5, color=P.INK, label="ln RV (daily)")
    P.date_axis(ax)
    ax.set_title("SPY daily ln RV", loc="left", color=P.INK)
    P.finish(fig, out_dir / "log_rv_timeseries")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(P.FIG_WIDTH, 2.8))
    P.style(ax, "days")
    ax.hist(log_rv.dropna(), bins=60, color=P.MUTED)
    ax.set_xlabel("ln RV", color=P.MUTED)
    ax.set_title("Histogram of daily ln RV", loc="left", color=P.INK)
    P.finish(fig, out_dir / "log_rv_histogram")
    plt.close(fig)

    from statsmodels.graphics.tsaplots import plot_acf

    fig, ax = plt.subplots(figsize=(P.FIG_WIDTH, 2.8))
    plot_acf(log_rv.dropna(), lags=60, ax=ax, color=P.INK, vlines_kwargs={"colors": P.INK})
    P.style(ax, "autocorrelation")
    ax.set_xlabel("lag (trading days)", color=P.MUTED)
    ax.set_title("ACF of daily ln RV", loc="left", color=P.INK)
    P.finish(fig, out_dir / "log_rv_acf")
    plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("fontTools").setLevel(logging.WARNING)  # PDF font subsetting chatter
    cfg = load_config()
    measures_path = resolve_path(cfg["paths"]["daily_measures_path"])
    calendar_path = resolve_path(cfg["paths"]["calendar_path"])
    qc_dir = resolve_path(cfg["paths"]["qc_dir"])
    qc_dir.mkdir(parents=True, exist_ok=True)

    measures = pd.read_parquet(measures_path)
    calendar = pd.read_parquet(calendar_path)

    checks = run_checks(measures, calendar)
    checks_df = pd.DataFrame([checks]).T.rename(columns={0: "value"})
    checks_df.to_csv(qc_dir / "checks_summary.csv")
    logger.info("QC checks:\n%s", checks_df.to_string())

    outliers = top_outliers(measures)
    for name, tbl in outliers.items():
        ensure_parent(qc_dir / f"{name}.csv")
        tbl.to_csv(qc_dir / f"{name}.csv", index=False)

    stats = descriptive_stats(measures, cfg["realized_measures"]["zero_rv_floor_percentile"])
    stats.to_csv(qc_dir / "descriptive_stats.csv", index=False)
    logger.info("Descriptive statistics:\n%s", stats.to_string())

    make_figures(measures, cfg, qc_dir)
    logger.info("QC report written to %s", qc_dir)


if __name__ == "__main__":
    main()
