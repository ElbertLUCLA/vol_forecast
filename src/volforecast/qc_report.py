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


def make_figures(measures: pd.DataFrame, cfg: dict, out_dir):
    df = measures.sort_values("date")
    log_rv = np.log(df["rv"].clip(lower=1e-12))

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df["date"], log_rv, lw=0.6)
    for event_cfg in cfg["events"].values():
        ax.axvspan(pd.Timestamp(event_cfg["event_window"][0]), pd.Timestamp(event_cfg["event_window"][1]),
                   color="red", alpha=0.15)
    ax.set_title("log RV, SPY (event windows shaded)")
    ax.set_ylabel("log RV")
    fig.tight_layout()
    fig.savefig(out_dir / "log_rv_timeseries.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(log_rv.dropna(), bins=60)
    ax.set_title("Histogram of log RV")
    fig.tight_layout()
    fig.savefig(out_dir / "log_rv_histogram.png", dpi=150)
    plt.close(fig)

    from statsmodels.graphics.tsaplots import plot_acf

    fig, ax = plt.subplots(figsize=(8, 4))
    plot_acf(log_rv.dropna(), lags=60, ax=ax)
    ax.set_title("ACF of log RV")
    fig.tight_layout()
    fig.savefig(out_dir / "log_rv_acf.png", dpi=150)
    plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
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

    make_figures(measures, cfg, qc_dir)
    logger.info("QC report written to %s", qc_dir)


if __name__ == "__main__":
    main()
