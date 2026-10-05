"""Interpretability (section 7) and per-cell figures.

- gbtree: total-gain importance (Teller et al. Table C.7), averaged over seeds, in percent.
- gblinear: coefficients on standardized features next to HAR-OLS on the same scaling.
- SHAP contributions over train_tail + forecast period via pred_contribs
  (built into xgboost): contribs.parquet, plotted for the headline horizons.
- Event plot: forecast vs realized, seed band, tree ceiling, peak/trough.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from volforecast.models.xgb.data import CellFrames  # noqa: E402
from volforecast.models.xgb.diagnostics import gblinear_coefficients, ols_coefficients  # noqa: E402
from volforecast.models.xgb.fit import FittedCell, dmatrix  # noqa: E402

# Categorical slots in fixed order (dataviz reference palette, light mode).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def importance(fitted: FittedCell, frames: CellFrames, model_dir: Path) -> pd.DataFrame:
    if fitted.cell.booster == "gbtree":
        gains = pd.DataFrame(
            [m.get_score(importance_type="total_gain") for m in fitted.models], columns=fitted.features
        ).fillna(0.0)
        pct = gains.div(gains.sum(axis=1), axis=0) * 100
        table = pd.DataFrame({"gain_pct": pct.mean(), "gain_pct_seed_std": pct.std(ddof=0)})
    else:
        coef = gblinear_coefficients(fitted.models[0], fitted.features)
        ols = ols_coefficients(frames.refit.X, frames.refit.y, fitted.scaler)
        table = pd.DataFrame({"coef_std_units": coef, "har_ols_coef_std_units": ols})
    table.index.name = "feature"
    table.to_csv(Path(model_dir) / "importance.csv")
    return table


def contributions(fitted: FittedCell, frames: CellFrames, model_dir: Path) -> pd.DataFrame:
    """Per-day additive contributions to the log forecast, averaged over seeds."""
    rows = pd.concat([frames.tail.data, frames.oos.data])
    d = dmatrix(rows[fitted.features], scaler=fitted.scaler)
    contribs = np.mean([m.predict(d, pred_contribs=True) for m in fitted.models], axis=0)
    out = pd.DataFrame(contribs, columns=fitted.features + ["bias"], index=pd.DatetimeIndex(rows.index, name="date"))
    out["sample"] = rows["sample"].astype(str).to_numpy()
    out.to_parquet(Path(model_dir) / "contribs.parquet")
    return out


def _style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _mark_event(ax, split: dict) -> None:
    ax.axvspan(pd.Timestamp(split["event"]["start"]), pd.Timestamp(split["event"]["end"]),
               color=GRID, alpha=0.5, linewidth=0, label="event window")
    ax.axvline(pd.Timestamp(split["peak"]), color=INK_MUTED, linewidth=1, linestyle=":")
    ax.text(pd.Timestamp(split["peak"]), 1.0, " peak", transform=ax.get_xaxis_transform(),
            va="top", fontsize=8, color=INK_MUTED)


def plot_forecast(forecasts: pd.DataFrame, split: dict, ceiling_var: float | None, title: str, path: Path) -> None:
    df = forecasts[forecasts["sample"] != "train_tail"]
    fig, ax = plt.subplots(figsize=(9, 4))
    _style(ax)
    _mark_event(ax, split)
    if df["forecast_var_seed_min"].notna().any():
        ax.fill_between(df["date"], df["forecast_var_seed_min"], df["forecast_var_seed_max"],
                        color=SERIES[0], alpha=0.2, linewidth=0, label="seed range")
    ax.plot(df["date"], df["realized_var"], color=INK_MUTED, linewidth=1.2, label="realized")
    ax.plot(df["date"], df["forecast_var"], color=SERIES[0], linewidth=2, label="forecast (smeared)")
    if ceiling_var is not None:
        ax.axhline(ceiling_var, color=SERIES[1], linewidth=1.5, linestyle="--", label="tree ceiling")
    ax.set_yscale("log")
    ax.set_ylabel(f"mean daily RV over next {int(df['h'].iloc[0])} days", fontsize=9, color=INK_MUTED)
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_contributions(contribs: pd.DataFrame, split: dict, title: str, path: Path) -> None:
    """What pushed the log forecast up or down, day by day, before and during the event."""
    end = pd.Timestamp(split["event"]["end"])
    df = contribs[(contribs["sample"] != "post_event") & (contribs.index <= end)]
    features = [c for c in contribs.columns if c not in ("bias", "sample")]
    fig, ax = plt.subplots(figsize=(9, 4))
    _style(ax)
    _mark_event(ax, split)
    ax.axhline(0, color=INK_MUTED, linewidth=0.8)
    ax.axvline(pd.Timestamp(split["val"]["end"]), color=INK_MUTED, linewidth=1, linestyle="--")
    ax.text(pd.Timestamp(split["val"]["end"]), 0.0, " cutoff", transform=ax.get_xaxis_transform(),
            va="bottom", fontsize=8, color=INK_MUTED)
    for color, feat in zip(SERIES, features):
        ax.plot(df.index, df[feat], color=color, linewidth=2, label=feat)
    ax.set_ylabel("contribution to log forecast", fontsize=9, color=INK_MUTED)
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def importance_summary(models_dir: Path) -> pd.DataFrame | None:
    """Stack every cell's importance.csv into one long table (features x cells)."""
    parts = []
    for path in sorted(Path(models_dir).glob("*/*/importance.csv")):
        table = pd.read_csv(path)
        table["event"], table["model"] = path.parent.parent.name, path.parent.name
        parts.append(table)
    if not parts:
        return None
    summary = pd.concat(parts, ignore_index=True)
    summary.to_csv(Path(models_dir) / "importance_summary.csv", index=False)
    return summary
