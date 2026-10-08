"""Figures for the model-comparison notebook (notebooks/02_model_comparison.ipynb).

Model families keep the same color in every figure. Realized variance is drawn in ink,
and variances are shown as annualized volatility in % (sqrt(252 * var) * 100).
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from volforecast.evaluation import FAMILY_ORDER, PEAKS, TROUGHS

# Fixed color per family (same palette as the XGBoost figures); color follows the model, never its rank.
FAMILY_COLORS = {"GARCH": "#2a78d6", "HAR": "#eb6834", "XGB-linear": "#1baf7a", "XGB-tree": "#eda100",
                 "HARNet": "#e87ba4", "LSTM": "#e34948", "Mamba": "#4a3aa7"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def ann_vol(var):
    return np.sqrt(252 * np.asarray(var, dtype=float)) * 100


def style(ax, ylabel: str | None = None):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color=MUTED)


def mark_event(ax, event: str, split: dict | None = None):
    ax.axvspan(pd.Timestamp(PEAKS[event]), pd.Timestamp(TROUGHS[event]), color=GRID, alpha=0.6, lw=0,
               label="peak → trough")
    ax.axvline(pd.Timestamp(PEAKS[event]), color=MUTED, lw=1, ls=":")
    if split is not None:
        ax.axvline(pd.Timestamp(split["val"]["end"]), color=MUTED, lw=1, ls="--")
        ax.text(pd.Timestamp(split["val"]["end"]), 0.98, " cutoff", transform=ax.get_xaxis_transform(),
                va="top", fontsize=7, color=MUTED)


def representatives(df: pd.DataFrame, fs: str = "ext") -> pd.DataFrame:
    """One cell per family: the given feature set (GARCH has none)."""
    return df[(df["fs"] == fs) | (df["family"] == "GARCH")]


def plot_forecasts(df, event, h, start, end, split=None, ax=None, title=None):
    d = representatives(df)
    d = d[(d["event"] == event) & (d["h"] == h) & d["date"].between(start, end)]
    ax = ax or plt.subplots(figsize=(10, 4))[1]
    style(ax, f"annualized vol over next {h} days (%)")
    mark_event(ax, event, split)
    real = d.drop_duplicates("date").sort_values("date")
    ax.plot(real["date"], ann_vol(real["realized_var"]), color=INK, lw=2.2, label="realized")
    for fam in FAMILY_ORDER:
        g = d[d["family"] == fam].sort_values("date")
        if len(g):
            ax.plot(g["date"], ann_vol(g["forecast_var"]), color=FAMILY_COLORS[fam], lw=1.6, label=fam)
    ax.set_title(title or f"{event}: forecasts vs realized, h = {h} (ext feature set)", loc="left",
                 fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left", bbox_to_anchor=(1.01, 1))
    return ax


def plot_lift(df, event, h, start, end, split=None, thresholds=(1.5, 2.0), ax=None):
    d = representatives(df)
    d = d[(d["event"] == event) & (d["h"] == h) & d["date"].between(start, end)]
    ax = ax or plt.subplots(figsize=(10, 3.6))[1]
    style(ax, "lift = forecast ÷ normal level")
    mark_event(ax, event, split)
    for tau in thresholds:
        ax.axhline(tau, color=MUTED, lw=0.8, ls="--")
        ax.text(d["date"].min(), tau, f" {tau}×", va="bottom", fontsize=7, color=MUTED)
    real = d.drop_duplicates("date").sort_values("date")
    ax.plot(real["date"], real["realized_lift"], color=INK, lw=2.2, label="realized")
    for fam in FAMILY_ORDER:
        g = d[d["family"] == fam].sort_values("date")
        if len(g):
            ax.plot(g["date"], g["lift"], color=FAMILY_COLORS[fam], lw=1.6, label=fam)
    ax.set_yscale("log")
    ax.set_title(f"{event}: early-warning lift, h = {h}", loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left", bbox_to_anchor=(1.01, 1))
    return ax


def bar_by_family(table: pd.DataFrame, value: str, ax=None, ref: float | None = None, title: str = "",
                  label_fmt: str = "{:.2f}"):
    """Grouped bars: one group per horizon, one bar per family (ext set; GARCH has none)."""
    t = table[(table["fs"].isin(["ext", "-"]))]
    hs = sorted(t["h"].unique())
    fams = [f for f in FAMILY_ORDER if f in set(t["family"])]
    ax = ax or plt.subplots(figsize=(10, 3.6))[1]
    style(ax, value)
    width = 0.8 / len(fams)
    for i, fam in enumerate(fams):
        vals = [t[(t["family"] == fam) & (t["h"] == h)][value].mean() for h in hs]
        x = np.arange(len(hs)) + i * width
        ax.bar(x, vals, width=width * 0.92, color=FAMILY_COLORS[fam], label=fam)
        for xi, v in zip(x, vals):
            if np.isfinite(v):
                ax.text(xi, v, label_fmt.format(v), ha="center", va="bottom", fontsize=6.5, color=MUTED)
    if ref is not None:
        ax.axhline(ref, color=INK, lw=1, ls="--")
    ax.set_xticks(np.arange(len(hs)) + 0.4 - width / 2, [f"h = {h}" for h in hs])
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=7.5, ncol=len(fams), loc="upper left", bbox_to_anchor=(0, -0.12))
    return ax


def plot_alarm_skill(alarms: pd.DataFrame, event: str, tau: float = 1.5, ax=None):
    """Hit rate vs false-alarm rate per model cell. The diagonal = no skill (random or always-alarm)."""
    a = alarms[(alarms["event"] == event) & (alarms["tau"] == tau)]
    ax = ax or plt.subplots(figsize=(5.2, 5))[1]
    style(ax, "hit rate  P(alarm | realized stress)")
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--")
    ax.text(0.62, 0.55, "no skill", rotation=45, fontsize=7, color=MUTED)
    markers = {1: "o", 5: "s", 22: "^"}
    for fam in FAMILY_ORDER:
        g = a[a["family"] == fam]
        for _, r in g.iterrows():
            ax.scatter(r["false_alarm_rate"], r["hit_rate"], s=46, color=FAMILY_COLORS[fam],
                       marker=markers.get(r["h"], "o"), edgecolor="white", linewidth=1.5,
                       alpha=1.0 if r["fs"] in ("ext", "-") else 0.45)
    ax.set_xlabel("false-alarm rate  P(alarm | no stress)", fontsize=9, color=MUTED)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    for fam in FAMILY_ORDER:
        if fam in set(a["family"]):
            ax.scatter([], [], color=FAMILY_COLORS[fam], label=fam, s=30)
    for h, m in markers.items():
        ax.scatter([], [], color=MUTED, marker=m, label=f"h = {h}", s=30)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    ax.set_title(f"{event}: alarm skill, forecast ≥ {tau:g}× normal (faded = base set)", loc="left", fontsize=10, color=INK)
    return ax


def plot_smearing_shift(comp: pd.DataFrame, value: str, h: int, ax=None, title: str = "", ref: float | None = None):
    """Paired bars per family: old split (validation with COVID, muted) vs new split (post-COVID, colored)."""
    t = comp[(comp["h"] == h) & comp["fs"].isin(["ext", "-"])]
    fams = [f for f in FAMILY_ORDER if f in set(t["family"])]
    ax = ax or plt.subplots(figsize=(8, 3.4))[1]
    style(ax, value)
    x = np.arange(len(fams))
    old = [t[t["family"] == f][f"{value}__val_with_covid"].mean() for f in fams]
    new = [t[t["family"] == f][f"{value}__val_post_covid"].mean() for f in fams]
    ax.bar(x - 0.2, old, width=0.38, color=GRID, edgecolor=MUTED, linewidth=0.6, label="validation with COVID (old)")
    ax.bar(x + 0.2, new, width=0.38, color=[FAMILY_COLORS[f] for f in fams], label="post-COVID validation (new)")
    for xi, v in zip(x - 0.2, old):
        if np.isfinite(v):
            ax.text(xi, v, f"{v:.2f}", ha="center", va="bottom", fontsize=6.5, color=MUTED)
    for xi, v in zip(x + 0.2, new):
        if np.isfinite(v):
            ax.text(xi, v, f"{v:.2f}", ha="center", va="bottom", fontsize=6.5, color=INK)
    if ref is not None:
        ax.axhline(ref, color=INK, lw=1, ls="--")
    ax.set_xticks(x, fams)
    ax.set_title(title or f"{value}, h = {h}", loc="left", fontsize=10, color=INK)
    ax.set_ylim(top=ax.get_ylim()[1] * 1.08)  # room for the value labels
    leg = ax.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    leg.legend_handles[1].set_color(MUTED)  # "new" bars use each family's color; show a neutral swatch
    return ax
