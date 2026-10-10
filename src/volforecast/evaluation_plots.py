"""Figures for the model-comparison notebook (notebooks/02_model_comparison.ipynb) and the paper.

Model families keep the same color in every figure. Realized variance is drawn in ink,
and variances are shown as annualized volatility in % (sqrt(252 * var) * 100).

Paper format: figures are FIG_WIDTH = 6.5 in wide with all text >= 9 pt (use_paper_style), carry one
shared legend below the panels and no per-bar value labels, shade each event's peak -> trough window
the same way (shade_event), and are saved as vector PDF next to the PNG (finish).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from volforecast.evaluation import FAMILY_ORDER, PEAKS, TROUGHS

# Fixed color per family (same palette as the XGBoost figures); color follows the model, never its rank.
FAMILY_COLORS = {"GARCH": "#2a78d6", "HAR": "#eb6834", "XGB-linear": "#1baf7a", "XGB-tree": "#eda100",
                 "HARNet": "#e87ba4", "LSTM": "#e34948", "Mamba": "#4a3aa7"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
EVENT_SHADE = dict(color="#c9c7c0", alpha=0.45, lw=0)

FIG_WIDTH = 6.5   # inches: the text width of the thesis
FONT_SIZE = 9     # pt: the smallest text in any figure
PAPER_RC = {
    "font.size": FONT_SIZE, "axes.titlesize": FONT_SIZE, "axes.labelsize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE, "ytick.labelsize": FONT_SIZE, "legend.fontsize": FONT_SIZE,
    "figure.titlesize": FONT_SIZE, "pdf.fonttype": 42, "ps.fonttype": 42,  # embed TrueType: editable text
}


def use_paper_style():
    plt.rcParams.update(PAPER_RC)


def ann_vol(var):
    return np.sqrt(252 * np.asarray(var, dtype=float)) * 100


def style(ax, ylabel: str | None = None):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=FONT_SIZE)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=FONT_SIZE, color=MUTED)


def date_axis(ax):
    loc = mdates.AutoDateLocator(minticks=4, maxticks=8)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))


def shade_event(ax, event: str, label: str | None = "peak → trough"):
    """The event's peak -> trough window, drawn identically in every time-series figure."""
    ax.axvspan(pd.Timestamp(PEAKS[event]), pd.Timestamp(TROUGHS[event]), label=label, **EVENT_SHADE)


def mark_event(ax, event: str, split: dict | None = None):
    shade_event(ax, event)
    ax.axvline(pd.Timestamp(PEAKS[event]), color=MUTED, lw=1, ls=":")
    if split is not None:
        ax.axvline(pd.Timestamp(split["val"]["end"]), color=MUTED, lw=1, ls="--")
        ax.text(pd.Timestamp(split["val"]["end"]), 0.98, " cutoff", transform=ax.get_xaxis_transform(),
                va="top", fontsize=FONT_SIZE, color=MUTED)


def shared_legend(fig, handles=None, labels=None, ncol: int | None = None):
    """One legend for the whole figure, centered below the panels; the panels are laid out above it.

    Without explicit handles, the labelled artists of all axes are collected (first occurrence of each label).
    """
    if handles is None:
        seen = {}
        for ax in fig.axes:
            for h, l in zip(*ax.get_legend_handles_labels()):
                seen.setdefault(l, h)
        labels, handles = list(seen), list(seen.values())
    elif labels is None:
        labels = [h.get_label() for h in handles]
    if not handles:
        fig.tight_layout()
        return None
    leg = fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0), frameon=False,
                     ncol=ncol or min(len(labels), 5), fontsize=FONT_SIZE, handlelength=1.6, columnspacing=1.2)
    fig.canvas.draw()
    frac = leg.get_window_extent().height / fig.bbox.height
    fig.tight_layout(rect=(0, frac + 0.01, 1, 1))
    return leg


def finish(fig, path_stem: str | Path, handles=None, labels=None, ncol: int | None = None, dpi: int = 200):
    """Shared legend, layout, then save path_stem.png (raster) and path_stem.pdf (vector)."""
    shared_legend(fig, handles, labels, ncol)
    stem = Path(path_stem)
    fig.savefig(stem.with_suffix(".png"), dpi=dpi)
    fig.savefig(stem.with_suffix(".pdf"))


def representatives(df: pd.DataFrame, fs: str = "ext") -> pd.DataFrame:
    """One cell per family: the given feature set (GARCH has none)."""
    return df[(df["fs"] == fs) | (df["family"] == "GARCH")]


def plot_forecasts(df, event, h, start, end, split=None, ax=None, title=None):
    d = representatives(df)
    d = d[(d["event"] == event) & (d["h"] == h) & d["date"].between(start, end)]
    ax = ax or plt.subplots(figsize=(FIG_WIDTH, 3))[1]
    style(ax, f"ann. vol, next {h} days (%)")
    mark_event(ax, event, split)
    real = d.drop_duplicates("date").sort_values("date")
    ax.plot(real["date"], ann_vol(real["realized_var"]), color=INK, lw=2.0, label="realized")
    for fam in FAMILY_ORDER:
        g = d[d["family"] == fam].sort_values("date")
        if len(g):
            ax.plot(g["date"], ann_vol(g["forecast_var"]), color=FAMILY_COLORS[fam], lw=1.3, label=fam)
    date_axis(ax)
    ax.set_title(title or f"{event}: forecasts vs realized, h = {h} (ext feature set)", loc="left", color=INK)
    return ax


def plot_lift(df, event, h, start, end, split=None, thresholds=(1.5, 2.0), ax=None, title=None):
    d = representatives(df)
    d = d[(d["event"] == event) & (d["h"] == h) & d["date"].between(start, end)]
    ax = ax or plt.subplots(figsize=(FIG_WIDTH, 3))[1]
    style(ax, "lift (forecast ÷ normal)")
    mark_event(ax, event, split)
    for tau in thresholds:
        ax.axhline(tau, color=MUTED, lw=0.8, ls="--")
        ax.text(d["date"].min(), tau, f" {tau:g}×", va="bottom", fontsize=FONT_SIZE, color=MUTED)
    real = d.drop_duplicates("date").sort_values("date")
    ax.plot(real["date"], real["realized_lift"], color=INK, lw=2.0, label="realized")
    for fam in FAMILY_ORDER:
        g = d[d["family"] == fam].sort_values("date")
        if len(g):
            ax.plot(g["date"], g["lift"], color=FAMILY_COLORS[fam], lw=1.3, label=fam)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(mticker.LogLocator(base=2))  # 0.25, 0.5, 1, 2, 4: plain multiples of normal
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    date_axis(ax)
    ax.set_title(title or f"{event}: early-warning lift, h = {h}", loc="left", color=INK)
    return ax


LIFT_WINDOWS = {"covid": ("2019-10-01", "2020-04-30"), "bear2022": ("2021-07-01", "2023-03-31")}
EVENT_NAMES = {"covid": "COVID crash", "bear2022": "2022 drawdown"}


def plot_lift_grid(df, splits, horizons=(5, 22), events=("covid", "bear2022"), thresholds=(1.5, 2.0),
                   windows=LIFT_WINDOWS, height: float = 6.6):
    """Early-warning lift, rows = horizon, columns = event; one log y-axis per row so events compare directly."""
    fig, axes = plt.subplots(len(horizons), len(events), figsize=(FIG_WIDTH, height), sharey="row", squeeze=False)
    for i, h in enumerate(horizons):
        for j, event in enumerate(events):
            ax = axes[i, j]
            letter = chr(ord("a") + i * len(events) + j)
            plot_lift(df, event, h, *windows[event], splits[event], thresholds=thresholds, ax=ax,
                      title=f"({letter}) {EVENT_NAMES[event]}, h = {h}")
            loc = mdates.AutoDateLocator(minticks=3, maxticks=5)  # half page width: fewer date ticks
            ax.xaxis.set_major_locator(loc)
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
            for t in ax.texts:  # narrow panels: keep labels clear of each other and of the peak line
                if t.get_text().strip() == "cutoff":
                    t.set(text="cutoff ", rotation=90, ha="right", va="top")
                elif t.get_text().strip() == f"{min(thresholds):g}×":
                    t.set_va("top")  # lowest threshold labelled below its line, the others above
            if j > 0:
                ax.set_ylabel("")
                ax.tick_params(labelleft=True)
    return fig, axes


def bar_by_family(table: pd.DataFrame, value: str, ax=None, ref: float | None = None, title: str = "",
                  ylabel: str | None = None):
    """Grouped bars: one group per horizon, one bar per family (ext set; GARCH has none). No value labels."""
    t = table[(table["fs"].isin(["ext", "-"]))]
    hs = sorted(t["h"].unique())
    fams = [f for f in FAMILY_ORDER if f in set(t["family"])]
    ax = ax or plt.subplots(figsize=(FIG_WIDTH, 2.8))[1]
    style(ax, ylabel or value)
    width = 0.8 / len(fams)
    for i, fam in enumerate(fams):
        vals = [t[(t["family"] == fam) & (t["h"] == h)][value].mean() for h in hs]
        ax.bar(np.arange(len(hs)) + i * width, vals, width=width * 0.92, color=FAMILY_COLORS[fam], label=fam)
    if ref is not None:
        ax.axhline(ref, color=INK, lw=1, ls="--")
    ax.set_xticks(np.arange(len(hs)) + 0.4 - width / 2, [f"h = {h}" for h in hs])
    ax.set_title(title, loc="left", color=INK)
    return ax


def plot_alarm_skill(alarms: pd.DataFrame, event: str, tau: float = 1.5, ax=None, title: str | None = None):
    """Hit rate vs false-alarm rate per model cell. The diagonal = no skill (random or always-alarm)."""
    a = alarms[(alarms["event"] == event) & (alarms["tau"] == tau)]
    ax = ax or plt.subplots(figsize=(FIG_WIDTH / 2, FIG_WIDTH / 2))[1]
    style(ax, "hit rate")
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--")
    ax.text(0.6, 0.53, "no skill", rotation=45, fontsize=FONT_SIZE, color=MUTED)
    markers = {1: "o", 5: "s", 22: "^"}
    for fam in FAMILY_ORDER:
        g = a[a["family"] == fam]
        for _, r in g.iterrows():
            ax.scatter(r["false_alarm_rate"], r["hit_rate"], s=40, color=FAMILY_COLORS[fam],
                       marker=markers.get(r["h"], "o"), edgecolor="white", linewidth=1.2,
                       alpha=1.0 if r["fs"] in ("ext", "-") else 0.45)
    ax.set_xlabel("false-alarm rate", fontsize=FONT_SIZE, color=MUTED)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title or f"{event}, forecast ≥ {tau:g}× normal", loc="left", color=INK)
    return ax


def alarm_skill_handles(alarms: pd.DataFrame) -> list:
    """Legend entries for plot_alarm_skill: family colors, horizon markers, and the faded base set."""
    fams = [f for f in FAMILY_ORDER if f in set(alarms["family"])]
    out = [Line2D([], [], ls="", marker="o", markersize=7, color=FAMILY_COLORS[f], label=f) for f in fams]
    out += [Line2D([], [], ls="", marker=m, markersize=7, color=MUTED, label=f"h = {h}")
            for h, m in {1: "o", 5: "s", 22: "^"}.items()]
    out.append(Line2D([], [], ls="", marker="o", markersize=7, color=MUTED, alpha=0.45, label="base set (faded)"))
    return out


def plot_smearing_shift(comp: pd.DataFrame, value: str, h: int, ax=None, title: str = "", ref: float | None = None,
                        ylabel: str | None = None):
    """Paired bars per family: old split (validation with COVID, muted) vs new split (post-COVID, colored)."""
    t = comp[(comp["h"] == h) & comp["fs"].isin(["ext", "-"])]
    fams = [f for f in FAMILY_ORDER if f in set(t["family"])]
    ax = ax or plt.subplots(figsize=(FIG_WIDTH, 2.8))[1]
    style(ax, ylabel or value)
    x = np.arange(len(fams))
    old = [t[t["family"] == f][f"{value}__val_with_covid"].mean() for f in fams]
    new = [t[t["family"] == f][f"{value}__val_post_covid"].mean() for f in fams]
    ax.bar(x - 0.2, old, width=0.38, color=GRID, edgecolor=MUTED, linewidth=0.6)
    ax.bar(x + 0.2, new, width=0.38, color=[FAMILY_COLORS[f] for f in fams])
    if ref is not None:
        ax.axhline(ref, color=INK, lw=1, ls="--")
    ax.set_xticks(x, fams, rotation=35, ha="right", rotation_mode="anchor")
    ax.set_title(title or f"{value}, h = {h}", loc="left", color=INK)
    return ax


def smearing_handles() -> list:
    """Legend entries for plot_smearing_shift ("new" bars use each family's color; a neutral swatch stands in)."""
    return [Patch(facecolor=GRID, edgecolor=MUTED, linewidth=0.6, label="validation with COVID (old split)"),
            Patch(facecolor=MUTED, label="post-COVID validation (new split)")]
