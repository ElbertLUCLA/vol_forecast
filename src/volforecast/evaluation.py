"""Shared evaluation of every model's forecast files (PROJECT_CONTEXT.md section 8).

Everything here reads only data/processed/forecasts/{event}/{model}.parquet (the shared schema),
so all models are scored identically. The measures answer three questions:

1. Early warning (COVID, 2022): when does each model's forecast rise well above normal?
   lift_t = forecast_var_t / long-run mean RV of the training sample (default; see load_forecasts
   for the alternative "own train-tail" baseline). Lead time = trading days from the first
   lift >= threshold to the market peak.
2. Continuation after the first drop (COVID onset) and tracking a slow decline (2022):
   - escalation_t = forecast_var_t / trailing 5-day RV at t. Above 1 means the model expects
     the next h days to be worse than what has just been seen.
   - coverage_t = forecast_var_t / realized_var_t (realized mean RV over t+1..t+h).
     1 = forecast the right level; < 1 = under-prediction.
3. Accurate or just alarm-happy? Alarms are judged against realized stress on all out-of-sample
   days, including a calm window after each event:
     alarm_t  = lift_t >= tau          (forecast variance >= tau x normal)
     stress_t = realized lift_t >= tau (realized variance >= tau x normal)
   hit rate = P(alarm | stress), false-alarm rate = P(alarm | no stress),
   Peirce skill score = hit rate - false-alarm rate (0 = no skill; always-alarm scores 0).
   Also: calm-window bias, mean ln(forecast / realized). Above 0 means it over-forecasts when nothing happens.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from volforecast.config import load_config, resolve_path

PEAKS = {"covid": "2020-02-19", "bear2022": "2022-01-03"}
TROUGHS = {"covid": "2020-03-23", "bear2022": "2022-10-12"}
THRESHOLDS = (1.5, 2.0, 3.0)
OOS = ("pre_event", "event", "post_event", "calm")

FAMILY_ORDER = ["GARCH", "HAR", "XGB-linear", "XGB-tree", "HARNet", "LSTM", "Mamba"]


def family(model: str) -> str:
    if model.startswith("xgb-gbtree"):
        return "XGB-tree"
    if model.startswith("xgb-gblinear"):
        return "XGB-linear"
    for prefix, fam in (("garch", "GARCH"), ("harnet", "HARNet"), ("har_", "HAR"),
                        ("lstm", "LSTM"), ("mamba", "Mamba")):
        if model.startswith(prefix):
            return fam
    return model


def longrun_rv(features: pd.DataFrame | None = None, splits_dir: str | Path | None = None) -> dict[str, float]:
    """Mean daily RV over each event's whole pre-cutoff sample [train.start, cutoff]: the "normal" level."""
    from volforecast import adapters

    features = features if features is not None else adapters.load_features()
    rv = features.set_index("date")["rv"]
    out = {}
    for event in PEAKS:
        split = adapters.load_split(event, splits_dir)
        out[event] = float(rv.loc[split["train"]["start"]:split["val"]["end"]].mean())
    return out


def load_forecasts(forecasts_dir: str | Path | None = None, basis: str = "longrun",
                   features: pd.DataFrame | None = None) -> pd.DataFrame:
    """All forecast files, long format, with family / lift / ratio columns added.

    Two baselines for "lift" (variance relative to normal):
      longrun (default): the long-run mean realized RV over the event's training sample, common to all
                         models, so lift >= 2 means "forecast variance is twice the normal level".
      tail:              each model's own mean forecast (and the mean realized) over its last 63
                         pre-cutoff days, as originally planned. Degenerate for 2022, because Jul-Sep 2021
                         was unusually calm (~8% vol), so every later day looks like an alarm.
    Both are kept as columns (lift_longrun / lift_tail, ...); `lift` / `realized_lift` hold the chosen basis.
    """
    if forecasts_dir is None:
        forecasts_dir = resolve_path(load_config()["paths"]["forecasts_dir"])
    parts = [pd.read_parquet(p) for p in sorted(Path(forecasts_dir).glob("*/*.parquet"))]
    df = pd.concat(parts, ignore_index=True)
    df["family"] = df["model"].map(family)
    df["fs"] = df["featureset"].replace({"returns": "-"})
    key = ["event", "model"]
    tail = df[df["sample"] == "train_tail"].groupby(key)
    df = df.merge(tail["forecast_var"].mean().rename("base_fc"), on=key)
    df = df.merge(tail["realized_var"].mean().rename("base_rv"), on=key)
    df["lift_tail"] = df["forecast_var"] / df["base_fc"]
    df["realized_lift_tail"] = df["realized_var"] / df["base_rv"]
    lr = df["event"].map(longrun_rv(features))
    df["lift_longrun"] = df["forecast_var"] / lr
    df["realized_lift_longrun"] = df["realized_var"] / lr
    df["lift"], df["realized_lift"] = df[f"lift_{basis}"], df[f"realized_lift_{basis}"]
    df["coverage"] = df["forecast_var"] / df["realized_var"]
    df["log_ratio"] = np.log(df["coverage"])
    return df


def trailing_rv(features: pd.DataFrame, window: int = 5) -> pd.Series:
    """Mean RV over the last `window` days up to and including t (what an observer saw at t)."""
    f = features.sort_values("date").set_index("date")
    return f["rv"].rolling(window).mean()


def add_escalation(df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    trv = trailing_rv(features)
    out = df.copy()
    out["trailing_rv5"] = out["date"].map(trv)
    out["escalation"] = out["forecast_var"] / out["trailing_rv5"]
    return out


# ---------------------------------------------------------------- accuracy

VAR_SCALE = 1e4   # daily decimal variance -> daily %^2, so squared errors are not ~1e-8


def _qlike(r, f):
    x = r / f
    return x - np.log(x) - 1.0


def _se_var(r, f):
    """Squared error on the variance scale, in (daily %^2)^2. Robust to a noisy RV proxy (Patton 2011)."""
    return ((r - f) * VAR_SCALE) ** 2


def _ae_var(r, f):
    """Absolute error on the variance scale, in daily %^2. Not robust; for comparability with the papers only."""
    return np.abs(r - f) * VAR_SCALE


LOSSES = {"qlike": _qlike, "mse_var": _se_var}


def accuracy_table(df: pd.DataFrame) -> pd.DataFrame:
    """Losses and bias per (event, model, h, segment), complete targets only.

    qlike (primary) and mse_var (secondary) are the two losses robust to the noisy RV proxy (Patton 2011);
    both score forecast_var, the smeared back-transformed forecast. mae_var is for comparability with
    Teller et al. / Reisenhofer et al. only, and mse_log is a diagnostic (it favours the unsmeared log models).
    mse_var_rel = mse_var / HAR-base mse_var in the same (event, h, sample). Never pool mse_var across
    windows: it is dominated by crash days.
    """
    d = df[df["target_complete"] & df["sample"].isin(OOS)].copy()
    d["qlike"] = _qlike(d["realized_var"], d["forecast_var"])
    d["se_var"] = _se_var(d["realized_var"], d["forecast_var"])
    d["ae_var"] = _ae_var(d["realized_var"], d["forecast_var"])
    d["se_log"] = (d["realized_log"] - d["forecast_log"]) ** 2
    g = d.groupby(["event", "family", "fs", "model", "h", "sample"])
    out = g.agg(qlike=("qlike", "mean"), mse_var=("se_var", "mean"), mae_var=("ae_var", "mean"),
                mse_log=("se_log", "mean"), bias_log=("log_ratio", "mean"),
                n=("qlike", "size")).reset_index()
    key = ["event", "h", "sample"]
    bench = out.loc[(out["family"] == "HAR") & (out["fs"] == "base"), key + ["mse_var"]]
    out = out.merge(bench.rename(columns={"mse_var": "_bench"}), on=key, how="left")
    out["mse_var_rel"] = out["mse_var"] / out.pop("_bench")
    return out


def dm_test(loss_a: np.ndarray, loss_b: np.ndarray, h: int) -> tuple[float, float]:
    """Diebold-Mariano with a Newey-West variance (lag h-1). Negative stat = A has lower loss."""
    from scipy import stats

    d = np.asarray(loss_a) - np.asarray(loss_b)
    n = len(d)
    dbar = d.mean()
    dc = d - dbar
    lag = max(h - 1, 0)
    var = dc @ dc / n
    for k in range(1, lag + 1):
        var += 2 * (1 - k / (lag + 1)) * (dc[k:] @ dc[:-k]) / n
    if var <= 0:  # identical losses on every day: no difference to test
        return 0.0, 1.0
    stat = dbar / np.sqrt(var / n)
    return float(stat), float(2 * (1 - stats.norm.cdf(abs(stat))))


def dm_vs_benchmark(df: pd.DataFrame, benchmark: str = "HAR", segments=("event",),
                    loss: str = "qlike") -> pd.DataFrame:
    """DM test of every model against the benchmark family's base model (same event, h).

    loss: "qlike" (default) or "mse_var", the two losses robust to the noisy RV proxy. The mean losses
    are in columns `{loss}` and `{loss}_har_base`, so the default output is unchanged apart from the
    added `loss` column.
    """
    if loss not in LOSSES:
        raise ValueError(f"loss must be one of {sorted(LOSSES)}, got {loss!r}")
    d = df[df["target_complete"] & df["sample"].isin(segments)].copy()
    d[loss] = LOSSES[loss](d["realized_var"], d["forecast_var"])
    rows = []
    for (event, h), g in d.groupby(["event", "h"]):
        bench = g[(g["family"] == benchmark) & (g["fs"] == "base")].set_index("date")[loss]
        for model, gm in g.groupby("model"):
            a = gm.set_index("date")[loss].reindex(bench.index).dropna()
            if gm["family"].iloc[0] == benchmark and gm["fs"].iloc[0] == "base":
                continue
            stat, p = dm_test(a.to_numpy(), bench.loc[a.index].to_numpy(), h)
            rows.append({"event": event, "h": h, "model": model, "family": gm["family"].iloc[0],
                         "fs": gm["fs"].iloc[0], "loss": loss, loss: a.mean(),
                         f"{loss}_har_base": bench.mean(), "dm_stat": stat, "p_value": p})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- early warning

def lead_times(df: pd.DataFrame, thresholds=THRESHOLDS) -> pd.DataFrame:
    """First out-of-sample date each model's lift crosses each threshold, in trading days vs the peak.

    Negative = before the peak (a genuine early warning). Searched from the cutoff to the trough.
    """
    rows = []
    for (event, model), g in df[df["sample"].isin(OOS)].groupby(["event", "model"]):
        g = g.sort_values("date")
        trough = pd.Timestamp(TROUGHS[event])
        g = g[g["date"] <= trough]
        dates = list(g["date"])
        peak_pos = int(np.searchsorted(np.array(dates, dtype="datetime64[ns]"), np.datetime64(PEAKS[event])))
        rec = {"event": event, "model": model, "family": g["family"].iloc[0], "fs": g["fs"].iloc[0],
               "h": int(g["h"].iloc[0]), "max_lift_to_trough": float(g["lift"].max())}
        for tau in thresholds:
            hit = np.flatnonzero(g["lift"].to_numpy() >= tau)
            rec[f"first_{tau}x"] = dates[hit[0]].date() if len(hit) else None
            rec[f"lead_days_{tau}x"] = int(peak_pos - hit[0]) if len(hit) else None
        rows.append(rec)
    return pd.DataFrame(rows)


def onset_table(df: pd.DataFrame, event: str = "covid", n_days: int = 10) -> pd.DataFrame:
    """First n event days: coverage (forecast / realized forward), escalation, cumulative shortfall."""
    d = df[(df["event"] == event) & (df["sample"] == "event") & df["target_complete"]].sort_values("date")
    first = sorted(d["date"].unique())[:n_days]
    d = d[d["date"].isin(first)]
    g = d.groupby(["family", "fs", "model", "h"])
    return g.agg(coverage_mean=("coverage", "mean"),
                 coverage_min=("coverage", "min"),
                 escalation_mean=("escalation", "mean"),
                 share_escalating=("escalation", lambda s: float((s > 1).mean())),
                 shortfall_var=("realized_var", "sum")).reset_index().assign(
        shortfall_var=lambda t: t["shortfall_var"] - g["forecast_var"].sum().to_numpy())


def tracking_table(df: pd.DataFrame, event: str) -> pd.DataFrame:
    """Whole event window: how well does each model stay on the realized level?"""
    d = df[(df["event"] == event) & (df["sample"] == "event") & df["target_complete"]].copy()
    d["qlike"] = _qlike(d["realized_var"], d["forecast_var"])
    d["se_var"] = _se_var(d["realized_var"], d["forecast_var"])
    g = d.groupby(["family", "fs", "model", "h"])
    return g.agg(qlike=("qlike", "mean"), mse_var=("se_var", "mean"), bias_log=("log_ratio", "mean"),
                 share_under=("coverage", lambda s: float((s < 1).mean())),
                 share_lift_2=("lift", lambda s: float((s >= 2.0).mean())),
                 coverage_median=("coverage", "median")).reset_index()


# ---------------------------------------------------------------- alarms

def alarm_table(df: pd.DataFrame, thresholds=THRESHOLDS) -> pd.DataFrame:
    """Signal-detection scores of lift-based alarms against realized stress, all OOS days."""
    d = df[df["target_complete"] & df["sample"].isin(OOS)]
    rows = []
    for (event, model), g in d.groupby(["event", "model"]):
        calm = g[g["sample"] == "calm"]
        for tau in thresholds:
            alarm, stress = g["lift"] >= tau, g["realized_lift"] >= tau
            hit = float(alarm[stress].mean()) if stress.any() else np.nan
            fa = float(alarm[~stress].mean()) if (~stress).any() else np.nan
            prec = float(stress[alarm].mean()) if alarm.any() else np.nan
            rows.append({
                "event": event, "model": model, "family": g["family"].iloc[0], "fs": g["fs"].iloc[0],
                "h": int(g["h"].iloc[0]), "tau": tau,
                "alarm_rate": float(alarm.mean()), "stress_rate": float(stress.mean()),
                "alarm_to_stress": float(alarm.mean() / stress.mean()) if stress.mean() > 0 else np.nan,
                "hit_rate": hit, "false_alarm_rate": fa, "precision": prec, "peirce": hit - fa,
                "calm_alarm_rate": float((calm["lift"] >= tau).mean()),
                "calm_false_alarm_rate": float(((calm["lift"] >= tau) & (calm["realized_lift"] < tau)).mean()),
                "calm_bias_log": float(calm["log_ratio"].mean()),
            })
    return pd.DataFrame(rows)


def training_summary(models_root: str | Path | None = None) -> pd.DataFrame:
    """Latest validation metrics of every model cell, gathered from each model's run_log.csv."""
    from volforecast.config import PROJECT_ROOT

    root = Path(models_root) if models_root else PROJECT_ROOT / "models"
    frames = []
    for name in ("garch", "har", "xgb", "harnet", "seq"):
        p = root / name / "run_log.csv"
        if not p.exists():
            continue
        t = pd.read_csv(p).drop_duplicates(["event", "model"], keep="last")
        t["family"] = t["model"].map(family)
        if name == "xgb":
            # XGBoost logs log-MSE as best_val_mse_log and QLIKE as best_val_qlike.
            t = t.rename(columns={"best_val_qlike": "val_qlike", "best_val_mse_log": "val_mse_log"})
        t["train_seconds"] = t[[c for c in t.columns if c.startswith("sec_")]].sum(axis=1)
        frames.append(t)
    cols = ["event", "family", "model", "h", "val_qlike", "val_qlike_rw", "val_mse_log", "train_seconds"]
    out = pd.concat(frames, ignore_index=True)
    # The random walk is the same benchmark for every model with the same event and horizon.
    out["val_qlike_rw"] = out.groupby(["event", "h"])["val_qlike_rw"].transform("max")
    return out[[c for c in cols if c in out]].sort_values(["event", "h", "family", "model"])



# ---------------------------------------------------------------- smearing finding (2022 validation block)

def smearing_comparison(old_dir: str | Path, new_dir: str | Path, event: str = "bear2022", tau: float = 2.0,
                        features: pd.DataFrame | None = None) -> pd.DataFrame:
    """Same models, two 2022 splits: validation block with COVID (old) vs post-COVID (new).

    The smearing factor S is recovered from the forecast files (forecast_var / forecast_var_naive,
    constant per file; 1 for the level models HARNet and GARCH). Also compared: event QLIKE and bias,
    calm-window bias, and alarm behaviour at `tau` x normal.
    """
    rows = []
    for label, d in (("val_with_covid", old_dir), ("val_post_covid", new_dir)):
        df = load_forecasts(d, basis="longrun", features=features)
        df = df[df["event"] == event]
        acc = accuracy_table(df)
        al = alarm_table(df, thresholds=(tau,))
        lt = lead_times(df, thresholds=(tau,))
        for model, g in df.groupby("model"):
            ev = acc[(acc["model"] == model) & (acc["sample"] == "event")]
            a = al[al["model"] == model].iloc[0]
            rows.append({
                "split": label, "model": model, "family": g["family"].iloc[0], "fs": g["fs"].iloc[0],
                "h": int(g["h"].iloc[0]),
                "S": float((g["forecast_var"] / g["forecast_var_naive"]).median()),
                "event_qlike": float(ev["qlike"].iloc[0]), "event_bias_log": float(ev["bias_log"].iloc[0]),
                "calm_bias_log": a["calm_bias_log"], "alarm_to_stress": a["alarm_to_stress"],
                "false_alarm_rate": a["false_alarm_rate"], "peirce": a["peirce"],
                f"lead_days_{tau}x": lt[lt["model"] == model][f"lead_days_{tau}x"].iloc[0],
            })
    long = pd.DataFrame(rows)
    wide = long.pivot_table(index=["family", "fs", "model", "h"], columns="split",
                            values=[c for c in long.columns if c not in ("split", "model", "family", "fs", "h")])
    wide.columns = [f"{v}__{s}" for v, s in wide.columns]
    return wide.reset_index()
