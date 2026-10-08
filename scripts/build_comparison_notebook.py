"""Build notebooks/02_model_comparison.ipynb (cells only), then execute it with nbconvert:

    uv run python scripts/build_comparison_notebook.py
    uv run jupyter nbconvert --to notebook --execute --inplace notebooks/02_model_comparison.ipynb

The written conclusions live in notebooks/02_summary.md; edit that file and rebuild so the prose stays
in sync with the numbers. Re-execute after any model re-run: a notebook only shows its last execution.
"""
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "notebooks" / "02_model_comparison.ipynb"
SUMMARY_PATH = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "notebooks" / "02_summary.md"
SUMMARY = SUMMARY_PATH.read_text(encoding="utf-8")

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# Which volatility model warns best? Model comparison on SPY, COVID-19 and the 2022 bear market

Every model below was trained only on data **before** its event cutoff, frozen, and then forecast
day by day with live inputs. Every model is scored from the same forecast files
(`data/processed/forecasts/`).

| Family | Inputs | Notes |
|---|---|---|
| GARCH | close-to-close returns | GARCH(1,1)-t, scaled to the RTH-RV level |
| HAR | log RV d/w/m (+ RS⁻, signed jump for `ext`) | OLS, smearing |
| XGB-linear / XGB-tree | same as HAR | Teller et al. (2022) search space, 100 TPE trials |
| HARNet | raw daily RV (+ RS⁻, ΔJ²) | PyTorch port, QLIKE, OLS-HAR init |
| LSTM | raw daily log RV (+ log RS⁻, jump) over a tuned 22–126-day lookback | control for "plain recurrence" |
| Mamba | same as LSTM | selective state-space (Gu & Dao 2023), trained on the AMD GPU |

**Three questions**

1. **COVID (fast crash):** who signalled first, and who said "this is going to continue" after the first drop?
2. **2022 (slow bear market):** who tracked the drawn-out decline best?
3. **Accuracy or alarm-happiness?** Did the models that looked good there forecast *accurately*,
   or do they simply raise alarms often? We judge alarms against realized stress on every
   out-of-sample day, including a **calm window** after each event (COVID model: 2020-07 → 2021-09;
   2022 model: 2023-04 → 2025-03).

**Measures** (`volforecast/evaluation.py`)

- **lift** = forecast variance ÷ the *normal* variance level, meaning the long-run mean RV of the training sample
  (≈ 14% annualized vol for both events). An **alarm** is lift ≥ τ, and **realized stress** is realized variance ≥ τ × normal.
  τ = 2 (variance double normal, ≈ 20% vol) is the headline threshold.
  The originally planned baseline, each model's own last 63 pre-cutoff days, is shown as a secondary view. It is degenerate
  for 2022 because Jul–Sep 2021 was unusually calm (≈ 8% vol), which makes every later day look like an alarm.
- **escalation** = forecast of the next h days ÷ RV over the last 5 days. Above 1 means the model expects things to get worse.
- **coverage** = forecast ÷ realized (forward). 1 = right level; below 1 = under-prediction.
- **Peirce skill** = hit rate − false-alarm rate. A model that always alarms scores 0, however many crashes it "catches".
""")

code(r"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from IPython.display import display
from volforecast import adapters, evaluation as E, evaluation_plots as P
from volforecast.config import load_config, resolve_path, PROJECT_ROOT

pd.set_option("display.precision", 3); pd.set_option("display.width", 160)
plt.rcParams.update({"figure.dpi": 110, "font.size": 9})
cfg = load_config()
features = adapters.load_features()
splits = {e: adapters.load_split(e) for e in ("covid", "bear2022")}
df = E.add_escalation(E.load_forecasts(basis="longrun", features=features), features)
df_tail = E.load_forecasts(basis="tail", features=features)
print({e: f"normal = {np.sqrt(252 * v) * 100:.1f}% ann. vol" for e, v in E.longrun_rv(features).items()})
out_dir = PROJECT_ROOT / "reports" / "eval"; out_dir.mkdir(parents=True, exist_ok=True)
print(df.groupby(["event", "family"])["model"].nunique().unstack(0))
""")

md(r"""
## 1. Training metrics

Validation-block scores of each model's **train-only** fit, the setting that picked its
hyperparameters, iterations or epochs. Lower QLIKE is better, and the random walk (RW) is the benchmark.
The 2022 models validate on 2021-01 → 2021-09, a calm stretch; COVID is in their training data (§7).
""")
code(r"""
train = E.training_summary()
train.to_csv(out_dir / "training_summary.csv", index=False)
pivot = train.pivot_table(index=["family", "model"], columns=["event", "h"], values="val_qlike")
rw = train.groupby(["event", "h"])["val_qlike_rw"].first().rename(("RW", "random walk")).to_frame().T
rw.columns = pd.MultiIndex.from_tuples(rw.columns)
display((pd.concat([pivot, rw])).style.format("{:.3f}").background_gradient(axis=0, cmap="Greens_r")
        .set_caption("Validation QLIKE (lower is better)"))
""")
code(r"""
t = train.groupby("family")["train_seconds"].agg(["count", "sum", "median"]).rename(
    columns={"count": "cells", "sum": "total s", "median": "median s / cell"})
display(t.loc[[f for f in E.FAMILY_ORDER if f in t.index]].style.format("{:.1f}", subset=["total s", "median s / cell"]))
""")

md(r"""
## 2. Out-of-sample accuracy by window

Mean QLIKE per window, out of sample (the frozen model's live forecasts). `pre_event`/`event`/`post_event` are
the drawdown windows; `calm` is the quiet stretch afterwards.
""")
code(r"""
acc = E.accuracy_table(df)
acc.to_csv(out_dir / "accuracy_by_window.csv", index=False)
for event in ("covid", "bear2022"):
    t = acc[acc["event"] == event].pivot_table(index=["family", "fs"], columns=["h", "sample"], values="qlike")
    t = t.reindex(columns=pd.MultiIndex.from_product([[1, 5, 22], E.OOS]), fill_value=np.nan)
    display(t.style.format("{:.3f}").background_gradient(axis=0, cmap="Greens_r")
            .set_caption(f"{event}: QLIKE by horizon and window (lower is better)"))
""")

md(r"""
## 3. COVID-19 crash (fast)

Peak 2020-02-19, trough 2020-03-23. Models were frozen on 2019-12-31.
For h = 22, pre-peak *realized* targets already contain crash days, so judge the run-up by the
forecast's own movement (lift), not by forecast-vs-realized loss.
""")
code(r"""
fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True)
for ax, h in zip(axes, (5, 22)):
    P.plot_forecasts(df, "covid", h, "2019-12-01", "2020-06-30", splits["covid"], ax=ax)
fig.tight_layout(); fig.savefig(out_dir / "covid_forecasts.png", dpi=150)
""")
code(r"""
fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
for ax, h in zip(axes, (5, 22)):
    P.plot_lift(df, "covid", h, "2019-10-01", "2020-04-30", splits["covid"], thresholds=(2.0, 3.0), ax=ax)
fig.tight_layout(); fig.savefig(out_dir / "covid_lift.png", dpi=150)
""")
md(r"""
### 3a. Early warning: when did each model's forecast cross 1.5× / 2× / 3× normal?

`lead_days` = trading days **before** the peak (negative = after the peak). Searched from the cutoff to the trough.
""")
code(r"""
lt = E.lead_times(df)
lt.to_csv(out_dir / "lead_times.csv", index=False)
cols = ["family", "fs", "h", "first_1.5x", "lead_days_1.5x", "first_2.0x", "lead_days_2.0x",
        "first_3.0x", "lead_days_3.0x", "max_lift_to_trough"]
display(lt[(lt["event"] == "covid") & lt["fs"].isin(["ext", "-"])].sort_values(["h", "family"])[cols]
        .set_index(["h", "family", "fs"]))
# Secondary view: lift against each model's own last 63 pre-cutoff days (the originally planned baseline)
lt_tail = E.lead_times(df_tail)
lt_tail.to_csv(out_dir / "lead_times_tail_baseline.csv", index=False)
display(lt_tail[(lt_tail["event"] == "covid") & lt_tail["fs"].isin(["ext", "-"])].sort_values(["h", "family"])
        [cols[:7]].set_index(["h", "family", "fs"]).style.set_caption("Own-tail baseline (secondary)"))
""")
md(r"""
### 3b. Did the model say "this will continue" after the first drop?

The first 10 event days (2020-02-20 → 03-04): the market had fallen sharply, but most of the realized
volatility was still ahead (it peaked in mid-March).

- **escalation** > 1: the forecast for the next h days sits above the last 5 days' realized level, so the model expects worse to come.
- **coverage**: share of the coming realized variance the forecast anticipated.
- **shortfall**: cumulative realized − forecast variance over those days. Smaller is better.
""")
code(r"""
onset = E.onset_table(df, "covid", n_days=10)
onset.to_csv(out_dir / "covid_onset.csv", index=False)
display(onset[onset["fs"].isin(["ext", "-"])].sort_values(["h", "coverage_mean"], ascending=[True, False])
        .set_index(["h", "family"]).drop(columns=["fs"]).style.format("{:.3g}"))
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
P.bar_by_family(onset, "escalation_mean", ax=axes[0], ref=1.0, title="Escalation in the first 10 event days (> 1 = expects worse)")
P.bar_by_family(onset, "coverage_mean", ax=axes[1], ref=1.0, title="Coverage of what came next (1 = right level)")
fig.tight_layout(); fig.savefig(out_dir / "covid_onset.png", dpi=150)
""")

md(r"""
## 4. 2022 bear market (slow)

Peak 2022-01-03, trough 2022-10-12. Models were frozen on 2021-09-30. They train through 2020-12 (COVID included) and validate on 2021-01 → 2021-09 (see §7). Volatility in 2022 was
elevated but never spiked like 2020, so the test is **staying on the right level for nine months**.
""")
code(r"""
fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True)
for ax, h in zip(axes, (5, 22)):
    P.plot_forecasts(df, "bear2022", h, "2021-09-01", "2023-03-31", splits["bear2022"], ax=ax)
fig.tight_layout(); fig.savefig(out_dir / "bear2022_forecasts.png", dpi=150)
""")
code(r"""
fig, ax = plt.subplots(figsize=(11, 3.8))
P.plot_lift(df, "bear2022", 22, "2021-07-01", "2023-03-31", splits["bear2022"], thresholds=(2.0, 3.0), ax=ax)
fig.tight_layout(); fig.savefig(out_dir / "bear2022_lift.png", dpi=150)
display(lt[(lt["event"] == "bear2022") & lt["fs"].isin(["ext", "-"])].sort_values(["h", "family"])[cols]
        .set_index(["h", "family", "fs"]))
""")
code(r"""
track = E.tracking_table(df, "bear2022")
track.to_csv(out_dir / "bear2022_tracking.csv", index=False)
display(track[track["fs"].isin(["ext", "-"])].sort_values(["h", "qlike"]).set_index(["h", "family"])
        .drop(columns=["fs"]).style.format("{:.3f}"))
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
P.bar_by_family(track, "qlike", ax=axes[0], title="2022 event window: QLIKE (lower is better)")
P.bar_by_family(track, "bias_log", ax=axes[1], ref=0.0, title="2022 event window: mean ln(forecast / realized)")
fig.tight_layout(); fig.savefig(out_dir / "bear2022_tracking.png", dpi=150)
""")

md(r"""
## 5. Accurate, or just alarm-happy?

A model can look like a great early-warning system simply because it forecasts high variance often.
Three checks separate the two:

1. **Alarm skill** (hit rate vs false-alarm rate over *all* out-of-sample days). The diagonal is no skill:
   an always-alarm model sits at (1, 1). **Peirce = hit − false alarm** is the distance above it.
2. **Alarm-to-stress ratio**: how often the model alarms relative to how often stress actually occurred
   (≈ 1 calibrated, > 1 alarm-happy).
3. **Calm-window bias**: mean ln(forecast/realized) when nothing is happening. Above 0 = over-forecasts in calm markets.
""")
code(r"""
alarms = E.alarm_table(df)
alarms.to_csv(out_dir / "alarm_skill.csv", index=False)
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, event in zip(axes, ("covid", "bear2022")):
    P.plot_alarm_skill(alarms, event, tau=2.0, ax=ax)
fig.tight_layout(); fig.savefig(out_dir / "alarm_skill.png", dpi=150)
""")
code(r"""
a = alarms[(alarms["tau"] == 2.0) & alarms["fs"].isin(["ext", "-"])]
show = ["hit_rate", "false_alarm_rate", "peirce", "alarm_to_stress", "calm_false_alarm_rate", "calm_bias_log"]
for event in ("covid", "bear2022"):
    t = a[a["event"] == event].set_index(["h", "family"])[show].sort_index()
    display(t.style.format("{:.2f}").background_gradient(subset=["peirce"], cmap="Greens")
            .set_caption(f"{event}: alarm diagnostics at forecast ≥ 2× normal (ext set)"))
""")
code(r"""
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
for ax, event in zip(axes, ("covid", "bear2022")):
    t = a[a["event"] == event]
    P.bar_by_family(t, "calm_bias_log", ax=ax, ref=0.0, title=f"{event} model, calm window: mean ln(forecast / realized)")
fig.tight_layout(); fig.savefig(out_dir / "calm_bias.png", dpi=150)
""")
md(r"""
### Is the event-window advantage statistically significant?

Diebold–Mariano tests on QLIKE against the HAR (base) benchmark, event window, with a Newey–West variance (lag h − 1).
A negative statistic means the model's loss is lower than HAR's.
""")
code(r"""
dm = E.dm_vs_benchmark(df, "HAR", segments=("event",))
dm.to_csv(out_dir / "dm_event_vs_har.csv", index=False)
d = dm[dm["fs"].isin(["ext", "-"])].pivot_table(index="family", columns=["event", "h"], values=["dm_stat", "p_value"])
display(d.style.format("{:.2f}"))
""")

md(r"""
## 7. Methodological finding: an unseen crash in the validation block inflates the smearing factor

The log-target models (HAR, XGBoost, LSTM, Mamba) are converted back to variance with Duan's smearing factor
S = mean(exp(validation residual)) of the **train-only** model. Originally the 2022 validation block was the last 10% of
pre-cutoff days (2019-12-23 → 2021-09-30). **It contained COVID, but the train-only models were trained only through 2019.**
They badly under-predicted March 2020, their residuals were huge, and S inflated to ≈ 2.2 at h = 22. The refit models,
which *had* seen COVID, inherited that factor, so every 2022 forecast was multiplied by ≈ 2.2.

**Fix (now the main specification):** the 2022 models train through 2020-12-31 and validate on 2021-01-04 → 2021-09-30.
The cutoff and the forecasts are unchanged in design. The old run is archived in `data/processed/archive/val_with_covid/`.
Validating on post-2023 data was rejected: that would be look-ahead, and it overlaps the calm window used for false alarms.

Grey bars = old split (validation with COVID); colored = new split. HARNet has no smearing (S = 1). For GARCH the column shows its fixed RV-scale factor c, not a smearing factor.
""")
code(r"""
comp = E.smearing_comparison(PROJECT_ROOT / "data/processed/archive/val_with_covid/forecasts",
                             resolve_path(cfg["paths"]["forecasts_dir"]), features=features)
comp.to_csv(out_dir / "smearing_finding_2022.csv", index=False)
fig, axes = plt.subplots(2, 2, figsize=(13, 7))
P.plot_smearing_shift(comp, "S", 22, ax=axes[0, 0], ref=1.0, title="Smearing factor S, h = 22")
P.plot_smearing_shift(comp, "S", 5, ax=axes[0, 1], ref=1.0, title="Smearing factor S, h = 5")
P.plot_smearing_shift(comp, "calm_bias_log", 22, ax=axes[1, 0], ref=0.0,
                      title="Calm window 2023-04 → 2025-03: mean ln(forecast / realized), h = 22")
P.plot_smearing_shift(comp, "alarm_to_stress", 22, ax=axes[1, 1], ref=1.0,
                      title="2022 model: alarms ÷ realized stress days (2× normal), h = 22")
fig.tight_layout(); fig.savefig(out_dir / "smearing_finding_2022.png", dpi=150)
""")
code(r"""
cols = ["S", "event_qlike", "event_bias_log", "calm_bias_log", "alarm_to_stress", "peirce", "lead_days_2.0x"]
show = comp[comp["fs"].isin(["ext", "-"])].set_index(["h", "family"]).sort_index()
show = show[[f"{c}__{s}" for c in cols for s in ("val_with_covid", "val_post_covid")]]
show.columns = pd.MultiIndex.from_tuples([tuple(c.split("__")) for c in show.columns])
display(show.style.format("{:.2f}", na_rep="–").set_caption("2022 event, ext set: old split vs new split"))
""")

md(SUMMARY)

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
nbf.write(nb, OUT)
print("wrote", OUT, len(cells), "cells")
