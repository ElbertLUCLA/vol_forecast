# Data Pipeline Plan — SPY Realized Volatility & Drawdown Study

*MASDS Thesis · Elbert Liu · drafted 2026-10-05*

## 1. Project context

**Research question:** Which volatility model gives the best early warning of a large market drawdown? The study compares two kinds of drawdown:

| Event | Character | Market peak (S&P 500) | Trough |
|---|---|---|---|
| COVID-19 crash | Fast, sudden crash | 2020-02-19 | 2020-03-23 |
| 2022 bear market | Slow, drawn-out decline | 2022-01-03 | 2022-10-12 |

**Asset:** SPY only (for this stage).

**Models and the inputs each needs**

| Model | Type | Library / code | Inputs it needs |
|---|---|---|---|
| GARCH(1,1) | Classical baseline | `arch` | Daily log returns (close-to-close) |
| HAR | Linear RV model | `statsmodels` OLS | log RV daily/weekly/monthly (+ extended set) |
| XGBoost | Classical ML | `xgboost` + `optuna` (TPE), following Teller et al. | Tabular HAR features (+ extended set) |
| HARNet | Deep learning (dilated CNN) | github.com/mdsunivie/HARNet (TF), or a PyTorch rewrite | Raw daily RV series (+ RS⁻ and signed jump for the extended version) |
| Multi-Transformer | Deep learning (attention) | github.com/EduardoRamosP/MultiTransformer (Keras), as a reference | 10-lag sequences of the feature set |

**Feature strategy (decided):** two tiers.

1. **Base set:** log RV_d, RV_w, RV_m. Every RV model uses it.
2. **Extended set:** positive and negative realized semivariance plus signed jump variation (Patton & Sheppard 2015). This matches HARNet §6 and the XGBoost paper's HAR-Lev/HAR-CJ specifications.

VIX, overnight returns and momentum are computed and stored for optional robustness checks, but they are not in the core comparison.

**Data source:** 5-minute SPY bars from the Massive (formerly Polygon.io) REST API. Bloomberg is used only for cross-checks and optional VIX.

---

## 2. Data source details: Massive Aggregates API

- **Endpoint:** `GET https://api.massive.com/v2/aggs/ticker/SPY/range/5/minute/{from}/{to}`
- **Query params:** `adjusted=true` (SPY has never split, so this has no effect), `sort=asc`, `limit=50000` (the maximum). Follow `next_url` if it is present.
- **Fields per bar:** `t` (bar start, Unix ms, UTC), `o`, `h`, `l`, `c`, `v`, `vw`, `n` (trade count).
- **History:** goes back to 2003-09-10.
- **Extended hours are included.** Pre-market and after-hours bars come back too, so the pipeline must filter to regular trading hours.
- **Empty bars are omitted.** If no trades happen in a 5-minute window, no bar is returned. The pipeline must rebuild the full intraday grid.
- **Daily bars:** the same endpoint with `range/1/day` gives the official daily OHLC, used for GARCH returns and cross-checks.
- **Volume:** about 4,000 days × ~190 bars (all sessions) ≈ 760k rows. That's small. Request in **monthly chunks** (~4k bars each, well under the 50k limit).
- **API key:** stored in a `.env` file as `MASSIVE_API_KEY`, read with `python-dotenv`. Never commit it, and add `.env` to `.gitignore`.

---

## 3. Sample period

| Item | Choice | Reason |
|---|---|---|
| Download range | 2004-01-02 → 2026-09-30 | Full history. Costs almost nothing and allows extra events later. |
| Modelling start | 2004-01-02 (first 22 days are burn-in for RV_m; HARNet's 80-day receptive field consumes more) | Includes the 2008 crisis in training, which matters: models need to have seen a high-volatility regime. |
| Event 1 analysis window | 2019-07-01 → 2020-06-30 | Covers calm before, crash, and recovery. |
| Event 2 analysis window | 2021-07-01 → 2023-03-31 | Covers the peak, the slow grind down, and the trough. |

### Train/test cutoffs per event

To judge whether a model *anticipates* an event, its forecasts in the run-up must be out-of-sample. So the training cutoff has to sit **before the pre-event window**, not on the day before the peak.

| | COVID-19 | 2022 bear |
|---|---|---|
| Train (incl. validation) ends | 2019-12-31 | 2021-09-30 |
| Validation block (last ~10% of the pre-cutoff sample, for XGBoost/DL tuning) | ≈ 2017-10 → 2019-12 | ≈ 2019-07 → 2021-09 ⚠ contains COVID |
| Pre-event window (out-of-sample) | 2020-01-02 → 2020-02-19 | 2021-10-01 → 2022-01-03 |
| Event window | 2020-02-20 → 2020-03-23 | 2022-01-04 → 2022-10-12 |
| Post-event window | 2020-03-24 → 2020-06-30 | 2022-10-13 → 2023-03-31 |

- Model parameters are **frozen at the cutoff**. Forecasts after it are made day by day using the actual lagged data as it arrives. No refitting, so no look-ahead.
- The 2022 training set includes COVID. That's realistic: a forecaster in late 2021 would have seen it. Note in the thesis that the validation block for event 2 contains an extreme regime. One alternative is ending validation before 2020-02, but the default is to keep it.
- All cutoffs live in `config.yaml`, so they can be changed (e.g. for a sensitivity analysis with a 1-month vs 3-month pre-window) without editing code.

---

## 4. Pipeline architecture

```
thesis/
├── .env                      # MASSIVE_API_KEY (git-ignored)
├── config.yaml               # tickers, dates, cutoffs, horizons, paths
├── data/
│   ├── raw/                  # immutable API responses
│   │   ├── spy_5min/YYYY-MM.parquet
│   │   └── spy_daily/spy_daily_raw.parquet
│   ├── interim/
│   │   └── spy_5min_rth.parquet    # cleaned regular-hours grid
│   ├── processed/
│   │   ├── spy_daily_measures.parquet   # one row per trading day
│   │   ├── spy_features.parquet         # features + targets
│   │   └── splits/{covid,bear2022}.json # index ranges per window
│   └── external/             # optional: VIX (Bloomberg), cross-check files
├── src/
│   ├── ingest_massive.py     # Stage 1
│   ├── calendar.py           # Stage 2
│   ├── clean_intraday.py     # Stage 3
│   ├── realized_measures.py  # Stage 4
│   ├── build_features.py     # Stage 5
│   ├── make_splits.py        # Stage 6
│   ├── qc_report.py          # Stage 7
│   └── adapters.py           # Stage 8: model-specific views
├── notebooks/
│   └── 01_data_qc.ipynb
└── reports/qc/               # QC figures and tables
```

Each stage reads only the previous stage's output and writes to disk. Any stage can be re-run without downloading again. Parquet via `pyarrow` throughout.

---

## 5. Stage-by-stage plan

### Stage 1: Ingest (`ingest_massive.py`)

- For each month from 2004-01 to 2026-09, request 5-minute bars, all sessions, and save the raw results to `data/raw/spy_5min/YYYY-MM.parquet`.
- **Idempotent:** skip months whose file already exists (add a `--force` flag to refetch). Re-fetch the current/last month at the end, because it may have been incomplete.
- Retry with exponential backoff on 429/5xx errors. Log `request_id`, `resultsCount` and a timestamp for each call to `data/raw/ingest_log.csv`.
- Follow `next_url` pagination even though monthly chunks shouldn't need it.
- Also pull daily bars (`range/1/day`) for the full range into `spy_daily_raw.parquet`.
- *Optional:* pull SPY dividends from Massive's reference endpoint if total-return GARCH is wanted. The effect is small (~0.4% on ex-dividend days, four times a year), so the default is price returns, noted in the thesis.
- Either `requests` or the official Massive Python client works. `requests` is fine and more transparent for a thesis appendix.

### Stage 2: Trading calendar (`calendar.py`)

- Use `pandas_market_calendars` (`"NYSE"`) to get every session's open and close, **including early closes** (13:00 ET on the day after Thanksgiving, Christmas Eve, sometimes July 3).
- This is the master list of trading days. Any day in the calendar with no data, or data on a day not in the calendar, gets flagged.

### Stage 3: Clean intraday data (`clean_intraday.py`)

1. Convert `t` (UTC ms) → `America/New_York` with timezone awareness. This handles daylight saving correctly; never hard-code UTC offsets.
2. Keep only bars whose **start** time is in [market open, market close − 5 min]. A normal day gives 09:30 … 15:55, which is **78 bars**. An early close gives 42 bars.
3. Re-index each day to its full 5-minute grid. For missing bars: forward-fill the close as the price (zero return in that slot) and record `n_missing`.
4. Intraday log returns:
   - r₁ = ln(c₀₉:₃₀ / o₀₉:₃₀): the first bar's own open-to-close, which captures the opening move within the session
   - rⱼ = ln(cⱼ / cⱼ₋₁) for j = 2…78
5. Flags per day: `n_bars_observed`, `n_missing`, `is_early_close`, `max_abs_5min_return`.
6. **Day exclusion rule (set in config):** drop a day if `n_missing > 10%` of its grid. Early-close days stay by default with a flag. Their RV covers fewer hours, so test excluding them as a robustness check. For SPY, missing bars should be close to zero.

### Stage 4: Daily realized measures (`realized_measures.py`)

All of these come from the 78 (or 42) regular-session returns. Units: decimal returns squared (daily variance).

| Measure | Formula | Used by |
|---|---|---|
| RV | Σ rⱼ² | Everyone (target and features) |
| RS⁺ | Σ rⱼ² · 1[rⱼ > 0] | Extended set |
| RS⁻ | Σ rⱼ² · 1[rⱼ < 0] | Extended set (HARNet §6, HAR-Lev) |
| ΔJ² (signed jump) | RS⁺ − RS⁻ | Extended set |
| BPV | (π/2) · Σ \|rⱼ\|\|rⱼ₋₁\| | Optional jump test |
| medRV | Andersen et al. (2012) formula | Optional (XGBoost paper's HAR-CJ) |
| Open-to-close return | ln(c_close / o_open) | Diagnostics |
| Overnight return | ln(o_open,t / c_close,t−1) | Optional (CJX-style robustness) |
| Close-to-close return | ln(C_t / C_{t−1}) from daily bars | GARCH input |

- Sanity identity: RS⁺ + RS⁻ = RV, checked to floating-point precision.
- **Scale note for GARCH comparability:** GARCH models close-to-close variance (which includes the overnight gap), while RV covers 09:30–16:00 only. Store both, and decide in the modelling stage between (a) comparing GARCH to RV with a scaling factor estimated on training data (e.g. mean(r²_cc)/mean(RV)), or (b) using a QLIKE-based ranking. Record the factor in the QC report.
- `arch` works best with returns in percent (×100). Store a `ret_cc_pct` column so nobody has to remember this.

### Stage 5: Features and targets (`build_features.py`)

Everything indexed by trading day *t*. **Features use information up to the close of day t. Targets start at t+1.**

**Base features (all RV models):**
- `log_rv_d` = ln RV_t
- `log_rv_w` = ln( (1/5) Σᵢ₌₀⁴ RV_{t−i} )
- `log_rv_m` = ln( (1/22) Σᵢ₌₀²¹ RV_{t−i} )

**Extended features:**
- `log_rs_neg` = ln RS⁻_t, `log_rs_pos` = ln RS⁺_t
- `log_rs_neg_w`, `log_rs_neg_m` (5- and 22-day averages, as in HARNet eq. 29)
- `jump_pct` = ln(1 + ΔJ²_t / RV_t), the Patton–Sheppard transform that keeps the log-safe version defined when ΔJ² < 0
- `sj_raw` = ΔJ²_t (for non-log models)

**Optional / robustness:** `ret_cc`, `ret_overnight`, `mom` = RV_w − RV_m, `log_medrv`, `log_vix` (Bloomberg).

**Inputs for the original Multi-Transformer specification** (to reproduce or compare against the paper's setup): `ret_cc` and the 5-day rolling standard deviation of returns.

**Targets** (direct forecasting, as in the XGBoost paper), for h ∈ {1, 5, 22}:
- `y_h{h}` = ln( (1/h) Σᵢ₌₁ʰ RV_{t+i} ), with the raw-scale version `rv_h{h}` also stored for QLIKE and MSE on the variance scale.
- For drawdown anticipation, **h = 5 and h = 22 are the headline horizons**: a warning one week or month ahead is more useful than a next-day one. Keep h = 1 for comparability with the literature.

**Zero-RV guard:** if any day has RV = 0 or RS⁻ = 0 (possible on a half-day), floor at a small ε (e.g. the 0.1th percentile of the training RV) before taking logs, and log how many days were floored.

### Stage 6: Splits (`make_splits.py`)

- Write `splits/covid.json` and `splits/bear2022.json`, each with the date ranges for `train`, `val`, `pre_event`, `event` and `post_event` from §3.
- Rows whose target window crosses the training cutoff (e.g. a training row at 2019-12-20 whose h = 22 target runs into January 2020) are **dropped from training**. This prevents leakage at the boundary. Purge h rows before each cutoff.
- Any normalization (e.g. HARNet's min-max scaling of log RV to [0, 1], or a deep-learning standard scaler) is **fit on the training split only** and saved with the split.

### Stage 7: Quality control (`qc_report.py` → `reports/qc/`)

| Check | Pass criterion |
|---|---|
| Bars per day | ≥ 99% of normal days have exactly 78 bars, and early closes have 42 |
| Calendar alignment | No NYSE session missing from the data. No data on non-sessions. |
| Close cross-check | Last RTH 5-min close vs daily-bar close: median absolute difference < 0.05% (they differ slightly because of the closing auction) |
| Bloomberg cross-check | SPY daily closes vs Bloomberg `PX_LAST` on a sample of dates, ideally all of them |
| Semivariance identity | RS⁺ + RS⁻ = RV |
| Outlier review | Top 20 days by RV and by max \|5-min return\|. Each should map to a known event (e.g. 2008-10, 2010-05-06 flash crash, 2015-08-24, 2020-03-16). Investigate any that don't. |
| Plausibility | Annualized mean √(252·RV) ≈ 13–17%. Corr(√(252·RV), VIX) is high (~0.8+). |
| Stylized facts | Plot log RV with both event windows shaded. ACF shows slow decay. log RV is roughly Gaussian (histogram/QQ plot). These become a figure in the Data chapter. |

The QC notebook doubles as the source for the thesis **Data** chapter: descriptive statistics like Table 1 of the XGBoost paper, the RV time-series plot, and the ACF.

### Stage 8: Model adapters (`adapters.py`)

One feature store, with thin functions that serve each model what it expects:

- `garch_view(split)` → series of `ret_cc_pct` for the training range, plus the out-of-sample dates for rolling 1/5/22-step variance forecasts with frozen parameters.
- `tabular_view(split, featureset, h)` → `X`, `y` DataFrames for HAR and XGBoost.
- `harnet_view(split, featureset)` → the raw daily `log_rv` (and `log_rs_neg`, `jump_pct`) series. HARNet builds its own 5/20/(40/80)-day aggregations inside the network, so it gets the **daily series, not the precomputed averages**.
- `sequence_view(split, featureset, lags=10)` → a 3-D array (samples × 10 lags × features) for the Multi-Transformer.

This keeps "the same information set" true across models by construction, which is an argument you'll want to make in the Methods chapter.

---

## 6. Evaluation implications for the data (decided now, used later)

"Which model predicted the highest variance before the drawdown" needs care: a model that *always* forecasts high variance would win it. To make the comparison fair, the processed data should support these measures, all computed on the out-of-sample pre-event and event windows:

1. **Forecast accuracy in each window:** QLIKE and MSE of the forecast vs realized, per horizon. QLIKE penalizes under-prediction heavily, which suits a risk question.
2. **Early-warning lift:** each model's forecast divided by *its own* average forecast over the last 3 months of training. Reported as the date the ratio first passes, e.g., 1.5× or 2×, relative to the market peak. This gives a lead time in days.
3. **Under-prediction at the onset:** cumulative forecast error over the first 10 days of the event window.
4. **False alarms:** how often the same threshold is crossed during calm out-of-sample periods. Without this, lead-time results can't be interpreted.
5. *(Optional)* **VaR backtest:** 1-day 99% VaR exceedances during the event window, using each model's variance forecast. This connects to the Multi-Transformer paper's risk-management framing.

The pipeline doesn't compute these, but it must output **dated, out-of-sample forecasts for every model** (`forecasts/{event}/{model}_{featureset}_h{h}.parquet`, with columns `date, forecast, realized`). That fixed schema lets a single evaluation script handle all models.

---

## 7. Task checklist

**Setup**
- [ ] Create the project repo and folder structure (§4), `config.yaml`, `.env`, `.gitignore`
- [ ] Environment: `python ≥3.10`, `pandas`, `pyarrow`, `requests`, `python-dotenv`, `pandas_market_calendars`, `numpy`, `matplotlib`, `statsmodels`, `arch`, `xgboost`, `optuna`, plus TensorFlow or PyTorch for later

**Data**
- [ ] Stage 1: test one month (2020-03), check the raw response by eye, then run the full download
- [ ] Stage 2: calendar table
- [ ] Stage 3: clean RTH grid and missing-bar report
- [ ] Stage 4: daily realized measures
- [ ] Stage 5: feature store and targets
- [ ] Stage 6: split files for both events
- [ ] Stage 7: QC report. Fix anything that fails before modelling.
- [ ] Optional: pull VIX and SPY `PX_LAST` from Bloomberg into `data/external/`
- [ ] Stage 8: adapters, plus a smoke test (fit HAR on the COVID split and print coefficients; they should look like the usual HAR estimates, roughly β_d≈0.3–0.4, β_w≈0.3–0.4, β_m≈0.2)

**Hand-off to modelling**
- [ ] Freeze `spy_features.parquet` with a version tag and record row count and checksum in the thesis appendix

---

## 8. Open decisions

| # | Decision | Default in this plan |
|---|---|---|
| 1 | Pre-event window length | ~7 weeks (COVID), ~3 months (2022). Sensitivity test with alternatives. |
| 2 | Early-close days | Keep and flag. Robustness test excluding them. |
| 3 | GARCH vs RV scale | Scaling factor estimated on training data. Report QLIKE as well. |
| 4 | Validation for event 2 containing COVID | Keep (realistic). Mention it in limitations. |
| 5 | Price vs total return for GARCH | Price returns |
| 6 | VIX in robustness set | Only if time permits (Bloomberg pull is quick) |

## 9. References for the pipeline

- Corsi (2009): HAR model
- Patton & Sheppard (2015): semivariance, signed jump variation, log(1+ΔJ²/RV) transform
- Andersen, Dobrev & Schaumburg (2012): medRV
- Reisenhofer, Bayer & Hautsch (2022): HARNet (`harnet.pdf`, §6 for extended inputs)
- Teller, Pigorsch & Pigorsch (2022): XGBoost RV forecasting (`rv_xgboost.pdf`)
- Ramos-Pérez, Alonso-González & Núñez-Velázquez (2021): Multi-Transformer (`multi_transformer.pdf`)
- Massive API docs: Custom Bars endpoint, `api.massive.com/v2/aggs/...`
