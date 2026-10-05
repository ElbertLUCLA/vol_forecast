# deep-learning-volatility-forecast

Data pipeline and models for a MASDS thesis comparing GARCH, HAR, XGBoost, HARNet
and Multi-Transformer forecasts of SPY realized volatility around the COVID-19
crash and the 2022 bear market. Full design rationale is in [`plan.md`](plan.md).

## Setup

```bash
uv sync
cp .env.example .env     # then fill in MASSIVE_API_KEY
```

(`MASSIVE_API_KEY` can also just be set as a regular environment variable instead
of using `.env`.)

## Running the pipeline

```bash
# Quick smoke test on one month first
uv run python scripts/run_pipeline.py --test-month 2020-03

# Full history (2004-01 to 2026-09); re-run anytime, it's idempotent
uv run python scripts/run_pipeline.py
```

This runs, in order: ingest raw bars from the Massive API -> build the NYSE
trading calendar -> clean the intraday 5-min grid -> compute daily realized
measures (RV, RS+/-, BPV, medRV, ...) -> build the feature store and
multi-horizon targets -> write per-event train/val/event splits -> write a QC
report to `reports/qc/`.

Each stage can also be run on its own, e.g.:

```bash
uv run python -m volforecast.ingest_massive --start 2020-03-01 --end 2020-03-31
uv run python -m volforecast.calendar
uv run python -m volforecast.clean_intraday
uv run python -m volforecast.realized_measures
uv run python -m volforecast.build_features
uv run python -m volforecast.make_splits
uv run python -m volforecast.qc_report
```

Config (date ranges, event cutoffs, feature windows, paths) lives in
`config.yaml`.

## Model adapters

`src/volforecast/adapters.py` serves the one feature store
(`data/processed/spy_features.parquet`) to each model family without
duplicating logic:

- `garch_view` — return series for `arch`
- `tabular_view` — X/y DataFrames for HAR (statsmodels) and XGBoost
- `harnet_view` — raw daily RV (+ RS-, signed jump) series for HARNet
- `sequence_view` — lagged 3-D arrays for the Multi-Transformer

Run `uv run pytest` to smoke-test that GARCH(1,1) and HAR-OLS actually fit on
the transformed output (requires the pipeline to have been run at least once).

## XGBoost

A new implementation of Teller, Pigorsch & Pigorsch (2022), adapted to the two-event
design. The design is in [`XGBoost_implementation.md`](XGBoost_implementation.md),
the code is in `src/volforecast/models/xgb/`, and the settings are in `configs/xgb.yaml`.
Dates, cutoffs and paths still come from `config.yaml` and the split files.

```bash
# Smoke run: one event, one feature set, one horizon, 10 trials
uv run python -m volforecast.models.xgb.run_all --event covid --fs base --h 1 --n-trials 10

# Full grid: 2 events x {base, ext} x h {1, 5, 22} x {gbtree, gblinear}, 100 TPE trials each
uv run python -m volforecast.models.xgb.run_all

# Robustness variants (need --tag so they don't overwrite the main results)
uv run python -m volforecast.models.xgb.run_all --fs ext_x --tag ext_x
uv run python -m volforecast.models.xgb.run_all --space wide --booster gbtree --tag wide
uv run python -m volforecast.models.xgb.run_all --event bear2022 --val-window 2018-01-02 2020-01-31 --tag covidfreeval
```

Each cell writes:

- `data/processed/forecasts/{event}/xgb-{booster}_{fs}_h{h}.parquet`: forecasts in the shared schema
- `models/xgb/{event}/{model}/`: the frozen model and its metadata (`model_seed*.json`, `params.json`, `features.json`, `smear.json`, `scaler.json`, `diagnostics.json`, `importance.csv`, `contribs.parquet`)
- `reports/xgb/{event}/`: forecast plots with the tree ceiling, and SHAP-contribution plots
- Optuna studies in `models/xgb/optuna.db`. Studies resume, so re-running tops each one up to `n_trials`.
- A per-cell run log in `models/xgb/run_log.csv`

## Reference repos

`HARNet/` and `MultiTransformer/` are the original authors' repos (TensorFlow /
Keras), kept as read-only references for the adapter contracts and are
git-ignored from this repo. They are likely candidates for a PyTorch rewrite
later — the adapters above don't assume either framework.
