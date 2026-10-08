# deep-learning-volatility-forecast

Data pipeline and models for a MASDS thesis comparing GARCH, HAR, XGBoost, HARNet
and Mamba forecasts of SPY realized volatility around the COVID-19
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
- `sequence_view` — lagged 3-D arrays for sequence models

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

## HARNet (PyTorch)

A PyTorch port of Reisenhofer, Bayer & Hautsch (2022), adapted to the event design.
The code is in `src/volforecast/models/harnet/` and the settings are in `configs/harnet.yaml`.
The TensorFlow reference in `HARNet/` doesn't run on the current stack (Keras 3, NumPy 2).
The model has about 10 weights, so it runs on the CPU build of PyTorch. A GPU (or ROCm) build would not make it faster.

```bash
uv run python -m volforecast.models.harnet.run_all                        # 2 events x {base, ext} x h {1, 5, 22}
uv run python -m volforecast.models.harnet.run_all --event covid --fs base --h 1 --max-iterations 2000
```

Each cell:

- initializes from the OLS HAR(1, 5, 20) fit on the direct target `rv_h{h}`
- trains with Adam and QLIKE on 4 windows × 5 labels per step, with 5 seeds trained together as one vectorized model
- picks the number of iterations on validation QLIKE; iteration 0 is the HAR itself
- refits on all pre-cutoff rows for that many iterations, then freezes

Forecasts go to the shared schema in `data/processed/forecasts/{event}/harnet20_{fs}_h{h}.parquet`, and the frozen model to `models/harnet/{event}/{model}/`.

## Baselines, sequence models and evaluation

```bash
uv run python -m volforecast.models.har.run_all       # log-HAR (OLS + smearing), 12 cells
uv run python -m volforecast.models.garch.run_all     # GARCH(1,1)-t, 6 cells
uv run python -m volforecast.models.seq.run_all --arch lstm --device cpu --threads 2   # LSTM control (CPU)
.venv-rocm/Scripts/python -m volforecast.models.seq.run_all --arch mamba             # Mamba (AMD GPU)
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/02_model_comparison.ipynb
```

- **Mamba** (`src/volforecast/models/seq/`) runs on the RX 7900 XTX through a separate environment, `.venv-rocm`, which has
  AMD's Windows PyTorch 2.9.1 + ROCm 7.2.1 wheels; setup steps are in `PROJECT_CONTEXT.md` §3. It also runs on CPU, just slower.
  The runner skips finished cells, so it resumes after a crash.
- **Forecast range.** Every model forecasts from its cutoff through a calm window after the event
  (`calm_window` in `config.yaml`). That window is the false-alarm reference.
- **Re-forecasting.** `--forecast-only` (XGBoost, HAR) re-forecasts from the frozen models without refitting.
- **Evaluation.** `src/volforecast/evaluation.py` computes lift, lead time, onset coverage and escalation, 2022 tracking,
  alarm skill (hit rate, false-alarm rate, Peirce score) and DM tests. `notebooks/02_model_comparison.ipynb` shows the results,
  and `reports/eval/` gets the tables and figures.

## Reference repos

`HARNet/` is the original authors' TensorFlow repo, kept read-only and git-ignored. It doesn't run on
the current stack, which is why `src/volforecast/models/harnet/` is a PyTorch port. The Multi-Transformer
reference repo was removed when Mamba replaced that model.
