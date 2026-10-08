# XGBoost Implementation Plan — SPY Realized Variance & Drawdown Study

*MASDS Thesis · Elbert Liu · drafted 2026-10-05 · companion to `plan.md` (data pipeline)*

## 1. Purpose and scope

This is a **new build** of the XGBoost realized-variance model in Teller, Pigorsch & Pigorsch (2022, `rv_xgboost.pdf`). The paper has no public code, so nothing is ported. It is built from `xgboost` + `optuna` and adapted to this thesis's design:

| Paper (Teller et al.) | This thesis |
|---|---|
| 16 assets, 2007–2019 | SPY only, 2004–2026 |
| One fixed 80/10/10 split | **Two event-specific splits** (COVID-19 2020, bear market 2022), each with its own training cutoff |
| Four predictor sets (HAR, Lev, CJ, CJX) | **Two feature sets:** `base` (HAR) and `ext` (semivariance + signed jump), shared with HAR, HARNet and Multi-Transformer |
| h ∈ {1, 5, 22, 44} | h ∈ {1, **5**, **22**}. 5 and 22 are the headline horizons for early warning. |
| Goal: average forecast accuracy over 300 test days | Goal: **behaviour before and during a drawdown** (lead time, under-prediction at onset, false alarms) |
| Train on 80%, tune on 10%, test on the last 10% | Tune on a validation block, **refit on all pre-cutoff data**, then forecast day by day with frozen parameters |

**Kept from the paper:**
- log-RV target with direct h-step forecasts
- tree and linear base learners
- the Table 4 hyperparameter ranges
- 100-trial Bayesian optimization using the Tree-Structured Parzen Estimator (TPE)
- MSE / MAE / QLIKE evaluation
- gain-based feature importance

**Inputs:** the frozen feature store `data/processed/spy_features.parquet` and `data/processed/splits/{covid,bear2022}.json` from the data pipeline (Stages 5–6 of `plan.md`), served through `adapters.tabular_view()`.

**Output:** dated out-of-sample forecasts in the shared schema (§8), so the common evaluation script scores XGBoost the same way as the other four models.

---

## 2. Experiment grid

| Dimension | Values | Count |
|---|---|---|
| Event | `covid`, `bear2022` | 2 |
| Feature set | `base`, `ext` | 2 |
| Horizon h | 1, 5, 22 | 3 |
| Base learner | `gbtree` (XGB), `gblinear` (Linear-XGB) | 2 |
| **Total model fits** | | **24** |

Each fit gets 100 Optuna trials, so 2,400 small XGBoost fits on about 4,000 rows each. That's minutes on a laptop CPU, so compute isn't a constraint.

Naming: `xgb-{booster}_{featureset}_h{h}` within an event, e.g. `xgb-gbtree_ext_h22`.

---

## 3. Features and target

### 3.1 Feature sets

The feature sets are the **same for HAR and XGBoost**. That way any difference between them comes from the model, not the inputs.

| Column | Definition (all known at the close of day t) | `base` | `ext` |
|---|---|:-:|:-:|
| `log_rv_d` | ln RV_t | ✓ | ✓ |
| `log_rv_w` | ln of the 5-day mean of RV | ✓ | ✓ |
| `log_rv_m` | ln of the 22-day mean of RV | ✓ | ✓ |
| `log_rs_neg` | ln RS⁻_t | | ✓ |
| `log_rs_neg_w` | ln of the 5-day mean of RS⁻ | | ✓ |
| `log_rs_neg_m` | ln of the 22-day mean of RS⁻ | | ✓ |
| `jump_pct` | ln(1 + ΔJ²_t / RV_t) | | ✓ |

The `ext` set is the log version of HARNet's extended baseline (HARNet eq. 29: RV and RS⁻ at daily/weekly/monthly scales, plus the signed jump). So "extended" means the same thing for HAR, XGBoost and HARNet.

RS⁺ is left out of the core set because RS⁺ = RV − RS⁻ makes it redundant. It stays in the feature store and can be added in a robustness run.

**Optional robustness set `ext_x`** (only if time permits): `ext` + `log_vix`, `ret_cc`, `ret_overnight`, `mom`. This mirrors the paper's CJX set, whose additional predictors helped mainly at h = 1.

### 3.2 Target

For forecast origin t and horizon h:

  `y_h{h}(t) = ln( (1/h) · Σᵢ₌₁ʰ RV_{t+i} )`

- **Direct forecasting:** one separate model per h, with no recursion. This is the paper's approach.
- **Training objective:** `reg:squarederror` on the log target, as in the paper.
- The raw-scale target `rv_h{h}` is kept for evaluation on the variance scale.

### 3.3 Back-transform to variance (important for this thesis)

exp(E[log RV]) under-predicts E[RV], because of Jensen's inequality. For a study about **under-prediction before crashes**, that bias would make every log model look worse than it is, for reasons unrelated to its warning ability. So two forecasts are stored:

- `forecast_var_naive` = exp(ŷ)
- `forecast_var` = exp(ŷ) · S, where S = mean(exp(residualᵢ)) is the **Duan smearing factor** estimated from **validation-block residuals** (out-of-sample, so not overfit).

`forecast_var` is the main forecast. The same correction is applied to the log-HAR model so the comparison stays fair; document this in Methods.

---

## 4. Data splitting, tuning and refit

### 4.1 Split per event (from `splits/{event}.json`)

```
2004-01 ─────────── train ──────────┬─ purge h ─┬── val ──┬ purge h ┤cutoff│ pre-event │ event │ post-event
                                    │           │ (~10%)  │         │      │   OOS forecasts, frozen params
```

| | COVID-19 | 2022 bear |
|---|---|---|
| Train + val ends (cutoff) | 2019-12-31 | 2021-09-30 |
| Validation block | last 10% of pre-cutoff rows: 2018-05-29 → 2019-12-31 | 2021-01-04 → 2021-09-30 (explicit window; train runs through 2020 so COVID is in training. The original 2019-12 → 2021-09 block contained COVID and inflated the smearing factor, see PROJECT_CONTEXT §9d) |
| Forecast period | 2020-01-02 → 2020-06-30 | 2021-10-01 → 2023-03-31 |

**Purging:** any row whose target window [t+1, t+h] crosses a boundary is dropped:
- the last h rows of `train` before `val` starts
- the last h rows of `val` before the cutoff

This rules out target overlap between train and validation, and any leakage past the cutoff. Implement it as `purge(df, boundary_date, h)` and unit-test it (§10).

### 4.2 Hyperparameter search (Optuna, TPE)

- `optuna.samplers.TPESampler(seed=42, multivariate=True)`, 100 trials, single objective.
- **Objective:** validation MSE on the log target, as in the paper. QLIKE on the smeared variance scale is computed for each trial too and stored as a user attribute, so a QLIKE-tuned variant can be selected later without re-running.
- Study storage: `sqlite:///models/xgb/optuna.db`, study name = model name. Resumable, and keeps an audit trail for the appendix.

**Search space** (paper Table 4, mapped to `xgboost` names):

| Paper name | xgboost param | gbtree | gblinear |
|---|---|---|---|
| Shrinkage step size | `learning_rate` | U[0.1, 0.3] | U[0.1, 0.3] |
| L1 weight regularization | `reg_alpha` | U[0, 1] | U[0, 1] |
| L2 weight regularization | `reg_lambda` | U[0, 1] | U[0, 1] |
| Maximum tree depth | `max_depth` | int[1, 20] | n/a |
| Number of trees | `n_estimators` | int[1, 60] | n/a |
| Min. split loss improvement | `gamma` | U[0, 1] | n/a |
| Ratio of input variables per tree | `colsample_bytree` | U[0.5, 1] | n/a |
| Boosting iterations | `n_estimators` | n/a | int[1, 60] |

Fixed settings: `objective="reg:squarederror"`, `tree_method="hist"`, `random_state=42`, `n_jobs=-1`. For `gblinear`: `updater="coord_descent"` and `feature_selector="cyclic"`, which is deterministic.

**Scaling for `gblinear`:** standardize features with mean and standard deviation from the **training block only**. Save the scaler (`scaler.json`) and reuse it at refit and forecast time. Trees don't need scaling.

**Optional sensitivity check (if time permits):** widen the gbtree space to `n_estimators` up to 500 with early stopping, plus `min_child_weight` ∈ [1, 20]. The paper's ≤ 60 trees at lr ≥ 0.1 is a narrow space. Report it as a robustness note, not the main result.

### 4.3 Refit and freeze

1. Take the best trial's parameters.
2. **Refit on train + val** (all pre-cutoff rows after purging), keeping `n_estimators` from the best trial. This **differs from the paper**, which kept the train-only model. The reason: for the drawdown study the model should use the most recent pre-cutoff data. Without refitting, the 2022 model would never be trained on COVID.
3. Freeze and save: `model.json` (`save_model`), `params.json`, `features.json` (ordered column list), `scaler.json` (gblinear only), `smear.json` (S and the residual stats used).
4. **Seed robustness:** refit the gbtree models with 5 seeds (42–46) and the same parameters. The main forecast is the average across the 5 seeds. Report the spread across seeds as a band on the event plots. gblinear with cyclic coordinate descent is deterministic, so one fit is enough.

---

## 5. Out-of-sample forecasting

For each trading day t from the first day after the cutoff to the end of the post-event window:

- Inputs: feature row at t, built only from data up to the close of t. The model never sees data after its cutoff when it is fit, but it does receive each new day's features as they arrive. This is the "frozen parameters, live inputs" design from `plan.md` §3.
- Output: a forecast of mean RV over [t+1, t+h].
- Near the end of the sample, the realized target is incomplete; store `realized_var = NaN` and `target_complete = False`.

Also store **in-sample fitted forecasts for the last 63 trading days before the cutoff** (`sample = "train_tail"`). The evaluation script needs these for the early-warning *lift* (forecast ÷ the model's own pre-cutoff average level).

---

## 6. A tree-specific issue to report: extrapolation ceiling

Tree ensembles predict piecewise-constant values and **cannot forecast above the highest leaf values learned in training**. For a crash study this matters:

- **COVID model:** training includes 2008, so the ceiling is high. It's still binding if March 2020 RV went beyond anything in the training targets.
- **2022 model:** training includes COVID, so the ceiling is unlikely to bind. 2022 was a slow decline, not a spike.

What to do (all cheap):
1. Log `max(train target)` and the model's **maximum achievable prediction**, approximated by the largest prediction over training rows plus a grid of extreme feature values. Put both in `diagnostics.json`.
2. On the event plots, draw the ceiling as a horizontal line, and report how many days the realized target exceeded it.
3. Compare with `gblinear`, which extrapolates linearly. The tree-vs-linear contrast during the COVID spike shows this effect directly and ties to the paper's finding that linear learners do better at longer horizons.

This is a finding, not a bug, and is worth a paragraph in the Discussion.

---

## 7. Interpretability

- **Gain importance** (as in the paper's Table C.7): `booster.get_score(importance_type="total_gain")`, normalized to percent. One table: features × (event, featureset, h).
- **SHAP contributions on the forecast period:** `booster.predict(DMatrix, pred_contribs=True)`. This is built into xgboost, so no extra dependency. Plot per-day contributions over the pre-event and event windows: *what pushed the forecast up before the peak?* For example, did `log_rs_neg` rise before `log_rv_d`? This connects the model's behaviour to the research question.
- **gblinear coefficients:** report them next to HAR OLS coefficients on standardized features. On the `base` set, Linear-XGB should come out similar to HAR. That doubles as a sanity check (§10).

---

## 8. Output schema (shared with all models)

`data/processed/forecasts/{event}/xgb-{booster}_{featureset}_h{h}.parquet` (`paths.forecasts_dir` in `config.yaml`)

| Column | Type | Notes |
|---|---|---|
| `date` | date | Forecast origin t (features known at close of t) |
| `target_start`, `target_end` | date | t+1 and t+h trading days |
| `h` | int | 1, 5, 22 |
| `forecast_log` | float | ŷ, the log forecast |
| `forecast_var` | float | exp(ŷ)·S, the main forecast |
| `forecast_var_naive` | float | exp(ŷ) |
| `forecast_var_seed_min`, `_max` | float | gbtree only: range across 5 seeds |
| `realized_var` | float | (1/h)ΣRV over the target window. NaN if incomplete. |
| `realized_log` | float | ln of `realized_var` |
| `target_complete` | bool | |
| `sample` | str | `train_tail` / `pre_event` / `event` / `post_event` |
| `model`, `booster`, `featureset`, `event` | str | Metadata for joining |

Each model folder `models/xgb/{event}/{model_name}/` also has `params.json`, `features.json`, `smear.json`, `diagnostics.json`, `importance.csv`, `model_seed{42..46}.json`, `val_predictions.parquet`, `contribs.parquet`, and `scaler.json` for gblinear.

---

## 9. Code layout

Inside the `volforecast` package (implemented):

```
src/volforecast/models/xgb/
├── config.py        # loads configs/xgb.yaml (grid, search space, seeds, paths)
├── data.py          # tabular_view wrapper + purge() + train/val/refit/forecast frames
├── space.py         # Optuna search-space functions for gbtree and gblinear
├── tune.py          # run one study → best params (resumable via sqlite)
├── fit.py           # refit on train+val, seeds, smearing factor, save artifacts
├── forecast.py      # out-of-sample forecasts + train_tail → forecasts/*.parquet
├── explain.py       # gain importance, SHAP-contribution plots, gblinear coefficients
├── diagnostics.py   # ceiling check, residual stats, seed spread
└── run_all.py       # loops over the 24-cell grid; --event/--fs/--h/--booster filters
configs/xgb.yaml
tests/test_xgb.py
```

Example `configs/xgb.yaml`:

```yaml
events: [covid, bear2022]
featuresets:
  base: [log_rv_d, log_rv_w, log_rv_m]
  ext:  [log_rv_d, log_rv_w, log_rv_m, log_rs_neg, log_rs_neg_w, log_rs_neg_m, jump_pct]
horizons: [1, 5, 22]
boosters: [gbtree, gblinear]
optuna: {n_trials: 100, seed: 42, objective: val_mse_log}
refit: {on: train_val, seeds: [42, 43, 44, 45, 46]}
smearing: {source: val_residuals}
train_tail_days: 63
paths: {features: data/processed/spy_features.parquet, splits: data/processed/splits,
        models: models/xgb, forecasts: forecasts}
```

Command: `uv run python -m volforecast.models.xgb.run_all` (whole grid), or for example `--event covid --fs ext --h 22 --booster gbtree` for one cell. Robustness variants: `--fs ext_x`, `--space wide`, `--val-window START END`, each with `--tag`.

---

## 10. Tests and sanity checks

Run these before trusting any result:

| # | Check | How |
|---|---|---|
| 1 | **No look-ahead** | For random rows, recompute `y_h{h}` from `spy_daily_measures` and assert it uses only RV from t+1…t+h. Assert every feature at t equals its value recomputed from data ≤ t. |
| 2 | **Purge works** | Assert max(`target_end`) of training rows < first `date` in validation, and max(`target_end`) of pre-cutoff rows ≤ cutoff. |
| 3 | **Frozen model** | Model file hash is unchanged between refit and the end of forecasting. Forecast code never calls `fit`. |
| 4 | **Reproducibility** | Re-running a cell with the same seed gives identical forecasts (gbtree with `hist` on CPU is deterministic for a fixed seed and thread count). |
| 5 | **Linear-XGB ≈ HAR** | On `base`, h = 1, with a weak penalty (`reg_alpha` = `reg_lambda` = 0, 60 iterations, lr 0.3), gblinear coefficients should roughly match HAR OLS (in the same standardized units). |
| 6 | **Beats naive** | Validation MSE of every tuned model < the random-walk forecast (`log_rv_d` for h=1, `log_rv_w` for h=5, `log_rv_m` for h=22). If not, something's wrong. |
| 7 | **Smearing sanity** | S is in about [1.05, 1.5]. Mean of `forecast_var` in validation ≈ mean realized, within ~10%. |
| 8 | **Schema** | The forecast parquet validates against §8 (columns, dtypes, no duplicate dates, `sample` labels match the split file). |

`tests/test_xgb.py` covers 1–4 and 8 with `pytest`. Checks 5–7 go in `diagnostics.json` and are reviewed by eye.

---

## 11. Thesis-specific design notes

- **(Resolved 2026-10-08; see PROJECT_CONTEXT §9d.) The original 2022 validation block contained COVID.** That makes the 2022 tuning objective dominated by an extreme period, which may push toward simpler models. It's realistic for a forecaster in late 2021. As a robustness check, re-tune with the validation block ending 2020-01-31, then train → refit as usual. Report only if the results differ.
- **Equal tuning effort:** 100 TPE trials matches what the deep models should get. Record wall-clock time per cell for the computational-cost table.
- **What XGBoost adds to the story:**
  - Does non-linearity help *early warning*, even if it doesn't help average accuracy at h = 5/22, as the paper found?
  - Does the tree ceiling hurt during spikes?
  - Does `ext` (downside semivariance) raise forecasts earlier than `base` in the run-up?

  These are three concrete sub-questions for the Results chapter.
- **Statistical comparison:** model confidence sets (Hansen et al. 2011) and Diebold–Mariano tests belong to the shared evaluation stage, not to this build. The forecast schema has everything they need.

---

## 12. Task checklist

**Prerequisites:** data pipeline Stages 1–8 done and `spy_features.parquet` frozen.

- [ ] `configs/xgb.yaml` + `config.py`
- [ ] `data.py` with `purge()`, plus tests 1–2
- [ ] `space.py` + `tune.py`. Smoke run: covid / base / h=1 / gbtree with 10 trials.
- [ ] `fit.py`: refit, seeds, smearing, artifacts
- [ ] `forecast.py` and the schema test (8)
- [ ] Sanity checks 5–7 on the smoke cell
- [ ] Full grid: `run_all.py` (24 cells × 100 trials)
- [ ] `diagnostics.py`: ceiling check, seed spread
- [ ] `explain.py`: importance table, SHAP-contribution plots for h=5 and h=22 in both events
- [ ] Robustness (if time permits): COVID-free validation for 2022, wider gbtree space, `ext_x` features
- [ ] Hand forecasts to the shared evaluation script

**Dependencies:** `xgboost>=2.0`, `optuna>=3.5`, `pandas`, `pyarrow`, `numpy`, `scikit-learn` (StandardScaler only), `pyyaml`, `matplotlib`, `pytest`.

---

## 13. References

- Teller, A., Pigorsch, U., & Pigorsch, C. (2022). Short- to long-term realized volatility forecasting using extreme gradient boosting. SSRN 4267541. (`rv_xgboost.pdf`: Table 4 search space, §4 procedure, Table C.7 importance)
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD*.
- Bergstra, J., Bardenet, R., Bengio, Y., & Kégl, B. (2011). Algorithms for hyper-parameter optimization (TPE). *NeurIPS*.
- Duan, N. (1983). Smearing estimate: A nonparametric retransformation method. *JASA*, 78(383).
- Patton, A. J., & Sheppard, K. (2015). Good volatility, bad volatility: Signed jumps and the persistence of volatility. *REStat*.
- Reisenhofer, R., Bayer, X., & Hautsch, N. (2022). HARNet. arXiv:2205.07719 (`harnet.pdf`, eq. 29)
- Hansen, P. R., Lunde, A., & Nason, J. M. (2011). The model confidence set. *Econometrica*.
