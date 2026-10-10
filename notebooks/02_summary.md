> **Superseded (2026-10-09).** Written before the Mamba 2022 re-run. Findings now live in the paper plan
> (`Graduate Thesis/paper_plan.md`, §1.5, §5, §6); numbers come from `reports/eval/`.

## 8. Summary of findings

Numbers are for the `ext` feature set unless noted. "Normal" = the long-run training-sample RV (≈ 14–15% annualized vol).
The 2022 models use the corrected split (trained through 2020, validated on 2021-01 → 2021-09; §7).
**The Mamba 2022 cells are pending a re-run on that split**, so Mamba appears only in the COVID results for now.

### Q1. COVID-19: who warned, and who said "it will continue"?

- **No model gave a real warning before the 19 Feb peak.** Every model first crossed 1.5× normal on 24–28 Feb, after the
  peak. GARCH was first (24 Feb), and the h = 22 LSTM and XGBoost were last.
- **After the first drop, HARNet was the model that said "this will get worse".** Over the first 10 event days at h = 5:
  - HARNet's forecasts stood 1.28× above the last week's realized level (escalation), against 0.92–1.09 for every other
    model including Mamba.
  - It anticipated 52% of the variance that followed (coverage), against 30–38% for the rest.
  - It beats HAR on event QLIKE at h = 5 (DM p = 0.013).
  - Mamba ranks second on escalation (1.33 at h = 1, 1.09 at h = 5), ahead of the LSTM (1.15 / 0.99).
- **At h = 22 every model badly under-forecast the crash** (coverage ≤ 17%). The bounded-output models were worst:
  - LSTM ceiling: 47–50% vol, exceeded on 17–18 of 23 event days.
  - Mamba ceiling: 37–41% vol, exceeded on 20–22 days.
  - Tree ceiling: ≈ 26% vol.

  The level models GARCH and HARNet reached 80% and 71% vol, but only with a lag.

### Q2. 2022 bear market: who tracked the slow decline?

- **Warning before the 3 Jan 2022 peak.** At h = 1 and 5, HARNet, the LSTM and XGB-tree crossed 2× normal on
  1–3 Dec 2021, about 20 trading days before the peak, after the Omicron spike. At h = 22 only HARNet rose early (1.5× on
  1 Dec). GARCH stayed below 2× until March 2022.
- **Over the nine-month decline, the level models tracked best at the monthly horizon.** At h = 22, against a realized
  mean of 19.1% vol:
  - HARNet averaged 18.5% (QLIKE 0.169) and GARCH 18.4% (0.175).
  - The log-target models stayed near *normal*: HAR 16.7%, XGB-tree 15.1%, LSTM 14.8%, XGB-linear 13.9%.
    XGBoost and LSTM are significantly worse than HAR (DM p ≤ 0.038).
  - At h = 1 and 5 the models are close; the LSTM is best (QLIKE 0.147 / 0.118).

### Q3. Accurate, or alarm-happy?

- **COVID: HARNet's edge is genuine.** It has the best alarm skill at h = 1 (Peirce 0.70 at 2× normal) and raises alarms
  about as often as stress occurs (alarm-to-stress 1.07–1.18).
- **2022, corrected split: no model is alarm-happy any more.** Calm-window bias at h = 22 is about 0 for the log models
  (−0.07 to +0.02) and +0.17 / +0.25 for GARCH / HARNet. The models that alarm well at h = 1 (LSTM Peirce 0.53, XGB-tree
  0.49) do so with a 4–6% false-alarm rate. At h = 22, however, the log models *under-warn*: they almost never reach 2×
  normal during 2022 (alarm-to-stress ≤ 0.21), so their alarm skill is about 0. HARNet and GARCH keep modest skill
  (0.19 / 0.24).
- **Before the correction, the opposite looked true.** With the COVID-contaminated smearing factor (S ≈ 2.2), the same
  log models looked like the best 2022 trackers and earliest warners. That was a uniform ≈ 2.2× over-forecast, visible as
  a +0.8 log bias in the calm window afterwards (§7). The same h = 22 QLIKE can come from opposite errors: HAR scores
  0.19 inflated and 0.20 corrected, with bias +0.52 vs −0.26.

### What this means for the thesis

1. **Fast crash:** none of these models predicts a crash from a calm market. What separates them is how fast and how far
   they react after the first drop. Level-trained models (HARNet, GARCH) react most. Bounded models (trees, LSTM, Mamba)
   are structurally capped at long horizons.
2. **Slow decline:** level models track a persistent high-volatility regime best at the monthly horizon. Log-target
   models mean-revert too strongly at h = 22.
3. **Methodological:**
   - A retransformation (smearing) factor estimated on a validation block that contains a shock unseen in training
     inflates every forecast. That inflation can masquerade as early-warning skill.
   - The fix (shock inside training) has its own caveat: validating on a calm 2021 favors models that expect calm. Report
     both splits; the notebook's §7 table does this.
4. **Long memory:** Mamba improves on the LSTM in the COVID onset but does not beat HARNet. In 2022 the LSTM now picks a
   126-day lookback in 4 of 6 cells, so long memory matters more once COVID is in training. Mamba's 2022 re-run will test
   whether selectivity helps there.
5. **Pending:** Mamba 2022 re-run (GPU); XGBoost `wide` robustness (the tree ceiling); model confidence set.
