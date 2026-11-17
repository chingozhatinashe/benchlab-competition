# Slow-wind bias: does the metric reward the over-forecast? (step 5 of the next steps)

**The question (2026-11-17).** The submitted linear model over-forecasts runs that stay slow by about 40 km/s ([model_diagnostics.md](model_diagnostics.md)), and every attempt to remove it failed ([slow_wind.md](slow_wind.md), [post_calibration.md](post_calibration.md)). Is the over-forecast what the official metric's tail-weighted TIMSE rewards, or a cost the model pays for not telling slow runs apart? This is step 5 of [next_steps.md](next_steps.md).

**Result: the metric doesn't reward the over-forecast. The model can't detect slow runs.**
- **With an oracle,** shifting only the true slow-wind runs down is 1.8–2.5% better on all six folds, through RMSE, with the tail term unchanged.
- **With a realistic indicator,** last speed below 450 km/s, it gains nothing: only 41% of the runs it flags stay slow.
- **A global shift is worse:** +3% at 20 km/s, so the current overall lean is right.
- **The ceiling:** even perfect detection gains only about 2–2.5% overall.
- **Study:** `python -m training.slow_bias` (`--pilot --folds 2024`; `--summary`; about 1.5 min). Results: `src/experiments/slow_bias.csv`. Notebook: `src/experiments/slow_bias.ipynb`.

## Setup

Per fold, the submitted config is refitted on the fold's training runs. Its forecasts of those runs (in-sample) and of the validation runs are shifted down by δ km/s at every lead, in one of three sets of runs:

| set | runs | share of runs | precision for slow wind | recall |
|---|---|---|---|---|
| oracle | regime "slow wind" (from the truth: no arrival, no CME, staying slow; `training.model_diagnostics`) | 30% | 100% | 100% |
| last<450 | the last observed hourly speed below 450 km/s (known at t0) | 68% | 41% | 97% |
| all runs | every run (a control) | 100% | 30% | 100% |

δ is chosen in three ways:
- **On the training runs:** the δ (0–80 km/s, steps of 5) that minimises S_final there.
- **Per lead:** the training runs' mean over-forecast at each lead.
- **In hindsight:** the best δ on the validation runs themselves, an upper bound.

**Reading, fixed before running:**
- **The oracle fails** (its fitted shift not more than 1% better on at least 4 of 6 folds): the over-forecast is about what the metric rewards; stop chasing it.
- **The oracle passes but last<450 doesn't:** the problem is detecting slow runs.
- **last<450 passes too:** a candidate model change.

## Six folds

| shift, chosen on the training runs | S_final vs reference | folds won | RMSE | TRMSE | slow-wind runs |
|---|---|---|---|---|---|
| oracle, one δ (24 km/s on average) | **−1.79%** | **6** | −3.4% | −0.1% | −10.9% |
| oracle, per lead (30 km/s on average) | **−2.47%** | **6** | −4.8% | −0.1% | −14.7% |
| last<450, one δ | 0.00% (δ = 0 chosen on every fold) | 0 | | | |
| last<450, per lead (11 km/s) | +0.79% | 1 | −2.3% | +3.9% | −8.2% |
| all runs, one δ | 0.00% (δ = 0 on every fold) | 0 | | | |
| all runs, per lead (15 km/s) | +1.30% | 1 | −5.3% | +8.3% | −10.8% |

Hindsight grid (S_final vs reference, geometric mean over the six folds):

| δ km/s | oracle | last<450 | all runs |
|---|---|---|---|
| 10 | −1.1% | +0.6% | +0.7% |
| 20 | −1.7% | +2.3% | +3.0% |
| 30 | **−1.8%** | +4.9% | +6.6% |
| 40 | −1.5% | +8.4% | +11.5% |
| 60 | +0.4% | +17.4% | +24.4% |

**The slow runs' mean residual** (observed − forecast) grows across the folds: −19 (2019), −40, −48, −62, −42, −71 km/s (2024).

## What it means

1. **The over-forecast is a cost, not what the metric rewards.**
   - **RMSE:** shifting the true slow runs down improves it by up to 3.5%.
   - **TRMSE:** it barely moves, because the slow runs' observations sit low in each lead's distribution, where the tail weights are small.
   - **The best shift is 25–30 km/s,** about two thirds of the mean over-forecast.
2. **The global lean is right.** Shifting every run down loses on the tail term (+8% per lead) far more than it gains on RMSE. The retune's blend (0.25) already balances the two across all runs.
3. **Detecting slow runs is the problem.** "Slow wind" is defined by the next 72 hours: no arrival and no CME. A low last speed catches nearly all slow runs (97%), but 59% of the runs it flags have an arrival coming. Shifting those down costs more than shifting the slow ones gains. Earlier attempts failed for the same reason: state terms on the last speed ([slow_wind.md](slow_wind.md)), and the PFSS switch ([pfss_switch.md](pfss_switch.md)), whose oracle gained 3.3–3.9% on 2024.
4. **The ceiling is about 2–2.5%** overall, even with perfect detection: slow runs are 30% of runs, and they improve by 11–15%. A realistic detector gets part of that.

**For the next steps:**
- **Step 2, the gated linear / Bayes-official combination:** its gate is a realistic slow-run detector, P(no arrival) from the Bayes stream model, with AUC 0.69 for P(N ≥ 1). On the slow-wind side, expect at most part of the 2–2.5% ceiling. The gate's quality matters more than the shift's size.
- **Step 1b:** (b) and (c), with WSA and the Bayes features, already improve slow-wind runs (−1.5% to −1.7%) where (a) made them worse. That is consistent with better detection, not a different lean.
