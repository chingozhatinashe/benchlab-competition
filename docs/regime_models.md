# Regime models (feasibility study, not built)

**What it is.** One model per solar-wind regime, with a classifier that predicts the regime at t0
and routes the forecast:
- study: `src/training/regime_models.py`;
- run labels: `training.solar_wind_events.label_runs` (in-situ event detection);
- analysis: `src/experiments/regime_models.ipynb`.

**Result:** not built. Routing by the true regime (the oracle) would gain 22.6% over the shipped
hybrid. The best realistic method gains 1.0% in 3 of 6 folds, short of the bar set in advance (more
than 2% in at least 4 folds).

## Setup

- **Regimes:** 5, labelled from what happens during each run's 72 target hours:

| Regime | Label |
|---|---|
| CME | any target hour is CME-affected (ICME, sheath, shock flank), or in a weak-field cold interval ("possible CME") |
| stream arrival | otherwise, a high-speed stream arrives during the target hours |
| fast stream | otherwise, mean observed speed ≥ 550 km/s |
| slow wind | otherwise, maximum observed speed < 450 km/s |
| intermediate | everything else |

- **Folds:** the six yearly walk-forward folds (2019-10 → 2025-10); each fold fits on everything before its start. The eval year is not used.
- **Score:** per-run mean TIMSE relative to persistence; the summary is the geometric mean over folds.
- **Global models (the reference):** the shipped linear model and hybrid, refitted per fold. They reproduce the selection scores (0.4387, 0.4258).
- **Per-regime models:** the same two configs fitted on one regime's fit runs only. Only fit runs whose target hours end before the fold start get a label, so no label looks past the start.
- **Classifiers:** P(regime) at t0 from the hybrid's 30 regressors plus the speed terms (last speed; the mean and maximum of each recurrence term over the target hours). Settings are fixed, not tuned:
  - multinomial logistic regression;
  - XGBoost `multi:softprob`, depth 3, 200 rounds.

## Routing methods and results

| Method | Forecast | Fold score | vs hybrid | Folds won |
|---|---|---|---|---|
| oracle | the true regime's model (the ceiling) | 0.330 | −22.6% | 6 |
| shrunk soft (XGBoost) | 0.3 × soft + 0.7 × global hybrid; weight chosen on the folds | 0.4215 | −1.0% | 3 |
| global hybrid | | 0.4258 | | |
| global linear | | 0.4387 | +3.0% | 0 |
| soft (XGBoost) | Σ P(regime) × that regime's model | 0.444 | +4.2% | 1 |
| hard (XGBoost) | the most probable regime's model | 0.535 | +25.6% | 0 |
| prior | Σ climatological share × regime model | 0.537 | +26.1% | 0 |

The linear-member versions behave the same way and score slightly worse.

## Why it fails

**1. Most of the oracle's gain is the label leaking the future.** "Slow wind" means the speed stays
below 450 km/s for the next 72 h, so a model fitted only on such runs learns that no stream is
coming. Pooled TIMSE of each regime's hybrid against the global hybrid, by true regime:

| True regime | Its own model | Share of the hybrid's error |
|---|---|---|
| slow wind | 0.22 | 18% |
| fast stream | 0.60 | 4% |
| stream arrival | 0.79 | 25% |
| intermediate | 0.85 | 20% |
| CME | 1.06 | 34% |

The CME model doesn't beat the global hybrid even on true CME runs, though CMEs carry the most
error.

**2. The regime can't be predicted at t0.**

| | Logistic | XGBoost |
|---|---|---|
| Accuracy (5 regimes) | 48% | 50% |
| Log-loss skill vs climatology | 0.12 | 0.20 |
| CME recall / precision | 38% / 34% | 36% / 36% |
| CME runs called quiet (slow or intermediate) | 42% | 50% |
| Stream-arrival runs called quiet | 40% | 46% |

Around the start of a CME episode, the median P(CME) stays at 0.23–0.28 (climatology 0.14) and
rises to 0.44 only about 72 h later, when the CME reaches L1. The classifier sees CMEs in progress,
not coming.

**3. The wrong model is expensive.** The slow-wind model on a true stream arrival costs 3.9× the
hybrid; on a fast stream, 4.5×; on a CME, 2.7×. Soft routing improves slow wind (0.66× the hybrid)
but loses on stream arrivals (1.29×) and CMEs (1.12×).

## Decision and what it points to

- **Not built.** No runtime change.
- **The useful signal:** on quiet wind, a "no stream coming" probability lowers the error. That is the 72 h quiet-wind over-forecast found in every eval analysis, and inputs that predict the absence of a stream are a more direct route to it.
- **Follow-up:** the HMM study (`docs/regime_hmm.md`) asked the causal question instead, "which regime are we in at t0?". It found no gain either.

```bash
XGB_NTHREAD=2 python -m training.regime_models --pilot --folds 2024   # one fold, timed
XGB_NTHREAD=2 python -m training.regime_models                        # all six folds (~3 min)
python -m training.regime_models --summary
```
