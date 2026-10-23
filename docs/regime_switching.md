# Regime switching: HMM-gated experts (not built)

**What it is.** A standalone regime-switching forecaster. Each hidden-Markov regime gets its own
forecasting model, and the HMM's causal filter at t0 weights them. It doesn't use the linear model
or the hybrid.
- model: `src/models/regime_switching.py` (numpy; experimental, not used live);
- fold study: `src/training/regime_switching.py`;
- analysis: `src/experiments/regime_switching.ipynb`.

**Result:** not built.
- **Speed-only experts:** a 4-state gate improves them by 2.1% over a single regression, in all 6 folds, reaching the final baseline's level.
- **With the regressors:** every multi-state version is worse than the single regression.

## Model

    forecast(k) = Σ_s g_s × expert_s(k)        soft; hard uses the most probable state's expert

- **Gate g:** P(state s at t0 | in-situ data up to t0), the forward filter of a K-state HMM (`models.regime_hmm`) at the in-situ row labelled t0 − 1h.
  - Emissions: level-only, on speed, log temperature ratio and log Bt.
  - Missing in-situ data gives the stationary mix.
  - It is causal, unlike the future-window labels of the routing study (`docs/regime_models.md`).
- **Experts:** one per state, a per-lead weighted ridge regression of v(t0+k) − last speed.
  - Inputs: the last speed and the 27-day and 13.5-day recurrence means at t0+k ("speed"), optionally plus the 4 regressors the linear model selected ("regs"). These are inputs only; the linear model isn't used.
  - Training: expert s is fitted on every training run with weight TIMSE weight × g_s(t0). Ridge 0.1 on the slopes. Targets and recurrence terms are despiked.
- **Control, K = 1:** one state, gate = 1, a single regression with the same inputs. The gain of K > 1 over K = 1 measures regime switching itself.
- **HMM alone:** the 4-state Markov-switching AR HMM's own forecast. State probabilities propagate through the transition matrix, with each state's AR mean collapsed over the previous state (`RegimeHMM.mean_forecast`).

## Evaluation

- **Folds:** the six yearly walk-forward folds (2019-10 → 2025-10). The HMM and the experts are fitted on the data before each fold start.
- **Score:** per-run mean TIMSE relative to persistence, as a geometric mean over folds.
- **Context rows:** the final baseline, linear and hybrid scores from earlier runs on the same folds.
- **Decision rule, set before running:** the best K > 1 must beat K = 1 with the same inputs by more than 2% and win at least 4 of 6 folds.

| | Fold score | vs K = 1, same inputs | Folds won vs K = 1 |
|---|---|---|---|
| hybrid (context) | 0.4258 | | |
| K = 1, speed + regressors | 0.4375 | | |
| linear (context) | 0.4387 | | |
| K = 2 / 3, speed + regressors | 0.4396 / 0.4421 | +0.5% / +1.1% | 1 / 1 |
| K = 4 / 5, speed + regressors | 0.4474 / 0.4489 | +2.3% / +2.6% | 0 / 1 |
| K = 4, speed only | 0.4836 | −2.1% | 6 |
| final baseline (context) | 0.4843 | | |
| K = 1, speed only | 0.4941 | | |
| K = 2 / 3 / 5, speed only | 0.4951 / 0.4940 / 0.4978 | +0.2% / 0.0% / +0.8% | 4 / 3 / 3 |
| HMM alone | 0.7910 | | |

- **Hard vs soft:** hard gating is 0.1–0.5% worse than soft throughout.
- **Sanity check:** K = 1 speed-only (0.4941) matches the linear model's speed-terms-only selection step (0.4949).

## Why

- **The 4-state gate's useful state is compressed plasma.** Its "intermediate" state (398 km/s, 8.1 nT, high temperature ratio) marks the interaction region ahead of a fast stream. On the 2024 fold, gating on it cut the speed-only expert's error on runs gated "intermediate" to 0.92× K = 1. By true regime, the gain was on stream arrivals (0.92×) and intermediate wind (0.93×), at a cost on fast streams (1.10×) and slow wind (1.15×). The other state counts (2, 3, 5) start from speed quantiles, find no such state, and don't help.
- **The regressors already carry the same information.** P(intermediate) correlates 0.42 with `bt_mean_1h` and 0.38 with `density_mean_24h`, and P(fast) correlates 0.79 with the last speed. With those inputs in every expert, the split only divides the training data, so the error grows with K.
- **The HMM alone barely beats persistence.** It has no recurrence term, so it can only relax toward each state's level.

## Decision

- **Not built.** No runtime change.
- **The same lesson as the earlier studies:**
  - [regime routing](regime_models.md): future regimes can't be predicted at t0;
  - [HMM features](regime_hmm.md): present-state regimes are redundant with the inputs;
  - here: the same holds for a standalone model.
- **Side result:** the K = 1 expert with the linear model's 4 regressors (0.4375) is level with the shipped linear model (0.4387), so a simpler per-lead ridge does as well as the GLM.

**Not tried:** nonlinear experts, other HMM emissions, and physically initialised states for K ≠ 4.

```bash
python -m training.regime_switching --pilot --folds 2024   # one fold, timed (~1.5 min)
python -m training.regime_switching --max-seconds 240      # all six folds (~6 min), resumable
python -m training.regime_switching --summary
```
