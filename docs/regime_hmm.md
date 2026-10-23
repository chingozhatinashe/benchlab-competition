# HMM regimes (not built)

**What it is.** A hidden Markov model of solar-wind regimes whose filtered state probabilities are
given to the existing models as extra inputs:
- model: `src/models/regime_hmm.py` (numpy; experimental, not called by the live forecast);
- fold study: `src/training/regime_hmm.py`;
- analysis: `src/experiments/regime_hmm.ipynb`.

**Result:** no gain. On the six yearly folds, every variant scores within noise of the shipped
hybrid (best +0.1%). The HMM's information is already in the hybrid's inputs. It is not used, and
step 2 of the study (regressor-driven transitions) was not pursued.

## Motivation

The regime-routing study (`docs/evaluation.md`, `src/experiments/regime_models.ipynb`) labelled
regimes from each run's future 72 hours, so a router had to predict them, and it couldn't. An HMM
answers a different, causal question: which regime is the solar wind in at t0? It learns the states
and the transition probabilities from the data instead of using hand-set thresholds.

The plan had three steps; each would run only if the previous one showed a gain:
1. HMM state features for the existing models (this document);
2. transitions driven by the regressors (coronal-hole area, the recurrence terms);
3. explicit state durations (a hidden semi-Markov model).

## Model

- **Hidden states:** 4, following a first-order Markov chain with a fixed transition matrix. EM starts from threshold labels (slow < 400 km/s, intermediate, fast ≥ 550 km/s, ejecta = cold with Bt ≥ 8 nT).
- **Observations each hour,** independent given the state, from the in-situ hourly table (`training.pull_features`, already shifted by the in-situ latency):

| Observation | Emission in state s |
|---|---|
| speed (ar4) | v_t ~ N(c_s + φ_s v_prev, σ_s²), a Markov-switching AR(1); v_prev is the last speed within 6 h, else N(m_s, sd_s²) |
| speed (level4) | v_t ~ N(m_s, sd_s²) |
| log temperature ratio | N(μ_s, τ_s²) |
| log10 Bt | N(ν_s, ω_s²) |

- **Missing values** drop their term from the likelihood.
- **Fitting:** EM (Baum-Welch), scaled forward-backward, transition counts smoothed with one pseudo-count. A fit on 8 years of hours takes 15–30 s.
- **Features at t0,** from the forward filter at the row labelled t0 − 1h, so only data available at t0 is used:
  - `hmm_p_<state>`: the filtered probability of each state;
  - `hmm_state_hours`: hours since the most probable state last changed, capped at 240.
- **Live:** filtering on the 90 days of history fetched live gives the same features as filtering from 2016, to 3 decimals on 200 test t0s.

## Evaluation

**Folds:** the six yearly walk-forward folds (2019-10 → 2025-10). Per fold, the HMM is fitted on the
in-situ hours before the fold start, and the shipped linear model and hybrid are refitted with and
without the HMM features. The score is per-run mean TIMSE relative to persistence, as a geometric
mean over folds. The eval year was not used.

| Model | Fold score | vs hybrid | Folds won vs hybrid |
|---|---|---|---|
| hybrid (shipped config, the reference) | 0.4258 | | |
| both + ar4 (HMM features in the linear anchor and the trees) | 0.4263 | +0.1% | 2 |
| hybrid + level4 | 0.4265 | +0.2% | 3 |
| both + level4 | 0.4272 | +0.3% | 2 |
| hybrid + ar4 | 0.4275 | +0.4% | 3 |
| linear + ar4 | 0.4372 | −0.3% vs linear | |
| linear (shipped config) | 0.4387 | | |
| linear + level4 | 0.4398 | +0.3% vs linear | |

## Why it doesn't help

- **The information is redundant.** Gradient-boosted trees on the hybrid's own inputs (last speed, `bt_mean_1h`, `log_t_ratio_mean_1h`, `speed_mean_24h`, …) predict the filtered state probabilities with R² 0.86–0.93 for level4 and 0.71–0.83 for ar4 on the 2024 fold. The filter is a smoothed function of recent in-situ data the trees already see.
- **The one feature the inputs can't reproduce,** hours in the current state (R² ≈ 0.03), doesn't help either.
- **No CME state.** CMEs are about 10% of hours and too varied for unsupervised EM to give them their own state:
  - ar4's cold state is flagged as ejecta-like in only 17% of its hours;
  - level4's "ejecta" state is another intermediate state.
- **The states:**
  - level4 is stable across folds: slow about 335 km/s, mean stay about 2.5 days; fast about 555 km/s, about 2 days; two intermediate states.
  - ar4's states separate mainly by volatility, last about half a day, and swap meaning between folds.

## Decision

- **The HMM is not built.** No feature or model change ships.
- **Step 2 was not pursued,** because it would use inputs the hybrid already has.
- **The code is kept** for reuse; it is numpy-only and could run live.
- **Together with the regime-routing study,** regime structure doesn't look like the missing ingredient. The remaining errors are onsets (stream arrivals, CMEs) that the inputs at t0 don't foresee, and the quiet-wind level at 72 h.

**Not tried:** more than 4 states, emissions with density or field variability, and fitting by
TIMSE instead of likelihood.

```bash
XGB_NTHREAD=2 python -m training.regime_hmm --pilot --folds 2024   # one fold, timed
XGB_NTHREAD=2 python -m training.regime_hmm --max-seconds 240      # all six folds, resumable
python -m training.regime_hmm --summary
```
