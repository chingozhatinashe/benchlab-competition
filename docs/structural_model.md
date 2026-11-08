# Structural model: ambient wind + stream arrivals, fitted in one step (not built)

**The question (user, 2026-11-08).** Can a single model, trained in one step, impose a structure on how the solar wind is generated?
- (i) a steady ambient (slow) wind;
- (ii) stream arrivals, each with a timing, duration and speed.

**Answer: yes, it can be built.** `models.structural` writes both parts as one differentiable forecast equation, and all of its ~40 parameters are fitted together in a single optimisation on the official-weighted loss.

**But it doesn't beat the submitted linear model.**
- **On its own:** 4.5% worse (wins 2 of 6 folds).
- **Without the current-sheet distance:** 3.1% worse.
- **As an input to the linear model:** 0.3% better, but on only 3 of 6 folds. The rule needs more than 1% and at least 4 folds.

Nothing is built.

- **Model:** `src/models/structural.py` (numpy; experimental, not used live).
- **Study:** `python -m training.structural` (`--pilot --folds 2024`; about 20 min for six folds).
- **Analysis:** `src/experiments/structural_model.ipynb`; results in `src/experiments/structural_model.csv` and `structural_model_params.csv`.
- **Dev requirement:** PyTorch (CPU) for the fit; the forecast is numpy only.
  - The venv's `torch 2.3.1+cpu` wheel predates numpy 2, so arrays go through DLPack (`training.structural.to_torch`).
  - torch is not in `requirements-dev.txt`, and the torch tests skip without it.

## Model

At t0, for leads k = 1…72, where x(u) holds the inputs for hour t0 + u, lead-aligned as in the linear model:

    v(k) = a + e0 · exp(−k / τ_c) + Σ_{u ≤ k} P_first(u) · A(u) · g(k − u)

| Part | Form |
|---|---|
| Ambient wind a | 250 + 250 · sigmoid(w · [1, q20, rec27]): q20 is the 20th percentile of the speed over the last 27 days, rec27 the 27-day recurrence mean (both centred, ÷100 km/s) |
| Current stream | e0 = last speed − a, relaxing over τ_c = 1 + 24 · softplus(w · x(1)) h when above the ambient (one fitted τ below it) |
| Stream arrival (timing) | hazard h(u) = sigmoid(w · x(u)); only the first arrival counts: P_first(u) = h(u) ∏_{s<u} (1 − h(s)) |
| Its speed | peak excess A(u) = 100 · softplus(w · x(u)) km/s |
| Its duration | profile g(d) = (1 − e^{−(d+1)/r}) e^{−d/τ_s}, peak 1; rise time r and decay time τ_s fitted |

- **x(u):** a constant; the last speed; the 27-day and 13.5-day recurrence at t0 + u; the submitted model's five regressors, standardised, at lead u (field strength, coronal-hole strip, X-ray, density, current-sheet distance); a missing flag for the coronal-hole strip.
- **Input preparation:** lead filling, imputation, the 1% cap, standardisation, despiking and the recurrence terms are all reused from `models.linear_model`, so both models see identical inputs.
- **Output:** the expected speed, clipped to the valid range.

**Training (one step):**
- every parameter fitted together, minimising the linear model's loss: official weights, blend 0.25, despiked targets, zero weight without a last speed;
- PyTorch L-BFGS in float64, from physical starting values, two restarts (hazard 0.25% and 1.1% per hour), keeping the lowest loss;
- ridge 1e-3 on all weights except the intercepts.

**Checks** (`tests/test_structural.py`):
- the numpy and torch forecasts agree to 1e-6;
- the fit recovers known parameters from synthetic data (error under the 2 km/s noise);
- the asset round trip;
- the stream term is never negative, and the current excess only relaxes.

## Evaluation

- **Folds:** the six yearly folds (`training.folds.YEARLY`). Each model is refitted on the runs before the fold, and PFSS gaps take the fit runs' median.
- **Score:** official S_final relative to persistence, as a geometric mean over folds, and per run regime (`training.model_diagnostics`).
- **Reference:** the submitted linear model, refitted per fold (0.7794).
- **Variants:**
  - the full model;
  - **ablations:** without new arrivals; without the current-sheet distance; one τ_c for all runs;
  - **linear + structural:** the structural forecast as a lead-aligned regressor (`structural_speed_lead_0h`) in the submitted linear model. Its fit runs use the structural model's in-sample forecasts.
- **Decision rule, set before running:** more than 1% better on at least 4 of 6 folds. The eval year is not used.

## Results

| Variant | / persistence | vs submitted | Folds won | Stream arrival | Fast stream | Intermediate | Slow | CME |
|---|---|---|---|---|---|---|---|---|
| linear + structural | 0.7771 | **−0.30%** | 3 | +0.1% | −3.6% | +0.9% | −1.2% | +0.1% |
| submitted (reference) | 0.7794 | | | | | | | |
| structural, no current-sheet distance | 0.8036 | +3.1% | 2 | +0.7% | −1.4% | −2.6% | +1.7% | +5.5% |
| structural | 0.8144 | +4.5% | 2 | +2.2% | +7.1% | +6.6% | +6.9% | +4.2% |
| structural, one decay time | 0.8220 | +5.5% | 2 | +4.4% | +7.8% | +7.5% | +11.1% | +4.5% |
| ambient + decay only | 0.8764 | +12.5% | 0 | +13.6% | +16.9% | −1.8% | −1.1% | +15.1% |

**By fold** (% change against the submitted model):

| Fold | Structural | No current-sheet distance | Linear + structural |
|---|---|---|---|
| 2019 | −0.1 | −0.9 | −3.2 |
| 2020 | −2.7 | +2.2 | −2.6 |
| 2021 | +7.9 | +1.3 | +0.4 |
| 2022 | +10.4 | +9.6 | +1.5 |
| 2023 | +2.0 | +7.5 | −0.1 |
| 2024 | +10.2 | −0.6 | +2.4 |

**The pilot misled again.** On the 2024 fold, the version without the current-sheet distance was −0.6%; over six folds it is +3.1%.

## Why it doesn't help

- **The structure costs flexibility the data uses.** The linear model has a free coefficient for every input at every lead. The structural model forces one shape for every run: one rise, one decay profile, one hazard form. It wins in the quiet 2019–2020 folds, where that shape fits. It loses 8–10% in the rising and maximum folds (2021, 2022, 2024), where streams and CMEs vary more.
- **Timing, duration and speed are not identified separately.** The rise time hits its 1-hour floor in every fold and every variant, so arrivals behave as steps. Hazard and amplitude trade off against each other:
  - early folds: about 1% per hour × about 280 km/s;
  - later folds: about 0.4% per hour × 550–790 km/s.

  The expected value only pins down their product. The fitted parameters therefore change from fold to fold, and big, rare predicted arrivals cost a lot when they don't come.
- **The expected value smooths the structure away.** An unknown arrival time, averaged over the hazard, becomes a gradual rise. That is close to what the linear model's per-lead coefficients already produce, so the structure adds little that isn't already there.
- **Slow wind isn't fixed either.**
  - The ambient term alone ("ambient + decay only") is 1% better on slow runs, but 14–17% worse on streams.
  - With arrivals added, slow runs get worse (+7%), because the hazard never goes to zero.

  This is the same trade-off as in [slow_wind.md](slow_wind.md) and [pfss_switch.md](pfss_switch.md).
- **As an input, it adds a little information,** −0.3% on 3 folds, below the 1% rule. It helps fast streams (−3.6%).

## What would change the picture

- **Not tested:** a probabilistic forecast, where the arrival structure could matter more. The competition scores a single deterministic path, and the expected value is optimal for its RMSE part.
- **A sharper arrival indicator.** The hazard's skill comes from the same inputs the linear model already uses ([pfss_switch.md](pfss_switch.md): in-situ AUC 0.86, no gain from PFSS).
