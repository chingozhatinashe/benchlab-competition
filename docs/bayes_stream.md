# Bayes stream: a Bayesian generative model of slow wind and stream arrivals (in progress)

**The question (user, 2026-11-09).** A more principled follow-up to the [structural model](structural_model.md) and the [stream events](stream_events.md) model: a proper Bayesian model with two parts.
- **Slow wind:** a component from which a 72 h speed path can be sampled.
- **Streams, hierarchical:** draw the number of arrivals, then their times (stick-breaking), then build the 72 h wind from them.

**The user's choices:**
- NumPyro / NUTS for inference;
- smooth max of the slow wind and the streams, with additive streams as a variant;
- arrivals both observed in training (the detected arrivals are data) and fully latent, compared.

**Status (2026-11-15): built and tested, not yet scored.**
- **NUTS doesn't yet mix on the real training data** (2024 fold), so there are no scores.
- The 2024 pilot was stopped at its first fit.
- Sampler runs since: the arrival block mixes on its own. The speed block mixes far better with every event's rise width at its population value (no tree-depth hits); a full-length run is next.
- See [Sampler status](#sampler-status).

- **Model:** `src/models/bayes_stream.py` (numpy simulator, point decisions, asset; experimental, not used live).
- **Study:** `python -m training.bayes_stream` (`--pilot --folds 2024`; `--summary`).
- **Tests:** `tests/test_bayes_stream.py`.
- **Analysis (planned):** `src/experiments/bayes_stream.ipynb`; results in `src/experiments/bayes_stream.csv` and `bayes_stream_params.csv`.
- **Dev requirements:** jax 0.6.2 and numpyro 0.19.0 (CPU), installed into the venv only. They are not in `requirements.txt` or `requirements-dev.txt`, and the tests skip without them. The NumPyro model is in `training`; `models` stays numpy only.

## Why a generative model

- **v1 and v2 fitted one speed loss.** Arrival rate, timing and event size were confounded:
  - v2 expected about 5× the detected arrivals;
  - the expected value over uncertain arrivals smoothed the stream profile away.
- **Here each level has its own likelihood and priors:**
  - the count of detected arrivals (Poisson);
  - their times (stick-breaking);
  - each event's marks, latent with population priors;
  - the speeds given all of that.
- **Parameter uncertainty is a posterior.** The forecast is a sample of 72 h paths.
- **One path is scored.** The deterministic forecast is a decision on those samples: either the predictive mean, or the Bayes action under the official-weighted loss.

## Model

For a run at t0, leads k = 1…72; x(u) are the structural model's lead-aligned standardised inputs at hour t0 + u (`models.structural.StructuralModel.prepare`: the submitted config's regressors, recurrence terms, last speed, missing flag).

**1. Slow wind**
- **Ambient level:** a = 250 + 250·sigmoid(w_a · [1, q20, rec27]), as in v1 (250–500 km/s).
- **Mean path:** S(k) = a + ρ^k (v0 − a), with ρ_up above the ambient and ρ_dn below, so the current excess relaxes at its own rate in each direction.

**2. Streams**

*Arrival times* run over hours u = 1…84, 12 h past the window, because a rise centred just after t0 + 72 starts inside it.

| Level | Form |
|---|---|
| Count | N ~ Poisson(Λ), truncated at N ≤ 2 (N ≤ 1 for the latent fit); log Λ = w_N · x̄, with x̄ the window-mean inputs |
| Times given N | stick-breaking over the hourly grid: v_u = sigmoid(w_τ · x(u)), π_u ∝ v_u ∏_{s<u} (1 − v_s), normalised. With N = 2: the first time u ≤ 84 − 48, the second u ≥ τ₁ + 48 (gap 48 h) |
| Marks per event | peak excess E ~ LogNormal(μ_E + w_E · x(τ), σ_E); rise width R ~ LogNormal(μ_R, σ_R); decay D ~ LogNormal(μ_D + w_D · x(τ), σ_D) |

- **Rate and timing are separate:** given N, the stick shapes only the timing; the rate is Λ.
- **Stream profile:** s_j(k) = a + E_j · logistic((k − τ_j)/R_j) · exp(−max(0, k − τ_j − 2R_j)/D_j).
  - τ_j is the midpoint of the rise, which is where the detector labels an arrival.

**3. Wind given the arrivals**
- **Main rule:** V(k) = smoothmax(S(k), s_1(k), s_2(k)), a log-sum-exp at temperature 5 km/s.
- **Variant:** S(k) + Σ_j (s_j(k) − a).
- **Noise:** then V(k) + η(k), with η an AR(1) process (ρ_η, σ_η).
  - The noise is added after the streams, so the speed likelihood is exact.
  - Missing hours are skipped exactly: the step is ρ^g and the variance σ²(1 − ρ^{2g})/(1 − ρ²) across a gap of g hours.

**Priors** (standardised inputs):

| Parameter | Prior |
|---|---|
| w_a | N(0, 1.5) |
| ρ_up, ρ_dn | Beta(19, 1) |
| ρ_η | Beta(8, 2) |
| σ_η | HalfNormal(50) |
| w_N | N(log 0.25, 1) intercept, N(0, 0.5) slopes |
| w_τ | N(−3.7, 1.5) intercept, N(0, 0.5) slopes |
| μ_E, μ_R, μ_D | N(log 250, 0.5), N(log 6, 0.5), N(log 48, 0.5) |
| w_E, w_D | N(0, 0.3) |
| σ_E, σ_R, σ_D | HalfNormal(0.5) |

## Training data and likelihood

**Training windows**
- Fit runs every 72 h, so windows don't overlap: each hour counts once, and the likelihood is exact given each window's start.
- Dropped:
  - windows with CME or possible-CME hours in t0…t0 + 84 h (CMEs aren't modelled; the forecast still covers every run);
  - windows with fewer than 36 observed target hours (`MIN_OBSERVED`).
- The fit ends 48 h before the fold start (`LABEL_GUARD`), because the arrival detector looks up to 48 h ahead.
- **2024 fold:** 761 windows and 192 detected arrivals.
  - Before the exact gap likelihood, only 269 of 974 windows were usable, because 65% had a missing hour.

**Observed arrivals (main fit)**
- The detected arrivals in hours 1…84 (`training.stream_events.labels`) are data.
- The count and the times have closed-form likelihoods.
- Each event's log E, log R and log D are latent, around the population means (partial pooling).
- About 190 × 3 event parameters plus 30 globals.

**Latent arrivals (variant)**
- N ≤ 1 per window: the detector can't label two arrivals within 3 days anyway.
- The arrival is summed out by enumeration: none, or one in each 3-hour bin (28 bins at their middle hours, `LATENT_STEP`), with a log-sum-exp.
- The marks are at their population values: without a label there are no event effects.

**Inference**
- NumPyro NUTS, 2 chains × (500 warm-up + 500 draws), in parallel (`numpyro.set_host_device_count`).
- Runs in jax's default float32.
- Gradient time on an idle machine: about 6 ms for the observed fit and 54 ms for the latent one.
  - The latent one was 630 ms before float32, the log-sum-exp combine and the 3-hour bins.
- Diagnostics: R-hat, bulk ESS, divergences and mean leapfrog steps, for the globals and the event sites.

## Forecast and decisions

`simulate_paths` (numpy) draws, for each run and each posterior draw:
1. N, then the times by stick-breaking;
2. the marks;
3. the AR(1) noise;
4. the combination (max or additive).

Runs are processed in chunks to bound memory, and 400 paths are drawn per run.

**Point forecasts**
- **Mean:** the posterior predictive mean, which minimises the squared error.
- **Official:** per lead, f = Σ w(y)·y / Σ w(y) over the paths. This is the Bayes action for the weighted squared loss.
  - w comes from the official percentile weights on the fit targets, with blend 0.25 as the linear model uses (`models.final_baseline.official_weights`).
  - It is interpolated on a 200–1200 km/s grid.

**Also kept:**
- quantiles 10/50/90;
- P(N ≥ 1) and the expected arrivals;
- CRPS and the 80% coverage against the observations.

## Planned study

Each variant is refitted per fold on the runs before the fold, with PFSS gaps at the fit median, as in v1 and v2.

| Variant | Fit | Decision |
|---|---|---|
| bayes | observed arrivals, smooth max | mean |
| bayes official | same posterior | official Bayes action |
| bayes additive | observed arrivals, additive | mean |
| bayes latent | latent arrivals, smooth max | mean |
| slow wind only | bayes' posterior with N = 0 | mean |
| linear + bayes | the submitted linear model + bayes' mean as a lead-aligned regressor | — |
| reference | the submitted linear model (2024 fold: S_final 107.81) | — |

**Also reported per fold:**
- arrival skill on the validation runs (AUC of P(N ≥ 1), expected against observed arrivals);
- CRPS and coverage;
- posterior summaries;
- NUTS diagnostics;
- posterior predictive checks on the training windows.

**Decision rule, set before running:** more than 1% better than the reference and winning at least 4 of 6 folds. The eval year is not used.

## Checks

Tests in `tests/test_bayes_stream.py`:

| Test | What it checks |
|---|---|
| `test_stick_breaking_probabilities` | the stick sums to 1 and matches a direct categorical |
| `test_ar1_likelihood_with_gaps_matches_the_marginal_gaussian` | the gap AR(1) likelihood equals scipy's multivariate normal |
| `test_count_probabilities` | the truncated Poisson count |
| `test_max_rule_dominates_components_and_additive_adds` | the combination rules |
| `test_no_arrivals_gives_the_slow_wind` | with no arrivals, the path is the slow wind; two arrivals are at least the gap apart |
| `test_numpy_jax_parity` | the numpy and jax paths agree |
| `test_latent_enumeration_matches_brute_force` | the latent enumeration matches a brute-force sum (rtol 1e-6) |
| `test_recovery_with_observed_arrivals` | known parameters are recovered from 500 synthetic windows |
| `test_asset_round_trip` | the asset saves and loads unchanged |
| `test_split_blocks_add_up_to_the_joint_density` | the arrival and speed blocks' log densities add up to the joint one; the latent fit refuses to split |

**Status of the tests**
- All 9 original tests passed with the non-centred event marks.
- After the switch to centred marks, the recovery test fails: in 11.6 min (1 chain, 300 + 300), the count slope came out at 0.01 against a true value of 0.6. The cause hasn't been checked yet.
- Two earlier fixes came from this test:
  - the synthetic count input is now constant within a window, because a slope on an hourly-varying input isn't identified from a window count;
  - the noise is now added after the streams. Before, σ_η came out at 9.95 against 12.

## Sampler status

**2024 pilot, observed fit**
- Default settings: init to median, diagonal mass, target accept 0.9, tree depth 8.
- Result: R-hat 34, ESS 1, 188 divergences, 1431 s.
- **The two chains were in different modes:**

  | | Chain 1 | Chain 2 |
  |---|---|---|
  | μ_D | 3.8 | 4.5 |
  | σ_D | 0.014 | 1.28 |
  | w_a | differs | differs |

- The pilot was stopped there, so no variant was scored.

**Experiment 1: MAP start, non-centred marks**
- Setup:
  - the posterior mode by SVI (AutoDelta, Adam 0.01, 4000 steps, 12 s);
  - then NUTS from that point, in float64, with a dense mass matrix on the globals, 300 + 300.
- **The mode:**

  | σ_E | σ_R | σ_D | ρ_up | ρ_η | σ_η |
  |---|---|---|---|---|---|
  | 1.46 | 2.38 | 2.41 | 0.989 | 0.984 | 10.9 km/s |

  So the rise and decay vary by about an order of magnitude between events.
- **The NUTS run:** 0 divergences, but every step hit the tree-depth limit (255 leapfrog steps), with R-hat 1.4–6 and ESS 1–2.
- **Reading:** each event's marks are pinned by its own speeds, and in that case a non-centred parameterisation mixes poorly. Hence the switch to centred marks.

**Experiment 2: centred marks, dense mass on all parameters, MAP start, 500 + 500**
- Stopped by the system for low memory before it printed anything.
- The laptop has 7.7 GB, with about 0.6–0.8 GB free on 2026-11-15 (browsers, editors). A dense mass matrix over every event mark grows with the square of the parameter count.

**Sampler runs, 2026-11-15** (`python -m training.bayes_stream --sampler LABEL`: one fit on a fold's windows, no scoring; results in `src/data/bayes_stream_cache/sampler_LABEL.json`)
- **Lean setup:** float64, the dense mass matrix on the globals only (diagonal on the marks), the chains one after another. Peak memory 1.0–1.1 GB.
- 2024 fold: 761 training windows, 192 events.

| Run | Setup | Time | Leapfrog | R-hat max (globals / marks) | ESS min |
|---|---|---|---|---|---|
| smoke | centred MAP start, 30 + 30 | 330 s | 255 (all) | 21 / 22 | 1 |
| smoke2 | non-centred MAP start, 30 + 30 | 327 s | 255 (all) | 3.2 / 18 | 1 |
| medium | non-centred MAP start, 200 + 100 | 1409 s | 255 (all) | 6.2 / 8.1 | 1 |
| arrivals | arrival block only, 200 + 100 | 71 s | 14 (none) | 1.01 / — | 112 |
| speed | speed block only, 200 + 100 | 914 s | 255 (all) | 3.8 / 6.2 | 1.1 |
| speed_ncE | speed block, non-centred `log_E` (`--noncentred log_E`), 200 + 100 | 936 s | 255 (all) | 4.4 / 9.0 | 1.1 |
| speed_hess | speed block, mass matrix started from the Hessian (`--mass-init hessian`), 200 + 100 | 912 s | 255 (all) | 1.9 / 3.7 | 1.6 |
| speed_hess_ncE | speed block, Hessian start and non-centred `log_E`, 200 + 100 | 898 s | 255 (all) | 2.3 / 5.7 | 1.2 |
| fixed_rise_hess | speed block, **rise width at its population value** (`--fit fixed_rise`), Hessian start, 200 + 100 | 270 s | 63 (none) | 1.4 / 2.3 | 4.2 |

All runs: 2 chains, 0 divergences.

- **The MAP start.** Found directly in the centred form, the mode collapses the event spreads (σ_E 0.003, σ_D 0.002). The centred joint density grows without bound as a spread goes to 0 with every mark at its mean, so NUTS started in the funnel's neck. `map_point` now finds the mode in the non-centred form (`LocScaleReparam`) and maps it back to centred marks (σ_E 1.46, σ_R 2.38, σ_D 2.41, as in experiment 1). That removed the split into two modes (R-hat on the globals 21 → 3.2 at 30 + 30) but not the saturated trees.
- **The posterior splits exactly into two independent blocks.** With observed arrivals, `w_N` and `w_tau` enter only the count and time terms, never the speed term. `numpyro_model(block=...)` fits either block alone, `fit_split` joins their draws, and `test_split_blocks_add_up_to_the_joint_density` checks the split. The latent fit does not split.
- **The arrival block mixes on its own.** In the joint fit, `w_N` and `w_tau` were the worst parameters (R-hat 6.2, 3.2) only because the speed block forced tiny steps on everything.
- **The speed block is the problem:**
  - Worst: `w_D` (R-hat 3.8; chain medians −0.24 and −0.63), `w_E` (2.2), `rho_dn` (1.8), `sigma_E` (1.6). Fine: `sigma_eta`, `rho_eta`, `mu_R` (R-hat 0.99–1.06).
  - σ_E falls from 1.45 at the start to about 0.17, so the data barely move each event's peak excess off its population value. That is where centred marks mix worst.
  - σ_D is about 1.0, and the median decay about 60 h: many decays run past the end of the window, so the decay regression is weakly identified.

- **Mixed parameterisation doesn't help.** Sampling `log_E` non-centred (the rise and decay marks centred) leaves every step at the tree-depth limit; the worst site moves to `w_E` (R-hat 4.4), then `w_D` (2.6). So the excess marks' parameterisation is not what sets the step size.

**Curvature at the MAP start** (`--curvature`: the Hessian of NUTS's potential in its unconstrained space, 607 dimensions for the speed block, 10 s; `sampler_curvature_speed.json`)

| | Largest eigenvalue | Smallest positive | Condition | Negative |
|---|---|---|---|---|
| raw | 9.7e4 | 0.096 | 1.0e6 | 4 |
| scaled by the diagonal | 3.1 | 0.0077 | 408 | 4 |

- **Raw: the well-determined globals set the step size; the event decays are the flattest.**
  - Stiffest: `sigma_eta` (curvature 9.7e4: about 50,000 observed hours pin the noise level to a posterior sd of about 0.003 in log space), then `w_a` (the ambient wind, up to 3200) and `rho_up` (1200), then single events' rise widths `log_R` (800–1100).
  - Flattest: single events' decays `log_D` (about 0.1). The decay regression `w_D` is the worst-mixing global in every run.
  - NUTS's step has to resolve the stiffest direction, so crossing the flattest takes about √(1e6) = 1000 steps: hence every iteration at 255.
- **An adapted mass matrix would remove most of that** (condition 408 after scaling by the diagonal). But the adaptation estimates the variances from warmup draws, and the chains barely move along the flat directions, so it never learns them.
- **What a diagonal mass leaves is the model's:**
  - the flattest scaled directions load half on an event's peak excess `log_E` and half on its rise width `log_R`: within an event the two trade off, a ridge;
  - 4 directions of negative curvature at the start, in single events' `log_R` and `log_D` (with some `log_E`).
- (A first version of this check labelled the Hessian's rows in the model's site order instead of jax's sorted flattening order, and named the wrong sites. The numbers were the same; the labels above are the corrected ones.)

- **Starting the mass matrix from the Hessian helps, but not enough.** `--mass-init hessian` (1/|diagonal| at the MAP start) halves the worst R-hat (globals 3.8 → 1.9, marks 6.2 → 3.7; `w_D` 3.8 → 1.6), but every step still hits the tree-depth limit. Worst now: `rho_dn`, `w_a` (1.9), `sigma_D` (1.8).
- **The MAP start is not where the posterior mass is.** σ_E is 1.45 at the start and about 0.17 in the draws, so the curvature at the start describes a different region from the one the chains sample. With σ_E that small, the peak excesses sit in a centred funnel.

- **Both together are no better.** The Hessian start with non-centred `log_E` still hits the tree-depth limit on every step, with R-hat 2.3 / 5.7 (worst `w_E`, `w_D`, `sigma_D`). So the sampler-side fixes (start, mass matrix, parameterisation) are used up: what is left is the model's geometry at the event level (each event's peak excess against its rise width, weakly identified decays).

- **The rise width at its population value fixes the trees** (`Config(event_rise=False)`, fit `fixed_rise`: every event's rise width is exp(μ_R), with no per-event `log_R` and no σ_R).
  - No step hits the tree-depth limit (63 leapfrog steps on average, against 255), so the fit takes 270 s instead of 912 s.
  - R-hat 1.4 on the globals and 2.3 on the marks, ESS min 4: not converged at 100 draws, but far better than any earlier run. Worst: `rho_up` (1.4), `mu_R` (1.2), `sigma_E` (1.2); `w_D` is at 1.07.
  - The posterior moves: the rise width is exp(1.05) ≈ 2.9 h, and the decay longer, exp(μ_D) ≈ 150 h (σ_D 1.46), well past the end of the window. The decays stay weakly identified.

**Next steps**
- The full 500 + 500 with `fixed_rise` and the Hessian start (about 0.4 s per iteration: about 15 min), to see whether it converges (R-hat < 1.01, ESS in the hundreds).
- If the decays still limit the mixing: tighter priors on `w_D` and σ_D, or bounded decays.
- Then the study needs the fixed-rise fit as its observed fit (and the additive one), and the 2024 pilot.
- If the excess–rise ridges then limit the mixing: bounded rise widths, or a tighter prior on σ_R.
- If `w_D` stays stuck, simplify the decay: tighter priors on `w_D` and σ_D, or bounded decays. Then the rest of the list: tie ρ_up to ρ_η, tighter prior on σ_R, bounded rise widths.
- Check the recovery test failure under the centred marks.
- Then run the 2024 pilot (timed, with a six-fold estimate), with the split fit, and wait for the go-ahead before the six folds.
