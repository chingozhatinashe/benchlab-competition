# Next steps after the Bayes studies (2026-11-17)

**Context.** The model freeze is on 2026-11-19. The submitted model is the linear model (official weights, coronal-hole lead alignment, PFSS current-sheet distance). The Bayes model is built as an option behind `SWP_MODEL=bayes` ([bayes_model.md](bayes_model.md), PR #57). Its constrained latent fit (`--fit latent_constrained`) converged on 2026-11-17 (R-hat 1.028); on the eval year it is 0.15% better than the reference. This plan ranks what is left to try, including avenues that need more time than the freeze allows.

**Bottom line.**
- **Before the freeze:** the pre-registered near-miss combination (step 1) passed on the folds but not on the eval year, so the submitted model stays; (a) runs in staging. Building (b) and (c) is step 1b.
- **After the freeze:** the strongest bet is a gated combination of the linear model and the Bayes stream's official-decision path (step 2). It comes before the diffusion model (step 4).

## What the work so far shows

- **Adding inputs has stopped paying off.** Since the coronal-hole lead alignment (−2.7%, [lead_aligned_ch.md](lead_aligned_ch.md)) and the PFSS current-sheet distance ([pfss.md](pfss.md)), every addition has landed between −0.2% and −0.8%. That covers images, protons, WSA, interactions, HMM regimes, GP corrections and the Bayes features.
- **Slow wind is the largest remaining error** ([model_diagnostics.md](model_diagnostics.md)):
  - Slow-wind runs are 30% of runs and carry 60% of the squared error.
  - The linear model is 24% worse than persistence there and over-forecasts by 37 km/s.
  - State terms, post-hoc corrections and the loss blend all failed to fix it. The blend trades slow-wind accuracy against active-wind accuracy one for one ([slow_wind.md](slow_wind.md), [post_calibration.md](post_calibration.md)).
- **The Bayes stream model is strong where the linear model is weak.** On the 2024 pilot fold ([bayes_stream.md](bayes_stream.md)), its official-decision path was:
  - 25% better than the linear model in slow wind and 17% better in intermediate wind;
  - 9% worse at stream arrivals and 2% worse in fast streams.

  As a regressor it is applied the same way in every run, and this contrast is lost.
- **Knowing the regime would be worth a lot, but it is hard to know.** Routing by the true regime would gain 22.6%. Realistic routing (classifiers, HMMs) gained about 1% ([regime_models.md](regime_models.md), [regime_switching.md](regime_switching.md)).
- **The large CME misses have no usable input.** The May 2024 storm was under-forecast by about 300 km/s. The CME catalogues are published too late (LASCO CDAW) or unreachable (DONKI), and the image detectors can't tell which CMEs reach Earth ([cme_detector.md](cme_detector.md)).

## The steps

| # | Step | Effort | Odds of passing the 1% / 4-of-6 rule | Before the freeze? |
|---|---|---|---|---|
| 1 | Near-miss combination, pre-registered | **Done 2026-11-17:** (a) passes on the folds (−1.08%), +0.45% on the eval year; not submitted, in staging | — | — |
| 1b | Build (b) and (c): WSA and Bayes features in the live linear model | 1–2 days, plus staging | Already pass on the folds (−1.42%, −1.92%) | No |
| 2 | Gated linear / Bayes-official combination | 1–2 days, plus the fold fits | Moderate to good | No |
| 3 | Bayes official path as a regressor | ~½ day once the fold fits exist | Low to moderate | Only if the fold fits exist |
| 4 | Residual diffusion model, with a direct neural control | 2–4 days to a pilot | About a coin flip | No |
| 5 | Is the slow-wind bias optimal for the metric? | **Done 2026-11-17:** no; detection is the problem (oracle −1.8% to −2.5%, 6 of 6; realistic indicator 0%) | — | — |
| 6 | Archive live WSA-Enlil | ~1 h to set up | — (pays off after the competition) | Set-up only |

### 1. Near-miss combination (pre-registered)

Each near-miss is at least 0.3% better on at least 4 of 6 folds, but short of the 1% bar on its own. Together they might clear it, if their signals are partly independent.

| Near-miss | Alone | Folds won | Runtime cost |
|---|---|---|---|
| Regressor cap 5% (`cap_quantile` 0.05 instead of 0.005) | −0.83% | 5 of 6 | none |
| WSA speed for the forecast hour (`wsa_speed_lead_0h`) | −0.75% | 4 of 6 | HUX from the PFSS rings at runtime |
| Bayes parr + mean (latent fit) | −0.41% | 4 of 6 | the Bayes model's numpy features |
| Density window 1 h instead of 24 h (`density_mean_1h`) | −0.31% | 4 of 6 | none |

The protons (−0.30%, 2 of 6) are left out: they fail the near-miss rule, and the live proton feed isn't in the image.

**Fixed before running:**
- **Candidates, in this order:**
  - (a) cap 5% + density 1 h, the two that cost nothing at runtime;
  - (b) (a) + WSA;
  - (c) (b) + Bayes parr + mean.
- **The rule:** more than 1% better than the submitted model on the six folds' geometric mean, winning at least 4 of 6, with slow-wind runs not more than 1% worse.
- **Picking:** the smallest set that passes wins. Nothing else is tried.
- **Selection bias:** the parts were chosen on these same folds, so the result is optimistic. Report each fold's gain next to the leave-that-fold-out choice. If the chosen set would change when a fold is left out, say so.

**Before it ships:**
- run the full test suite and `ci/check_submission.py`;
- tag `upload_candidate/<name>` and run it in staging for at least a day;
- compare its live runtime with the 900 s limit, and its failed or missed cycles with the submitted model's.

A failed run scores as persistence (+33% for that run), so live reliability counts for more than a 0.5% gain. If (b) or (c) passes but (a) doesn't, consider whether the extra runtime and the WSA and Bayes code paths are worth shipping two days before the freeze.

**Result (2026-11-17, [near_miss.md](near_miss.md)):**
- **On the six folds,** all three candidates pass; the rule picks the smallest, (a): −1.08%, 5 of 6, slow-wind runs +0.8%. (b) is −1.42% and (c) −1.92%, both 5 of 6.
- **(a) is borderline:** without 2021 or 2022, the rule picks (b).
- **On the eval year,** (a) is 0.45% worse than the submitted model: better at stream arrivals (−3.3%) and in fast streams (−2.6%), worse in slow wind (+3.4%) and CME runs (+1.4%).
- **Decision (user):** not submitted; (a) runs in staging as `upload_candidate/near-miss-a`.

### 1b. Build (b) and (c) (after the freeze)

(b) and (c) pass the fold rule with larger gains than (a), and they improve slow-wind runs, where (a) and the eval year are weakest. They weren't picked because each step beyond (a) wins only 3 of 6 folds, and because they need new runtime code. Building them makes them testable live, in staging, next to (a) and the submitted model.

**(b): (a) + the WSA speed. Built 2026-11-17** ([near_miss.md](near_miss.md#candidate-b-built-2026-11-17)): eval year −0.50% against the submitted model; in staging as `upload_candidate/near-miss-b`. Steps 1–5 below are done; step 6 is running.
1. **Runtime:** a `models.wsa` path in the live run. Each run already solves the newest GONG map for the PFSS rings (`inference.fetch_inputs`, `feature_engineering.pfss`). Add the WSA source speed and HUX propagation to Earth from those rings, read at the lead-aligned lags (`wsa_speed_lead_0h`). Use the offline calibration fitted on every training run, stored in an asset.
2. **Calibration:** `training.wsa_model`'s offline calibration on all training runs (2016-10 → 2025-10). It is cached per fold now; fit it once more on everything.
3. **Fallbacks:** a run without a GONG map or ring gets the training median, as the PFSS current-sheet distance does now (`training.linear_model.with_pfss`).
4. **No future data:** the WSA column uses only the ring available at t0; extend `tests/test_features.py`'s scrambling check to it.
5. **Retrain** the linear model with (b)'s config, score the eval year once (descriptive), run the tests and the contract checks, and time a live run (`inference.check_features`).
6. **Staging:** tag `upload_candidate/near-miss-b`, at least several days next to (a).

**(c): (b) + the Bayes parr and mean.**
1. **Runtime:** the code exists. `models.bayes_model` computes parr and mean in numpy from 32 posterior draws, and `SWP_MODEL=bayes` runs it in staging. (c) is the same with (b)'s regressors and cap.
2. **Refit:** the constrained latent fit (`training.bayes_model train --fit latent_constrained`, ~45 min, in a terminal) on (b)'s regressors, since the draws depend on the stream model's inputs. Then refit the linear part with (c)'s config.
3. **Check on the folds first:** the six-fold (c) used the plain latent fold fits; rerun `training.near_miss` with the constrained fold fits (~2 h per fold) before trusting −1.92% for the shipped version.
4. **Then** the eval year once, the tests, and staging as `upload_candidate/near-miss-c`.

**Decide from staging,** not the folds alone: (a), (b) and (c) passed the same fold rule, and the eval year disagreed for (a). A week or more of live scores, next to the submitted model, is the evidence that's left.

### 2. Gated linear / Bayes-official combination (after the freeze)

**The idea:** forecast F_k = (1 − g_k)·linear_k + g_k·bayes_official_k for each lead k. The gate g_k = σ(β_k · [1, P(arrival by k), last speed, ambient level]) comes from the same posterior. This uses the Bayes model where it is strong (slow and intermediate wind) and keeps the linear model at arrivals. The gate rests on physically meaningful quantities instead of a separately trained classifier, the weak point of the earlier regime routing.

**Steps:**
1. Refit the constrained latent fit on each of the six folds (it converged on all the training data; ~45 min to 2 h per fold, in your terminal), with draws cached per fold.
2. Compute per fold, from the cached posteriors: the official-decision path, P(arrival by lead) and the ambient level, for training and validation runs.
3. Fit β per lead on the training runs only (official metric; a smooth penalty across leads), and score on each fold.
4. Pilot on the 2024 fold: time it and report; continue only if it is within reach of the bar.
5. Six folds, the same rule as step 1, and a breakdown by regime.

**Risks:**
- The arrival probability's skill is modest (AUC 0.69), so the gate may hand over runs where an arrival actually comes.
- The official decision needs posterior path simulation at runtime: more numpy work per run, though it is well inside the time limit.

### 3. Bayes official path as a regressor

This is a cheaper approximation of step 2: add the official-decision path (6% better than the posterior mean on the pilot) to the linear model, beside parr and mean. Run it with `training.bayes_features` once the constrained fold fits exist. It is unlikely to beat step 2, because a linear coefficient can't switch the path on and off by regime.

### 4. Residual diffusion model (after the freeze)

**The idea:** a conditional diffusion model that generates the 72-hour residual of the linear forecast. The inputs are the speed history and the linear model's regressors. It runs on CPU at about 1M parameters, with torch `+cpu` and the weights in `assets/`. Sampling 100 paths takes seconds.

**For a fair test:**
- **A direct neural control:** the same network and inputs, trained with MSE to output 72 values in one go. It separates "a neural net helps" from "diffusion helps".
- **Two decision rules:** the sample mean, and the path minimising the expected official score over the samples.
- **A pilot on 2024 first,** timed, before six folds.

**Risks:**
- Only about 3,300 independent windows, few of them stream arrivals.
- The metric scores a single path, which averages away the samples' sharpness.
- The linear model's record against richer models.

A separate plan comes before any build.

### 5. Is the slow-wind bias optimal for the metric?

The loss blend trades slow-wind accuracy against active-wind accuracy one for one. That suggests the 37 km/s over-forecast may be what the official percentile-TIMSE weights reward.

**Test:** on the six folds, subtract a constant bias in slow-wind runs, defined by the truth (an oracle). See whether S_final improves. Then repeat with a realistic slow-wind indicator.
- **If even the oracle doesn't help:** the bias is optimal for the metric. Stop chasing it, and drop slow-wind work from future plans.
- **If the oracle helps but the realistic version doesn't:** the problem is detection, which points back to step 2's gate.
- **Untested either way:** a target transform such as log speed.

**Result (2026-11-17, [slow_bias.md](slow_bias.md)):** by the reading fixed before running, the oracle passes and the realistic indicator fails: the problem is detecting slow runs.
- **The oracle:** shifting only the true slow-wind runs down, by a shift chosen on the training runs, is 1.8% better (one shift) to 2.5% better (per lead) on all six folds, through RMSE, with TRMSE unchanged. So the metric doesn't reward the over-forecast.
- **Last speed < 450:** it chooses no shift; only 41% of the runs it flags stay slow.
- **A global shift:** worse (+3% at 20 km/s); the current overall lean is right.
- **The ceiling is about 2–2.5%** overall, even with perfect detection. That bounds the slow-wind part of step 2, whose gate is a realistic detector.

### 6. Archive live WSA-Enlil

`json/enlil_time_series.json` on SWPC is the only live input that forecasts CME arrivals. It can't be trained on, because there is no archive ([features.md](features.md)). A scheduled pull every 6 hours, stored with the staging records, would build the archive for future model work. It doesn't help this competition.

## Doors that look closed

These were tried, are documented, and showed no realistic path to the bar. Don't revisit them without a new reason:
- recency weights ([recency_weights.md](recency_weights.md));
- quantile mapping ([quantile_mapping.md](quantile_mapping.md));
- post-hoc calibration ([post_calibration.md](post_calibration.md));
- interaction terms ([interactions.md](interactions.md));
- HMM regime features and switching ([regime_hmm.md](regime_hmm.md), [regime_switching.md](regime_switching.md));
- GP corrections ([gaussian_process.md](gaussian_process.md));
- coronagraph and magnetogram image features ([coronagraph.md](coronagraph.md), [cme_detector.md](cme_detector.md), [magnetogram.md](magnetogram.md));
- the coronal-hole fallbacks ([ch_fallback.md](ch_fallback.md));
- the standalone structural, stream-event and Bayes models: 4–10% worse alone ([structural_model.md](structural_model.md), [stream_events.md](stream_events.md), [bayes_stream.md](bayes_stream.md)).

## Order of work

1. **Done 2026-11-17:** step 1 ran; (a) is in staging, not submitted ([near_miss.md](near_miss.md)). Step 5 ran: the over-forecast is a detection problem ([slow_bias.md](slow_bias.md)).
2. **Done 2026-11-17:** the constrained fit converged; [bayes_model.md](bayes_model.md) and PR #57 are updated. The fold fits for steps 2 and 3 can start in your terminal when the laptop is free.
3. **By 19 Nov:** the submitted model stays (step 1's (a) didn't hold on the eval year).
4. **After the freeze:** step 1b ((b), then (c)), with staging deciding between them; step 2, then step 3; step 4 if step 2 doesn't pass. Set up step 6 at any time.
