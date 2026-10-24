# Review before further model development (2026-10-23)

**Why this review.** After the linear model, the XGBoost hybrid, a Gaussian process and three
regime studies, the gains have flattened: the hybrid gains about 3% over the linear model, and
every later idea came out flat ([evaluation.md](evaluation.md), "Studies that were not built").

**Six questions:**
1. are the regressors and the target aligned in time?
2. are useful features dropped too early?
3. which physical drivers are missing?
4. should image embeddings be added?
5. are time-series foundation models or attention models worth it?
6. what do the papers in `docs/literature/` do that we don't?

**Short answer.** The remaining error looks like missing *information*, not too simple a model:
- coronal magnetic-field structure;
- coronal-hole geometry;
- advance warning of CMEs.

There are also a few timing details to fix or confirm.

**Papers read:**

| Paper | What it is |
|---|---|
| Brown et al. 2022, *Attention-based machine vision models … using solar EUV images* (Space Weather 20) | a Swin transformer on single AIA 211 Å images, 4-day horizon |
| Collin et al. 2025, *Forecasting High-Speed Solar Wind Streams from Solar Images* (arXiv 2410.05068) | coronal-hole area on a grid + 27-day persistence, polynomial regression |
| Bailey et al. 2021, *Using gradient boosting regression to improve ambient solar wind model predictions* (Space Weather 19) | XGBoost on WSA magnetic-model features |
| Lampani et al. 2025, *Predicting CME travel times using enhanced model-guided machine learning* (arXiv 2512.19492) | a drag-based physics model plus a neural network for CME arrival |

## 1. Timing: regressors and target

**Sound:**
- **Same place, same instrument.** The briefing names the target: ACE SWEPAM speed at L1. Our target and every in-situ regressor are ACE's real-time L1 data, so no Sun–Earth or bow-shock shift is needed. The papers use OMNI, which is shifted to the bow shock; we correctly don't.
- **No future data.** Each source has an availability rule: in-situ data up to t0 − 10 min, AIA images up to t0 − 1 h, flares by peak time, F10.7 from the previous day. `tests/test_features.py` checks that scrambling future data never changes a regressor.

**To fix or confirm:**

| Issue | Effect | Action |
|---|---|---|
| **The speed input is cut at t0, the other regressors at t0 − 10 min.** The last speed hour in training includes minutes that live runs can't see yet (open issue in [features.md](features.md)). | small train/live mismatch on the strongest input | cut the speed inputs at t0 − 10 min like the other regressors, and refit |
| **Hour convention.** Lead k is compared with the hour *starting* at t0 + k ([t0 + k, t0 + k + 1 h)). If Benchlab means the hour *ending* at t0 + k, every forecast is an hour late. | largest at stream and CME onsets, which TIMSE weights heavily | ask Benchlab; meanwhile measure the cost on the folds |
| **Solar inputs use fixed lags from t0.** Coronal-hole areas are taken 0–5 days before t0. The relevant image for lead k is about 4 days before t0 + k, and the travel time depends on speed (fast wind 2–3 days, slow 4–5). Collin et al. align coronal-hole features to the target time (4–7 days before it); the 4–5 day images matter most. | per-lead linear coefficients only partly compensate; the linear model keeps one lag for all leads | lead-aligned coronal-hole features |
| *Measured 2026-10-23 (six folds, shipped linear model and hybrid):* | 2a: the last speed hour built from its first 50 minutes (as live) changes the fold score by +0.09% (linear) / +0.08% (hybrid), and the lead-1 forecast by 1.3 km/s on average. 2b: scored against hour-ending truth, our forecasts lose 0.17% / 0.23% to the same forecasts shifted by one hour, with the sign varying by fold. | negligible: no refit needed; the convention question to Benchlab is low priority |
| **Flare timing.** Training counts flares by peak time (NCEI science lists). The live SWPC 7-day list may only show a flare after it ends. | tens of minutes; negligible at these horizons | verify the live list's behaviour |

## 2. Feature pruning

- **The linear model already prunes as part of tuning.** Forward selection over all 120 windowed candidates is scored on the six folds.
- **The hybrid prunes before tuning.** Its candidate sets were built from the linear model's *single-feature* ranking (the best window per family, set C).
  - **The risk:** a regressor that only helps through thresholds or interactions can be dropped before the trees see it. The flare counts and the CME share are typical examples.
  - **The full set D (all 120)** was not run on 9 years, because it needs 3–4 GB.

**Proposals:**
- **A forward-add test for the hybrid:** add each dropped family's best window to the shipped 30, one at a time, and keep what gains more than 0.5% on the folds. It fits in memory and directly answers "was something useful dropped?".
- **Pruning as a tuned setting for the trees:** rank by importance within each fold's training data, and choose how many to keep on the folds. This is the same idea as Collin et al.'s Lasso-then-model step, done inside the cross-validation.

## 3. Missing physical drivers (by expected value)

1. **The coronal magnetic field.**
   - **Evidence:** Bailey et al. find the flux-tube expansion factor and the distance to the coronal-hole boundary (from magnetograms via PFSS/WSA) the most decisive inputs. With 27-day persistence they reach RMSE 78 km/s against 98 km/s for persistence alone (2006–2017).
   - **Our gap:** no magnetic-field input at all.
   - **Feasibility:** GONG synoptic magnetograms have an archive and a live feed, and a potential-field model is CPU-feasible. This is the largest build on the list.
2. **Coronal-hole geometry beyond one strip:**
   - area on a latitude × longitude grid. Collin et al. find central, low-latitude cells matter most; finer latitude bands help the peak speed;
   - Earth's heliographic latitude B0 (±7° over the year, computable from the date), with a hemisphere-asymmetry term;
   - hole darkness: darker holes give faster streams (Obridko et al. 2009, also learned by Brown et al.'s model);
   - two-channel detection (193 + 211 Å, k-means) instead of 193 Å alone.
3. **Advance warning of CMEs,** the largest error source. No input sees a CME before it reaches L1 ([regime_models.md](regime_models.md)).
   - **Coronagraph images:** LASCO C2/C3 real-time images are on `soho.nascom.nasa.gov`, already a declared domain. A halo-CME brightness feature could give 1–3 days of warning. Its history and latency need checking.
   - **The drag-based arrival model** (Lampani et al., MAE about 13 h) needs CME speed, mass and width from a coronagraph catalogue. Those catalogues are published too late (LASCO CDAW) or unreachable (DONKI), so it isn't usable live as is.
4. **Energetic protons and cycle phase:**
   - GOES ≥ 10 MeV protons, a marker of an Earth-connected CME shock, with a live feed and an archive. We have only ACE EPAM at lower energies;
   - the solar-cycle trend (the slope of the 81-day F10.7 mean), not only its level. Collin et al. find the input–speed relationships change with cycle phase.

## 4. Image embeddings

**Worth a time-boxed test, but after better engineered coronal-hole features.**
- **Engineered features already match deep models:** Collin et al.'s few features (coronal-hole area on a grid plus 27-day persistence) match or beat Brown et al.'s 29-million-parameter transformer (RMSE about 68–70 km/s, 2010–2018).
- **Data is the limit:** about 3,300 training days, highly correlated (roughly 120 solar rotations). Embeddings would need to be compressed hard, to 10–20 principal components.
- **Feasible:** embedding 14,000 historical images on CPU takes hours, and one image live takes about a second. A solar-pretrained encoder should beat an ImageNet one. The browse-JPEG archive's gaps carry over.

## 5. Time-series foundation models and attention models

- **Foundation models** (Chronos-2, TimesFM, Moirai): cheap to benchmark zero-shot on the folds. On the speed history alone they should land near the final baseline (0.484), because the gains so far came from the solar and in-situ regressors. Versions that accept covariates are the ones worth trying. CPU-only torch in the image is allowed, at a size and pull-time cost.
- **Attention or sequence models trained from scratch:** low expected value.
  - **Evidence:** every model change after the hybrid came out flat: the Gaussian process, regime routing, HMM features and regime switching.
  - **Data:** with about 120 independent rotations, capacity isn't the bottleneck.

## 6. What the literature does that we don't

| In the papers | Here | Gap |
|---|---|---|
| Coronal-hole features timed against the target (4–7 days before it) | timed against t0 | lead-aligned features |
| Coronal-hole grid, Earth's B0, darkness, 193 + 211 Å | one strip, 193 Å | richer coronal-hole features |
| Magnetic-model features (expansion factor, distance to hole boundary) | none | the main physics gap |
| Persistence of v, density, pressure, temperature and Bt at 26, 27 and 28 days before the target | speed recurrence per lead; density, pressure and polarity 27 days earlier | add Bt and temperature; the 26/28-day offsets |
| Distribution correction: MSE shrinks the peaks (Collin et al.: peak RMSE −22%) | TIMSE-weighted fitting, but still over-forecasting quiet wind and under-forecasting storms | a quantile-mapping post-step, chosen on the folds (tested: worse, see [quantile_mapping.md](quantile_mapping.md)) |
| Recalibration to the solar-cycle phase | the old years didn't help at solar maximum ([evaluation.md](evaluation.md)) | recency or cycle-similarity sample weights, chosen on the folds (recency tested: worse, see [recency_weights.md](recency_weights.md)) |
| Event metrics (stream hit, miss, peak error) | regime breakdowns | a diagnostic only; TIMSE is the score |

The papers' RMSE values (68–80 km/s) exclude CMEs (Collin et al.) or cover other periods, so
they aren't comparable with our eval-year RMSE (about 100 km/s, solar maximum, CMEs included).

## Plan (live evaluation starts 2026-11-11)

1. **Decide which model ships.** The submission still runs the final baseline by default; the hybrid beats it by 14% on the per-run mean.
2. **Quick, low-risk fixes** (branch `quick_fixes`):
   - a. cut the speed inputs at t0 − 10 min, and refit;
   - b. measure the cost of the one-hour alternative convention on the folds, and ask Benchlab which convention they use;
   - c. cap the regressors at a training quantile (the 61 nT Bt run of 2026-01-20). *Done 2026-10-24:* q = 0.005, −0.27% (linear) / −0.11% (hybrid) on the folds; shipped ([linear_model.md](linear_model.md), data fix 3);
   - d. a quantile-mapping post-step. *Done 2026-10-24:* not built; worse at every blend strength, because it removes the upward offset that TIMSE weighting makes optimal ([quantile_mapping.md](quantile_mapping.md));
   - e. recency or cycle-phase sample weights. *Recency done 2026-10-24:* not built; worse at every half-life, and the F10.7 level doesn't explain which folds gain ([recency_weights.md](recency_weights.md)).

   Each is chosen on the folds.
3. **Feature work:**
   - lead-aligned coronal-hole features;
   - B0 and a latitude grid;
   - the hybrid forward-add test.
4. **Larger builds, if time allows:** the PFSS magnetic-field features, or a coronagraph CME feature.
